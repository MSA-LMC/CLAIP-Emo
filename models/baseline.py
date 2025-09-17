
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPVisionModel, ClapAudioModelWithProjection, CLIPProcessor
from transformers.models.clip.modeling_clip import CLIPEncoderLayer, CLIPMLP
from torch.nn import init
from peft import get_peft_model, LoraConfig, TaskType
from einops import rearrange
import os
from einops import rearrange


from .modules import *
from .htsat import CLAP_Audio_Encoder
from .fusion_modules import *

try:
    from timm import register_model
except:
    from timm.models import register_model


class CLIP_CLAP(nn.Module):
    def __init__(self, lora_r=8, lora_alpha=32, lora_dropout=0.1,
                 num_classes=7,
                 audio_encoder="laion-clap",
                 vision_encoder="openai/clip-vit-base-patch16"):
        super(CLIP_CLAP, self).__init__()

        # 根据模态选择性地初始化编码器
        self.vision_encoder = self.build_vision_encoder(lora_r=lora_r, lora_alpha=lora_alpha,
                                                        lora_dropout=lora_dropout, model_name=vision_encoder)

        self.audio_encoder = self.build_audio_encoder(lora_r=lora_r, lora_alpha=lora_alpha,
                                                      lora_dropout=lora_dropout, model_name=audio_encoder)
        embdim = 1024 if 'large' in vision_encoder else 768

        self.temporal_v = Temporal_Transformer_Cls(
            num_patches=16,
            input_dim=embdim,
            depth=1,
            heads=8,
            mlp_dim=1024,
            dim_head=64
        )
        input_dim = embdim + 768
        self.fusion_module = ConcatFusion(
            output_dim=num_classes, input_dim=input_dim)
        self.print_trainable_parameters()

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    @torch.jit.ignore
    def get_num_layers(self):
        return 1

    def print_trainable_parameters(self):
        sum_param = sum(p.numel() for p in self.parameters()) / 1e6
        learning_params = sum(p.numel()
                              for p in self.parameters() if p.requires_grad) / 1e6
        print(
            f"Total parameters: {sum_param:.2f}M, Learning parameters: {learning_params:.2f}M | {learning_params/sum_param:.2%}")

    def build_vision_encoder(self, lora_r=8, lora_alpha=32, lora_dropout=0.1, model_name="openai/clip-vit-base-patch16"):
        print(
            f"Building vision encoder: {model_name} with LoRA config: r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")
        # Initialize the vision encoder
        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            inference_mode=False,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"])
        encoder = CLIPVisionModel.from_pretrained(model_name)
        encoder = get_peft_model(encoder, lora_config)
        return encoder.base_model

    def build_audio_encoder(self, lora_r=8, lora_alpha=32, lora_dropout=0.1, model_name="laion/clap-htsat-fused"):

        encoder = self.build_hsat_encoder(lora_r=lora_r, lora_alpha=lora_alpha,
                                          lora_dropout=lora_dropout, model_name=model_name)
        return encoder

    def build_hsat_encoder(self, lora_r=8, lora_alpha=32, lora_dropout=0.1, model_name="laion/clap-htsat-fused"):
        print(
            f"Building audio encoder: {model_name} with LoRA config: r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")
        ckpt = '/data2/chenyin/DFER/checkpoints/CLAP/audio_branch/630k-audioset-fusion-best.pt'
        encoder = CLAP_Audio_Encoder(
            num_classes=0, enable_fusion=True, use_lora=True, ckpt=ckpt)
        return encoder

    def forward_video_features(self, x):
        # 视频输入: [B, C, T, H, W]
        b, c, t, h, w = x.shape
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.vision_encoder.vision_model(x)
        last_hidden_state = rearrange(
            x.last_hidden_state, '(b t) n d -> b t n d', b=b, t=t)  # [B, T, N, 768]

        x = rearrange(x.pooler_output, '(b t) d -> b t d',
                      b=b, t=t)  # [B, T, 768]

        return x, last_hidden_state  # [B, T, 768], [B, T, N, 768]

    def forward_audio_features(self, x):
        x, last_hidden_state = self.audio_encoder(x)

        return x, last_hidden_state

    def forward(self, x, a, return_features=False):
        v, hidden_v = self.forward_video_features(x)
        v = self.temporal_v(v)[:, 0]
        a, hidden_a = self.forward_audio_features(a)
        x, feat = self.fusion_module(a, v)
        if return_features:
            return x, feat
        return x


@register_model
def clipb32_clap_simple_concat_tv(num_classes=7, pretrained_cfg=None, pretrained=False, **kwargs):
    model = CLIP_CLAP(num_classes=num_classes,
                      audio_encoder="laion-clap",
                      vision_encoder="openai/clip-vit-base-patch32",
                      )
    return model


@register_model
def clipb16_clap_simple_concat_tv(num_classes=7, pretrained_cfg=None, pretrained=False, **kwargs):
    model = CLIP_CLAP(num_classes=num_classes,
                      audio_encoder="laion-clap",
                      vision_encoder="openai/clip-vit-base-patch16",
                      )
    return model


@register_model
def clipl14_clap_simple_concat_tv(num_classes=7, pretrained_cfg=None, pretrained=False, **kwargs):
    model = CLIP_CLAP(num_classes=num_classes,
                      audio_encoder="laion-clap",
                      vision_encoder="openai/clip-vit-large-patch14",
                      )
    return model


# 使用示例
if __name__ == "__main__":
    # 1. 音视频模态
    model_both = clipb32_clap_align_latefusion(num_classes=7, modality='both')
    v, a = torch.randn(2, 3, 16, 224, 224), torch.randn(2, 1, 1024, 64)
    y = model_both((v, a))
    print(f"Both modalities output: {y.shape}")  # [2, 7]

    # # 2. 仅视频模态
    # model_video = clipb32_clap_latefusion(num_classes=7, modality='video')
    # v = torch.randn(2, 3, 16, 224, 224)
    # y = model_video(v)
    # print(f"Video only output: {y.shape}")  # [2, 7]

    # # 3. 仅音频模态
    # model_audio = clipb32_clap_latefusion(num_classes=7, modality='audio')
    # a = torch.randn(2, 1, 1024, 64)
    # y = model_audio(a)
    # print(f"Audio only output: {y.shape}")  # [2, 7]

    # # 4. 使用字典输入（更灵活）
    # y = model_both({'video': v, 'audio': a})
    # print(f"Dict input output: {y.shape}")  # [2, 7]
