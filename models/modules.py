import torch
from einops import rearrange, repeat
from torch import nn, einsum
import math


class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden_dim),
                                 GELU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(hidden_dim, dim),
                                 nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout)) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = dots.softmax(dim=-1)               
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([Residual(PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout))),
                                              Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)))]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x)
            x = ff(x)
        return x
    
    
###########################################################
############# output = mean of the all tokens #############
###########################################################
class Temporal_Transformer_Mean(nn.Module):
    def __init__(self, num_patches, input_dim, depth, heads, mlp_dim, dim_head):
        super().__init__()
        dropout=0.0
        self.num_patches = num_patches
        self.input_dim = input_dim
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, input_dim))
        self.temporal_transformer = Transformer(input_dim, depth, heads, dim_head, mlp_dim, dropout)

    def forward(self, x):
        x = x.contiguous().view(-1, self.num_patches, self.input_dim)
        b, n, _ = x.shape
        x = x + self.pos_embedding[:, :n]
        x = self.temporal_transformer(x)
        x = x.mean(dim=1)
        return x

###########################################################
#############      output = class tokens      #############
###########################################################
class Temporal_Transformer_Cls(nn.Module):
    def __init__(self, num_patches, input_dim, depth, heads, mlp_dim, dim_head):
        super().__init__()
        dropout=0.0
        self.num_patches = num_patches
        self.input_dim = input_dim
        self.cls_token = nn.Parameter(torch.randn(1, 1, input_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches+1, input_dim))
        self.temporal_transformer = Transformer(input_dim, depth, heads, dim_head, mlp_dim, dropout)

    def forward(self, x):
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embedding[:, :(n+1)]
        x = self.temporal_transformer(x)
        return x
    
###########################################################
#############        output = all tokens      #############
###########################################################
class Temporal_Transformer_All(nn.Module):
    def __init__(self, num_patches, input_dim, depth, heads, mlp_dim, dim_head):
        super().__init__()
        dropout=0.0
        self.num_patches = num_patches
        self.input_dim = input_dim
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, input_dim))
        self.temporal_transformer = Transformer(input_dim, depth, heads, dim_head, mlp_dim, dropout)

    def forward(self, x):
        x = x.contiguous().view(-1, self.num_patches, self.input_dim)
        b, n, _ = x.shape
        x = x + self.pos_embedding[:, :n]
        x = self.temporal_transformer(x)
        return x
    
class Multi_Modal_Transformer_With_Modality(nn.Module):
    def __init__(self, num_patches=16, input_dim=768, depth=6, heads=8, mlp_dim=3072, dim_head=64):
        super().__init__()
        
        self.num_patches = num_patches
        self.num_patches_total = num_patches + 1
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, input_dim))
        
        # **新增**: 创建音频和视频的模态嵌入
        self.audio_modality_embedding = nn.Parameter(torch.randn(1, 1, input_dim))
        self.video_modality_embedding = nn.Parameter(torch.randn(1, 1, input_dim))

        # 位置编码现在是通用的，模态信息由上面的嵌入提供
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches_total, input_dim))

        self.fusion_transformer = Transformer(input_dim, depth, heads, dim_head, mlp_dim, dropout=0.1)


    def forward(self, audio_tokens, video_tokens):
        b = audio_tokens.shape[0]
        
        # **步骤 1: 为每个模态添加自己的模态嵌入**
        # 广播机制会自动将 [1, 1, D] 的模态嵌入加到 [B, T, D] 的每个Token上
        audio_tokens = audio_tokens + self.audio_modality_embedding
        video_tokens = video_tokens + self.video_modality_embedding

        # 步骤 2: 拼接
        x = torch.cat((audio_tokens, video_tokens), dim=1)

        # 步骤 3: 添加CLS Token
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # 步骤 4: 添加位置编码
        x = x + self.pos_embedding

        # 步骤 5 & 6 & 7: 和方案一相同
        fused_output = self.fusion_transformer(x)
        fused_cls_token = fused_output[:, 0]

        return fused_cls_token
    
    
class CTCModule(nn.Module):
    def __init__(self, in_dim, out_seq_len):
        '''
        这个模块执行从模态A（例如音频）到模态B（例如视频）的对齐。
        :param in_dim: 输入模态A的特征维度
        :param out_seq_len: 输出模态B的序列长度
        '''
        super(CTCModule, self).__init__()
        # 使用LSTM来预测从A到B的位置对齐
        # LSTM的输出维度为out_seq_len + 1，其中+1表示“空白”标记
        self.pred_output_position_inclu_blank = nn.LSTM(in_dim, out_seq_len + 1, num_layers=2, batch_first=True)
        self.out_seq_len = out_seq_len
        # Softmax层用于将LSTM的输出转换为概率分布
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        '''
        :param x: 输入张量，形状为[batch_size, in_seq_len, in_dim]
        '''
        # 通过LSTM网络，得到对齐位置的预测（包括“空白”标记）
        pred_output_position_inclu_blank, _ = self.pred_output_position_inclu_blank(x)
        # 应用Softmax，得到对齐到每个位置（包括“空白”）的概率
        prob_pred_output_position_inclu_blank = self.softmax(pred_output_position_inclu_blank)  # [batch_size, in_seq_len, out_seq_len + 1]
        # 移除“空白”标记的概率，得到对齐到有效输出位置的概率
        prob_pred_output_position = prob_pred_output_position_inclu_blank[:, :, 1:]  # [batch_size, in_seq_len, out_seq_len]
        # 转置张量，以便于后续的矩阵乘法
        prob_pred_output_position = prob_pred_output_position.transpose(1, 2)  # [batch_size, out_seq_len, in_seq_len]
        # 使用批量矩阵乘法，计算伪对齐的输出特征
        pseudo_aligned_out = torch.bmm(prob_pred_output_position, x)  # [batch_size, out_seq_len, in_dim]
        return pseudo_aligned_out
    
class AlignSubNet(nn.Module):
    def __init__(self, in_dim_a, in_dim_v, seq_len_a, seq_len_v, mode):
        """
        mode: the way of aligning
            avg_pool, ctc, conv1d
        """
        super(AlignSubNet, self).__init__()
        assert mode in ['avg_pool', 'ctc', 'conv1d']

        # in_dim_a, in_dim_v = 768, 768
        # seq_len_a, seq_len_v = 256, 16
        self.dst_len = seq_len_v  # Align to video sequence length
        self.mode = mode

        self.ALIGN_WAY = {
            'avg_pool': self.__avg_pool,
            'ctc': self.__ctc,
            'conv1d': self.__conv1d
        }

        if mode == 'conv1d':
            self.conv1d_A = nn.Conv1d(seq_len_a, self.dst_len, kernel_size=1, bias=False)
        elif mode == 'ctc':
            self.ctc_a = CTCModule(in_dim_a, self.dst_len)

    def get_seq_len(self):
        return self.dst_len

    def __ctc(self, audio_x):
        audio_x = self.ctc_a(audio_x) if audio_x.size(1) != self.dst_len else audio_x
        return audio_x

    def __avg_pool(self, audio_x):
        def align(x):
            raw_seq_len = x.size(1)
            if raw_seq_len == self.dst_len:
                return x
            if raw_seq_len // self.dst_len == raw_seq_len / self.dst_len:
                pad_len = 0
                pool_size = raw_seq_len // self.dst_len
            else:
                pad_len = self.dst_len - raw_seq_len % self.dst_len
                pool_size = raw_seq_len // self.dst_len + 1
            pad_x = x[:, -1, :].unsqueeze(1).expand([x.size(0), pad_len, x.size(-1)])
            x = torch.cat([x, pad_x], dim=1).view(x.size(0), pool_size, self.dst_len, -1)
            x = x.mean(dim=1)
            return x
        audio_x = align(audio_x)
        return audio_x

    def __conv1d(self, audio_x):
        audio_x = self.conv1d_A(audio_x.transpose(1, 2)).transpose(1, 2) if audio_x.size(1) != self.dst_len else audio_x
        return audio_x

    def forward(self, audio_x, video_x):
        # already aligned
        if audio_x.size(1) == video_x.size(1):
            return audio_x, video_x
        aligned_audio = self.ALIGN_WAY[self.mode](audio_x)
        return aligned_audio, video_x
    
    