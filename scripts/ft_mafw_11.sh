export TRANSFORMERS_OFFLINE=1

server=170
pretrain_dataset='clip+clap/baseline'
# dataset
finetune_dataset='MAFW'
num_labels=11
# model
models=(clipb16_clap_simple_concat_tv)
# input
input_size=224
input_size_audio=1024
sr=4
# parameter
lr=1e-5
epochs=100
batch_size=8
device=0,1
splits=(1 2 3 4 5)
for model in "${models[@]}";
do
for split in "${splits[@]}";
do
    # output directory
    OUTPUT_DIR="./saved/model/finetune/${finetune_dataset}11/audio_video/${pretrain_dataset}/${model}/eval_split0${split}_lr_${lr}_epoch_${epochs}_bs_${batch_size}_size${input_size}_a${input_size_audio}_sr${sr}"
    if [ ! -d "$OUTPUT_DIR" ]; then
      mkdir -p $OUTPUT_DIR
    fi
    echo "Save dir: ${OUTPUT_DIR}"
    # path to split files
    DATA_PATH="./data/MAFW/audio_visual/single/split0${split}"

    CUDA_VISIBLE_DEVICES=$device python \
        train.py \
        --model ${model} \
        --data_set ${finetune_dataset^^} \
        --nb_classes ${num_labels} \
        --data_path ${DATA_PATH} \
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --batch_size ${batch_size} \
        --num_sample 1 \
        --input_size ${input_size} \
        --short_side_size ${input_size} \
        --save_ckpt_freq 1000 \
        --num_frames 16 \
        --sampling_rate ${sr} \
        --opt adamw \
        --lr ${lr} \
        --opt_betas 0.9 0.999 \
        --weight_decay 0.05 \
        --epochs ${epochs} \
        --dist_eval \
        --test_num_segment 2 \
        --test_num_crop 2 \
        --num_workers 8 \
        --layer_decay 1 \
       >>${OUTPUT_DIR}/nohup.out 2>&1
done
done
echo "Done!"

