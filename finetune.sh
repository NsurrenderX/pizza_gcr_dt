export NCCL_IB_HCA=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_7:1,mlx5_8:1,mlx5_9:1
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=en,eth,em,bond,ib #bond0
export NCCL_DEBUG=INFO
export NCCL_NVLS_ENABLE=0

export PRETRAIN_MODEL_PATH="/mnt/wangxiaofa/RDT_module_params/rdt_param/rdt-170m/"

export VISION_ENCODER_NAME="/mnt/wangxiaofa/RDT_module_params/rdt_param/siglip-so400m-patch14-384"
export TEXT_ENCODER_NAME="/mnt/wangxiaofa/RDT_module_params/rdt_param/t5-v1_1-xxl"
# export OUTPUT_DIR="/mnt/wangxiaofa/rdt_checkpoint/170M_action_chunk_repetive_padding/"
export OUTPUT_DIR="/mnt/wangxiaofa/rdt_checkpoint/170M_test_ac16_bs32x8/"
# export PRETRAIN_MODEL_PATH="/mnt/wangxiaofa/RDT_module_params/rdt_param/rdt-1b/"

# export TEXT_ENCODER_NAME="/datahdd_8T/vla_pizza/RDT_module_params/t5-v1_1-xxl/"
# export VISION_ENCODER_NAME="/datahdd_8T/vla_pizza/RDT_module_params/siglip-so400m-patch14-384/"
# export PRETRAIN_MODEL_PATH="/datahdd_8T/vla_pizza/RDT_module_params/rdt-170m/"
# export PRETRAIN_MODEL_PATH="/datahdd_8T/vla_pizza/RDT_module_params/rdt-1b/"
# export OUTPUT_DIR="/datahdd_8T/vla_pizza/rdt_checkpoint/170M_test_16chunk/"
# export OUTPUT_DIR="/datahdd_8T/vla_pizza/rdt_checkpoint/1000M-fp32/"
export CFLAGS="-I/usr/include"
export LDFLAGS="-L/usr/lib/x86_64-linux-gnu"
export CUTLASS_PATH="/path/to/cutlass"

export WANDB_PROJECT="robotics_diffusion_transformer"

if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir "$OUTPUT_DIR"
    echo "Folder '$OUTPUT_DIR' created"
else
    echo "Folder '$OUTPUT_DIR' already exists"
fi

# For run in a single node/machine
# accelerate launch main.py \
#     --deepspeed="./configs/zero2.json" \
#     ...

deepspeed --hostfile=hostfile.txt main.py \
    --deepspeed="./configs/zero2fp32.json" \
    --pretrained_model_name_or_path=$PRETRAIN_MODEL_PATH \
    --pretrained_text_encoder_name_or_path=$TEXT_ENCODER_NAME \
    --pretrained_vision_encoder_name_or_path=$VISION_ENCODER_NAME \
    --output_dir=$OUTPUT_DIR \
    --train_batch_size=4 \
    --sample_batch_size=4 \
    --max_train_steps=300000 \
    --checkpointing_period=2000 \
    --sample_period=2000 \
    --checkpoints_total_limit=100 \
    --lr_scheduler="constant" \
    --learning_rate=1e-4 \
    --mixed_precision="bf16" \
    --dataloader_num_workers=10 \
    --image_aug \
    --dataset_type="finetune" \
    --state_noise_snr=40 \
    --load_from_hdf5 \
    --report_to=tensorboard \
    --precomp_lang_embed \
    --resume_from_checkpoint="latest"
    # --mixed_precision="bf16" \

    # Use this to resume training from some previous checkpoint
    # --resume_from_checkpoint="checkpoint-36000" \
    # Use this to load from saved lanuage instruction embeddings,
    # instead of calculating it during training
    # --precomp_lang_embed \
