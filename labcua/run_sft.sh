#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# UI-TARS 1.5 7B 微调启动脚本（ms-swift SFT）
# 使用 conda 环境 sw，数据集为 traj_to_swift_dataset.py 生成的 traj_v2
#
# 用法:
#   ./run_sft.sh
#   CUDA_VISIBLE_DEVICES=0,1 USE_DEEPSPEED=1 ./run_sft.sh   # 多卡需先 pip install deepspeed
# -----------------------------------------------------------------------------

# set -e

# 环境配置
export HF_ENDPOINT='https://hf-mirror.com'
export HF_HOME='/data/250010149/hf_cache'
unset XDG_CACHE_HOME
export NVIDIA_DRIVER_CAPABILITIES="${NVIDIA_DRIVER_CAPABILITIES:-compute,utility}"
if [ -d /lib/x86_64-linux-gnu ]; then
    export LD_LIBRARY_PATH="/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

# 激活 conda 环境 sw（已 pip install -e . 安装 ms-swift）
# source "$(conda info --base)/etc/profile.d/conda.sh"
# conda activate sw

# 使用 HuggingFace 上的 base model（UI-TARS-1.5 基于 Qwen2.5-VL）
MODEL_ID="${MODEL_ID:-ByteDance-Seed/UI-TARS-1.5-7B}"
# 轨迹转换后的 SFT 数据（train.jsonl / val.jsonl）
TRAJ_V2_DIR="${TRAJ_V2_DIR:-/data/250010149/hf_cache/traj_v2}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/250010149/ms-swift/labcua/output/uitars15_7b_sft}"

# 单卡训练（无需 deepspeed）；多卡需先 pip install deepspeed，再设 CUDA_VISIBLE_DEVICES=0,1 和 USE_DEEPSPEED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
# 多卡时设为 1 并安装 deepspeed 后启用 --deepspeed zero2
USE_DEEPSPEED="${USE_DEEPSPEED:-1}"
# Qwen2.5-VL 图像分辨率上限，避免 OOM（参考 ms-swift 多模态示例）
export MAX_PIXELS="${MAX_PIXELS:-1003520}"


cd /data/250010149/ms-swift

# DeepSpeed 要求多进程（每 GPU 一进程），不能走 device_map。设置 NPROC_PER_NODE 后 swift 会自动用 torchrun 启动
if [ "$USE_DEEPSPEED" = "1" ]; then
    NGPU=$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | wc -l)
    [ "$NGPU" -lt 1 ] && NGPU=1
    export NPROC_PER_NODE=$NGPU
    echo "Using DeepSpeed with $NGPU GPU(s) (NPROC_PER_NODE=$NPROC_PER_NODE)"
fi

swift sft \
    --model "$MODEL_ID" \
    --model_type qwen2_5_vl \
    --template qwen2_5_vl \
    --use_hf true \
    --dataset "${TRAJ_V2_DIR}/train_v1.jsonl" \
    --val_dataset "${TRAJ_V2_DIR}/val_v1.jsonl" \
    --tuner_type lora \
    --torch_dtype bfloat16 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --freeze_vit true \
    --freeze_aligner true \
    --gradient_accumulation_steps 16 \
    --eval_steps 10 \
    --save_steps 20 \
    --save_total_limit 2 \
    --logging_steps 10 \
    --max_length 32768 \
    --eval_on_start true \
    --eval_metric cua \
    --predict_with_generate true \
    --max_new_tokens 1024 \
    --output_dir "$OUTPUT_DIR" \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    $([ "$USE_DEEPSPEED" = "1" ] && echo "--deepspeed zero2")

echo "Training finished. Output: $OUTPUT_DIR"
