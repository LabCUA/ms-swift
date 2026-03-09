#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# UI-TARS 1.5 7B 只跑评估脚本（ms-swift SFT + CUA metric）
# 使用 conda 环境 sw，对 traj_v2 的 val.jsonl 跑一次 eval（不做训练）。
#
# 用法示例：
#   # 评估基础模型（或你指定的模型）
#   ./run_eval.sh
#   # 评估某个 checkpoint
#   MODEL_ID=/path/to/checkpoint ./run_eval.sh
# -----------------------------------------------------------------------------

# 环境配置（与 run_sft.sh 对齐）
export HF_ENDPOINT='https://hf-mirror.com'
export HF_HOME='/data/250010149/hf_cache'
unset XDG_CACHE_HOME
export NVIDIA_DRIVER_CAPABILITIES="${NVIDIA_DRIVER_CAPABILITIES:-compute,utility}"
if [ -d /lib/x86_64-linux-gnu ]; then
    export LD_LIBRARY_PATH="/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

# 激活 conda 环境 sw（已在其中 pip install -e . 安装 ms-swift）
# source "$(conda info --base)/etc/profile.d/conda.sh"
# conda activate sw

# 模型与数据配置
# 默认评估 ByteDance-Seed/UI-TARS-1.5-7B，可通过 MODEL_ID 覆盖为任意 checkpoint 路径
MODEL_ID="${MODEL_ID:-ByteDance-Seed/UI-TARS-1.5-7B}"
TRAJ_V2_DIR="${TRAJ_V2_DIR:-/data/250010149/hf_cache/traj_v2}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/250010149/ms-swift/labcua/output/uitars15_7b_eval}"

# 单卡 eval 即可；如需多卡 + deepspeed，可仿照 run_sft.sh 自行扩展
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MAX_PIXELS="${MAX_PIXELS:-1003520}"

cd /data/250010149/ms-swift

swift sft \
    --model "$MODEL_ID" \
    --model_type qwen2_5_vl \
    --template qwen2_5_vl \
    --use_hf true \
    --dataset "${TRAJ_V2_DIR}/train_v1.jsonl" \
    --val_dataset "${TRAJ_V2_DIR}/val_v1.jsonl" \
    --tuner_type lora \
    --torch_dtype bfloat16 \
    --num_train_epochs 0 \
    --max_steps 0 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --freeze_vit true \
    --freeze_aligner true \
    --gradient_accumulation_steps 16 \
    --eval_steps 100 \
    --save_steps 1000000 \
    --save_total_limit 1 \
    --logging_steps 10 \
    --max_length 32768 \
    --eval_on_start true \
    --eval_metric cua \
    --predict_with_generate true \
    --max_new_tokens 1024 \
    --output_dir "$OUTPUT_DIR" \
    --warmup_ratio 0.0 \
    --dataloader_num_workers 4

echo "Eval finished. Logs & metrics under: $OUTPUT_DIR"