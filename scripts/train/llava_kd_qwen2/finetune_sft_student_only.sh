#!/bin/bash
# ============================================================
# SFT训练脚本 - 仅训练Student模型
# - Vision Tower: 从 HuggingFace 加载 (frozen)
# - LLM: 从 HuggingFace 加载 (full tuning)
# - Connector: 从 pretrain checkpoint 加载 (full tuning)
# - 中间checkpoint: 保存 connector + llm 权重
# - 最终保存: 保存完整模型
# ============================================================

if [ $# -ne 10 ]; then
    echo "Usage: $0 <DATA_PATH> <IMAGE_PATH> <LLM_VERSION> <VT_VERSION> <VT_VERSION2> <CN_VERSION> <VERSION> <TRAIN_RECIPE> <MODEL_MAX_LENGTH> <PRETRAIN_CKPT_PATH>"
    exit 1
fi

# Assign the arguments to variables
DATA_PATH="$1"
IMAGE_PATH="$2"
LLM_VERSION="$3"
VT_VERSION="$4"
VT_VERSION2="$5"
CN_VERSION="$6"
VERSION="$7"
TRAIN_RECIPE="$8"
MODEL_MAX_LENGTH="$9"
PRETRAIN_CKPT_PATH="${10}"

VT_VARIANT="${VT_VERSION#*/}"
LLM_VARIANT="${LLM_VERSION#*/}"

# Change to project root directory
cd "$(dirname "$0")/../../.."

# Add project root to PYTHONPATH
export PYTHONPATH="${PWD}:${PYTHONPATH}"

# 创建输出目录并设置日志文件
OUTPUT_DIR="./checkpoints/${VERSION}"
mkdir -p "$OUTPUT_DIR"
LOG_FILE="${OUTPUT_DIR}/train_$(date +%Y%m%d_%H%M%S).log"

echo "=========================================="
echo "SFT Student Only Training"
echo "=========================================="
echo "Pretrain checkpoint: $PRETRAIN_CKPT_PATH"
echo "Output directory: $OUTPUT_DIR"
echo "Log file: $LOG_FILE"
echo "=========================================="

deepspeed --include localhost:0,1,2,3 --master_port $((29500 + RANDOM % 500)) llavakd/train/train_sft_student_only.py \
    --deepspeed scripts/zero2.json \
    --data_path  $DATA_PATH \
    --image_folder $IMAGE_PATH \
    --is_multimodal True \
    --conv_version qwen2_base \
    --model_name_or_path $LLM_VERSION \
    --vision_tower $VT_VERSION \
    --vision_tower2 "$VT_VERSION2" \
    --connector_type $CN_VERSION \
    --mm_vision_select_layer -2 \
    --image_aspect_ratio square \
    --attn_implementation flash_attention_2 \
    --bf16 True \
    --training_recipe $TRAIN_RECIPE \
    --tune_type_llm full \
    --tune_type_vision_tower frozen \
    --tune_vision_tower_from_layer 0 \
    --tune_type_connector full \
    --group_by_modality_length True \
    --pretrained_model_path $PRETRAIN_CKPT_PATH \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 1 \
    --max_steps 15 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 5 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length $MODEL_MAX_LENGTH \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --dataloader_pin_memory True \
    --lazy_preprocess True \
    --report_to tensorboard \
    --tokenizer_use_fast False \
    --run_name ${VERSION} \
    2>&1 | tee -a "$LOG_FILE"
