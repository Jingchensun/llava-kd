#!/bin/bash
# ============================================================
# Distill After SFT 训练入口脚本
#
# 模型加载策略:
#   - Teacher: 从 HuggingFace 加载 (Zhang199/TinyLLaVA-Qwen2.5-3B-SigLIP)
#   - Student Vision Tower: 从 HuggingFace 加载 (google/siglip-so400m-patch14-384)
#   - Student LLM + Connector: 从 SFT checkpoint 加载
#
# 训练策略:
#   - Vision Tower: frozen (冻结)
#   - LLM: full tuning (全量微调)
#   - Connector: full tuning (全量微调)
#
# 保存策略:
#   - 中间 checkpoint: 保存 LLM + Connector 权重
#   - 最终保存: 保存完整模型 (VT + Connector + LLM)
# ============================================================

# ============== 数据路径配置 ==============
DATA_PATH=/home/jsun/llava-kd/dataset/llava_v1_5_mix665k.json     # finetune annotation file
IMAGE_PATH=/home/jsun/llava-kd/dataset/finetune_data              # finetune image dir

# ============== 模型配置 ==============
# 注意: Teacher 模型在 train_distill_after_qwen2_sft.py 中配置
# Teacher: Zhang199/TinyLLaVA-Qwen2.5-3B-SigLIP
# Student VT: google/siglip-so400m-patch14-384
# Student LLM: Qwen/Qwen2.5-0.5B

LLM_VERSION=Qwen/Qwen2.5-0.5B                    # student llm (用于配置)
VT_VERSION=google/siglip-so400m-patch14-384     # student vision tower (用于配置)
VT_VERSION2=""                                   # 第二个 vision tower (MoF用，留空)
CN_VERSION=mlp2x_gelu                           # connector type
VERSION=qwen25-0_5b-distill-after-sft           # 实验名称
TRAIN_RECIPE=common                             # training recipe
MODEL_MAX_LENGTH=2048                           # max sequence length

# ============== SFT Checkpoint 路径 ==============
# 指定 SFT 阶段保存的 checkpoint 路径，用于加载 LLM + Connector 权重
SFT_CKPT_PATH=./checkpoints/qwen25-0_5b-sft

# ============== 开始训练 ==============
echo "=========================================="
echo "Starting Distill After SFT Training"
echo "=========================================="
echo "Data path: $DATA_PATH"
echo "Image path: $IMAGE_PATH"
echo "SFT checkpoint: $SFT_CKPT_PATH"
echo "Output: ./checkpoints/${VERSION}"
echo "=========================================="

bash finetune_distill_after_qwen2_sft.sh \
    "$DATA_PATH" "$IMAGE_PATH" "$LLM_VERSION" "$VT_VERSION" "$VT_VERSION2" \
    "$CN_VERSION" "$VERSION" "$TRAIN_RECIPE" "$MODEL_MAX_LENGTH" "$SFT_CKPT_PATH"
