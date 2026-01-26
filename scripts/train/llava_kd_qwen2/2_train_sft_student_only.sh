#!/bin/bash
# ============================================================
# SFT训练入口脚本 - 仅训练Student模型
# 
# 模型加载策略:
#   - Vision Tower: 从 HuggingFace 加载 (google/siglip-so400m-patch14-384)
#   - LLM: 从 HuggingFace 加载 (Qwen/Qwen2.5-0.5B)
#   - Connector: 从 pretrain 阶段的 checkpoint 加载
#
# 训练策略:
#   - Vision Tower: frozen (冻结)
#   - LLM: full tuning (全量微调)
#   - Connector: full tuning (全量微调)
#
# 保存策略:
#   - 中间checkpoint: 保存 connector + llm 权重
#   - 最终保存: 保存完整模型 (VT + Connector + LLM)
# ============================================================

# ============== 数据路径配置 ==============
DATA_PATH=/home/jsun/llava-kd/dataset/llava_v1_5_mix665k.json     # SFT annotation file
IMAGE_PATH=/home/jsun/llava-kd/dataset/finetune_data                            # SFT image dir

# ============== 模型配置 ==============
LLM_VERSION=Qwen/Qwen2.5-0.5B                    # student llm
VT_VERSION=google/siglip-so400m-patch14-384     # student vision tower
VT_VERSION2=""                                   # 第二个 vision tower (MoF用，留空)
CN_VERSION=mlp2x_gelu                           # connector type
VERSION=qwen25-0_5b-sft                         # 实验名称
TRAIN_RECIPE=common                             # training recipe
MODEL_MAX_LENGTH=2048                           # max sequence length

# ============== Pretrain Checkpoint 路径 ==============
# 指定 pretrain 阶段保存的 checkpoint 路径，用于加载 connector 权重
# 注意: 需要指向具体的 checkpoint 目录 (如 checkpoint-50)，不是父目录
PRETRAIN_CKPT_PATH=./checkpoints/qwen25-0_5b-pretrain/checkpoint-50

# ============== 开始训练 ==============
echo "=========================================="
echo "Starting SFT Student Only Training"
echo "=========================================="
echo "Data path: $DATA_PATH"
echo "Image path: $IMAGE_PATH"
echo "Pretrain checkpoint: $PRETRAIN_CKPT_PATH"
echo "Output: ./checkpoints/${VERSION}-sft"
echo "=========================================="

bash finetune_sft_student_only.sh \
    "$DATA_PATH" "$IMAGE_PATH" "$LLM_VERSION" "$VT_VERSION" "$VT_VERSION2" \
    "$CN_VERSION" "$VERSION" "$TRAIN_RECIPE" "$MODEL_MAX_LENGTH" "$PRETRAIN_CKPT_PATH"
