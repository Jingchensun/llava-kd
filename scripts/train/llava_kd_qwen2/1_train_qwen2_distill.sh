#!/bin/bash
# ============== 环境配置 ==============
# export LD_LIBRARY_PATH=/usr/local/lib/python3.12/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
# export LD_LIBRARY_PATH=/usr/local/cuda-12.6/extras/CUPTI/lib64:$LD_LIBRARY_PATH
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/python3.12/site-packages/nvidia/nccl/lib

# ============== 数据路径配置 ==============
DATA_PATH=/home/jsun/llava-kd/dataset/blip_laion_cc_sbu_558k.json     # pretrain annotation file
IMAGE_PATH=/home/jsun/llava-kd/dataset/llava/llava_pretrain/images   # pretrain image dir

# ============== 模型配置 ==============
# 注意: Teacher 和 Student 模型的 HuggingFace ID 在 train_distill_qwen2.py 中配置
# Teacher: Zhang199/TinyLLaVA-Qwen2.5-3B-SigLIP
# Student LLM: Qwen/Qwen2.5-0.5B
# Student Vision Tower: google/siglip-so400m-patch14-384

LLM_VERSION=Qwen/Qwen2.5-0.5B                    # student llm (用于配置和命名)
VT_VERSION=google/siglip-so400m-patch14-384     # student vision tower (用于配置和命名)
VT_VERSION2=""                                   # 第二个 vision tower (MoF用，留空)
CN_VERSION=mlp2x_gelu                           # connector type
VERSION=qwen25-0_5b-pretrain                      # 实验名称
TRAIN_RECIPE=common                             # training recipe
MODEL_MAX_LENGTH=2048                           # max sequence length

# ============== 开始训练 ==============
bash pretrain_qwen2_distill.sh \
    "$DATA_PATH" "$IMAGE_PATH" "$LLM_VERSION" "$VT_VERSION" "$VT_VERSION2" \
    "$CN_VERSION" "$VERSION" "$TRAIN_RECIPE" "$MODEL_MAX_LENGTH"
