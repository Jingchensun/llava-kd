#!/bin/bash

# SLURM job directives
#SBATCH --job-name=pretrain                     # 作业名称
#SBATCH --output=0_pretrain.out                   # 输出日志文件
#SBATCH --error=0_pretrain.err                    # 错误日志文件
#SBATCH --ntasks=1                             # 启动 1 个任务
#SBATCH --cpus-per-task=16                      # 每个任务 8 个 CPU 核心
#SBATCH --mem=64GB                            # 每个 CPU 核心分配 32GB 内存
#SBATCH --time=4-23:00:00                        # 运行时间（1小时）
#SBATCH --partition=prod_long                       # 分区设置（prod 分区）
#SBATCH --gres=shard:16 
#SBATCH --constraint=GPUMODEL_A100-SXM4|GPUMODEL_A100-PCIE|GPUMODEL_H100-SXM5|GPUMODEL_H200-SXM5


export WANDB_API_KEY="595cc8071abc681aa346ae6017f73fc16a9b2033"  # 替换为你的API Key
export WANDB_MODE=online  # 确保 wandb 处于在线模式

source /home/onsi/jsun/miniconda3/bin/activate llava-kd        # 激活 Conda 环境


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


sleep infinity