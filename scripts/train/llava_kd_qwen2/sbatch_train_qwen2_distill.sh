#!/bin/bash

# SLURM job directives
#SBATCH --job-name=llava_qwen                     # 作业名称
#SBATCH --output=llava_qwen.out                   # 输出日志文件
#SBATCH --error=llava_qwen.err                    # 错误日志文件
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

export LD_LIBRARY_PATH=/usr/local/lib/python3.12/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/extras/CUPTI/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/python3.12/site-packages/nvidia/nccl/lib


DATA_PATH=/home/jsun/llava-kd/dataset/blip_laion_cc_sbu_558k.json #pretrain annotation file path
FINETUNE_DATA_PATH=/home/jsun/llava-kd/dataset/llava_v1_5_mix665k.json #finetune annotation file path
IMAGE_PATH=/home/jsun/llava-kd/dataset/llava/llava_pretrain/images #pretrain image dir
FINETUNE_IMAGE_PATH=/home/jsun/llava-kd/dataset #finetune image dir

LLM_VERSION=Qwen/Qwen2.5-0.5B # llm path in huggingface
VT_VERSION=google/siglip-so400m-patch14-384 #vision tower path in huggingface
VT_VERSION2="" #if you are not using mof vision tower, keep it empty
CN_VERSION=mlp2x_gelu #connector type, other options are: qformer, resampler, etc
CONV_VERSION=qwen2_base #chat template, other options are: phi, llama, gemmma, etc
VERSION=qwen2-0_5b_base #experiment name for recording different runnings
TRAIN_RECIPE=common #training recipes, other options are: lora, qlora
MODEL_MAX_LENGTH=2048 #max model length for llm

bash /home/jsun/llava-kd/scripts/train/llava_kd_qwen2/pretrain_qwen2_distill.sh "$DATA_PATH" "$IMAGE_PATH" "$LLM_VERSION" "$VT_VERSION" "$VT_VERSION2" "$CN_VERSION" "$VERSION" "$TRAIN_RECIPE" "$MODEL_MAX_LENGTH"

sleep infinity