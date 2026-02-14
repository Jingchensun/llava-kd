#!/bin/bash

# SLURM job directives
#SBATCH --job-name=pretrain                     # 作业名称
#SBATCH --output=0_pretrain.out                   # 输出日志文件
#SBATCH --error=0_pretrain.err                    # 错误日志文件
#SBATCH --ntasks=1                             # 启动 1 个任务
#SBATCH --cpus-per-task=16                      # 每个任务 8 个 CPU 核心
#SBATCH --mem=128GB                            # 每个 CPU 核心分配 32GB 内存
#SBATCH --time=4-23:00:00                        # 运行时间（1小时）
#SBATCH --partition=prod_long                       # 分区设置（prod 分区）
#SBATCH --gres=shard:16 
#SBATCH --constraint=GPUMODEL_H200-SXM5


export WANDB_API_KEY="595cc8071abc681aa346ae6017f73fc16a9b2033"  # 替换为你的API Key
export WANDB_MODE=online  # 确保 wandb 处于在线模式

source /home/onsi/jsun/miniconda3/bin/activate llava-kd        # 激活 Conda 环境

# bash 1_train_qwen2_distill.sh
# bash 2_train_sft_student_only.sh
# bash 3_train_qwen2_distill_after_sft.sh
bash 3_train_qwen2_distill_after_sft.sh type3

sleep infinity