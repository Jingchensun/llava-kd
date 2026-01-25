#!/bin/bash

# SLURM job directives
#SBATCH --job-name=l_eval                     # 作业名称
#SBATCH --output=l_eval.out                   # 输出日志文件
#SBATCH --error=l_eval.err                    # 错误日志文件
#SBATCH --ntasks=1                             # 启动 1 个任务
#SBATCH --cpus-per-task=16                      # 每个任务 8 个 CPU 核心
#SBATCH --mem=32GB                            # 每个 CPU 核心分配 32GB 内存
#SBATCH --time=4-23:00:00                        # 运行时间（1小时）
#SBATCH --partition=prod_long                       # 分区设置（prod 分区）
#SBATCH --gres=shard:16 
#SBATCH --constraint=GPUMODEL_A100-SXM4|GPUMODEL_A100-PCIE|GPUMODEL_H100-SXM5|GPUMODEL_H200-SXM5

# 切换到项目根目录
cd /home/jsun/llava-kd
source /home/onsi/jsun/miniconda3/bin/activate llava-kd        # 激活 Conda 环境

MODEL_PATH="/home/jsun/llava-kd/pretrained_checkpoints/LLaVA_KD_ckpts/tiny-llava-Qwen2.5-3B-siglip-so400m-patch14-384-qwen2-0_5b_base-finetune"
MODEL_NAME="LLaVA_KD_Qwen25_3B"

echo "开始串行评估 - $(date)"

# 串行执行每个评估任务，避免OOM
echo "=== 运行 GQA 评估 ==="
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/eval/gqa.sh "$MODEL_PATH" "$MODEL_NAME"

echo "=== 运行 SQA 评估 ==="
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/eval/sqa.sh "$MODEL_PATH" "$MODEL_NAME"

echo "=== 运行 TextVQA 评估 ==="
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/eval/textvqa.sh "$MODEL_PATH" "$MODEL_NAME"

echo "=== 运行 POPE 评估 ==="
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/eval/pope.sh "$MODEL_PATH" "$MODEL_NAME"

echo "=== 运行 MME 评估 ==="
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/eval/mme.sh "$MODEL_PATH" "$MODEL_NAME"

echo "=== 运行 MMBench 评估 ==="
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/eval/mmbench.sh "$MODEL_PATH" "$MODEL_NAME"

echo "所有评估完成 - $(date)"


# CUDA_VISIBLE_DEVICES=4 bash scripts/eval/mmmu.sh "$MODEL_PATH" "$MODEL_NAME" &
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/eval/vqav2.sh "$MODEL_PATH" "$MODEL_NAME" &
# CUDA_VISIBLE_DEVICES=6 bash scripts/eval/vizwiz.sh "$MODEL_PATH" "$MODEL_NAME" &
# CUDA_VISIBLE_DEVICES=0 bash scripts/eval/mmbench_cn.sh "$MODEL_PATH" "$MODEL_NAME" &

sleep infinity     