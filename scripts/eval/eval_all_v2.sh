# #!/bin/bash
# export LD_LIBRARY_PATH=/usr/local/lib/python3.12/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
# export LD_LIBRARY_PATH=/usr/local/cuda-12.6/extras/CUPTI/lib64:$LD_LIBRARY_PATH
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/python3.12/site-packages/nvidia/nccl/lib

# MODEL_PATH="/home/jsun/llava-kd/checkpoints/qwen25-0_5b-distill-after-sft"
# MODEL_NAME="Qwen25_0.5B_Local"

MODEL_PATH="Zhang199/TinyLLaVA-Qwen2.5-3B-SigLIP"
MODEL_NAME="HF_TinyLLaVA_Qwen2_0.5B"


cd /home/jsun/llava-kd

echo "开始串行评估 - $(date)"

# 串行执行每个评估任务，避免OOM
# echo "=== 运行 GQA 评估 ==="
# CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/eval/gqa.sh "$MODEL_PATH" "$MODEL_NAME"

# echo "=== 运行 SQA 评估 ==="
# CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/eval/sqa.sh "$MODEL_PATH" "$MODEL_NAME"

# echo "=== 运行 TextVQA 评估 ==="
# CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/eval/textvqa.sh "$MODEL_PATH" "$MODEL_NAME"

# echo "=== 运行 POPE 评估 ==="
# CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/eval/pope.sh "$MODEL_PATH" "$MODEL_NAME"

# echo "=== 运行 MME 评估 ==="
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/eval/mme.sh "$MODEL_PATH" "$MODEL_NAME"

# echo "=== 运行 MMBench 评估 ==="
# CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/eval/mmbench.sh "$MODEL_PATH" "$MODEL_NAME"

echo "All evaluations completed - $(date)"

# CUDA_VISIBLE_DEVICES=4 bash scripts/eval/mmmu.sh "$MODEL_PATH" "$MODEL_NAME" &
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/eval/vqav2.sh "$MODEL_PATH" "$MODEL_NAME" &
# CUDA_VISIBLE_DEVICES=6 bash scripts/eval/vizwiz.sh "$MODEL_PATH" "$MODEL_NAME" &
# CUDA_VISIBLE_DEVICES=0 bash scripts/eval/mmbench_cn.sh "$MODEL_PATH" "$MODEL_NAME" &

wait        