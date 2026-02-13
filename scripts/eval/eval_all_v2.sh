# #!/bin/bash
# export LD_LIBRARY_PATH=/usr/local/lib/python3.12/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
# export LD_LIBRARY_PATH=/usr/local/cuda-12.6/extras/CUPTI/lib64:$LD_LIBRARY_PATH
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/python3.12/site-packages/nvidia/nccl/lib

# MODEL_PATH="/home/jsun/llava-kd/checkpoints/qwen25-0_5b-distill-after-sft"
# MODEL_NAME="Qwen25_0.5B_Local"

MODEL_PATH="Zhang199/TinyLLaVA-Qwen2.5-3B-SigLIP"
MODEL_NAME="HF_TinyLLaVA_Qwen25_3B"


cd /home/jsun/llava-kd

# Initialize result file
RESULT_FILE="eval/results/${MODEL_NAME}_eval.txt"
mkdir -p eval/results
> "$RESULT_FILE"

echo "Starting serial evaluation - $(date)"
echo "Results will be saved to: $RESULT_FILE"

# Execute evaluation tasks serially to avoid OOM
echo "=== Running GQA Evaluation ==="
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/eval/gqa.sh "$MODEL_PATH" "$MODEL_NAME" || echo "GQA evaluation failed, continuing to next evaluation..."

echo "=== Running SQA Evaluation ==="
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/eval/sqa.sh "$MODEL_PATH" "$MODEL_NAME" || echo "SQA evaluation failed, continuing to next evaluation..."

echo "=== Running TextVQA Evaluation ==="
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/eval/textvqa.sh "$MODEL_PATH" "$MODEL_NAME" || echo "TextVQA evaluation failed, continuing to next evaluation..."

echo "=== Running POPE Evaluation ==="
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/eval/pope.sh "$MODEL_PATH" "$MODEL_NAME" || echo "POPE evaluation failed, continuing to next evaluation..."

echo "=== Running MME Evaluation ==="
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/eval/mme.sh "$MODEL_PATH" "$MODEL_NAME" || echo "MME evaluation failed, continuing to next evaluation..."

echo "=== Running MMBench Evaluation ==="
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/eval/mmbench.sh "$MODEL_PATH" "$MODEL_NAME" || echo "MMBench evaluation failed, continuing to next evaluation..."

echo "All evaluations completed - $(date)"
echo ""
echo "========== Evaluation Results Summary =========="
cat "$RESULT_FILE"
echo "================================================="
echo "Results saved to: $RESULT_FILE"
     