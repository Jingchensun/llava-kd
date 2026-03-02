# !/bin/bash
bash 3_train_qwen2_distill_after_sft.sh type2

# Change to project root directory
cd /home/jsun/llava-kd

MODEL_PATH="/home/jsun/llava-kd/checkpoints/qwen25-0_5b-distill-after-sft-type2"
MODEL_NAME="Qwen25_0.5B_Local-last-type2"
echo "=== Running SQA Evaluation ==="
CUDA_VISIBLE_DEVICES=0,1,2,3 bash /home/jsun/llava-kd/scripts/eval/sqa.sh "$MODEL_PATH" "$MODEL_NAME" || echo "SQA evaluation failed, continuing to next evaluation..."