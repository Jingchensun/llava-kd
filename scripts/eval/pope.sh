#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

MODEL_PATH=$1
MODEL_NAME=$2
EVAL_DIR="./eval_dataset"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python3.12 -m llavakd.eval.model_vqa_pope \
        --model-path $MODEL_PATH \
        --question-file $EVAL_DIR/pope/llava_pope_test.jsonl \
        --image-folder $EVAL_DIR/pope/val2014 \
        --answers-file $EVAL_DIR/pope/answers/$MODEL_NAME/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode phi &
done

wait

output_file=$EVAL_DIR/pope/answers/$MODEL_NAME/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat $EVAL_DIR/pope/answers/$MODEL_NAME/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

mkdir -p eval/results

eval_output=$(python3.12 llavakd/eval/eval_pope.py \
    --annotation-dir $EVAL_DIR/pope/coco \
    --question-file $EVAL_DIR/pope/llava_pope_test.jsonl \
    --result-file "$output_file" 2>&1)

echo "$eval_output"

# Extract all three accuracies and calculate average
accuracies=$(echo "$eval_output" | grep "^Accuracy:" | awk '{print $2}')
avg_accuracy=$(echo "$accuracies" | awk '{sum+=$1; count+=1} END {printf "%.4f", sum/count}')

# Extract F1 and Precision for summary (using last one as representative)
f1_line=$(echo "$eval_output" | grep "F1 score:" | tail -1)
precision_line=$(echo "$eval_output" | grep "Precision:" | tail -1)

echo ""
echo "========================================"
echo "POPE Average Accuracy: $avg_accuracy"
echo "========================================"

if [ -n "$avg_accuracy" ]; then
    echo "POPE: Accuracy: $avg_accuracy, $precision_line, $f1_line" >> eval/results/${MODEL_NAME}_eval.txt
fi
