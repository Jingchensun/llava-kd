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

# Extract and save result
accuracy_line=$(echo "$eval_output" | grep "Accuracy:" | tail -1)
f1_line=$(echo "$eval_output" | grep "F1 score:")
precision_line=$(echo "$eval_output" | grep "Precision:")
if [ -n "$accuracy_line" ]; then
    echo "POPE: $accuracy_line, $precision_line, $f1_line" >> eval/results/${MODEL_NAME}_eval.txt
fi
