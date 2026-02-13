#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

SPLIT="llava_gqa_testdev_balanced"
GQADIR="./eval_dataset/gqa"

MODEL_PATH=$1
MODEL_NAME=$2
EVAL_DIR="./eval_dataset"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python3.12 -m llavakd.eval.model_vqa_loader \
        --model-path $MODEL_PATH \
        --question-file $EVAL_DIR/gqa/$SPLIT.jsonl \
        --image-folder $EVAL_DIR/gqa/images \
        --answers-file $EVAL_DIR/gqa/answers/$SPLIT/$MODEL_NAME/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode phi &
done

wait

output_file=$EVAL_DIR/gqa/answers/$SPLIT/$MODEL_NAME/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat $EVAL_DIR/gqa/answers/$SPLIT/$MODEL_NAME/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python3.12 scripts/convert_gqa_for_eval.py --src $output_file --dst $GQADIR/testdev_balanced_predictions.json

mkdir -p eval/results

cd $GQADIR
eval_output=$(python3.12 eval.py --tier testdev_balanced --questions testdev_balanced_questions.json --predictions testdev_balanced_predictions.json 2>&1)

cd -
# Extract and save result
accuracy_line=$(echo "$eval_output" | grep "^Accuracy:" | head -1)
if [ -n "$accuracy_line" ]; then
    echo "GQA: $accuracy_line"
    echo "GQA: $accuracy_line" >> eval/results/${MODEL_NAME}_eval.txt
fi
