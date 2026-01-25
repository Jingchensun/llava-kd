#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

MODEL_PATH=$1
MODEL_NAME=$2
EVAL_DIR="./eval_dataset" # If the evaluation fails, try changing the path to an absolute path

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python3.12 -m llavakd.eval.model_vqa_loader \
        --model-path $MODEL_PATH \
        --question-file $EVAL_DIR/mme/llava_mme.jsonl \
        --image-folder $EVAL_DIR/mme/images \
        --answers-file $EVAL_DIR/mme/answers/$MODEL_NAME/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode phi &
done

wait

output_file=$EVAL_DIR/mme/answers/$MODEL_NAME/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat $EVAL_DIR/mme/answers/$MODEL_NAME/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

cd $EVAL_DIR/mme

python3.12 convert_answer_to_mme.py --experiment $MODEL_NAME

cd eval_tool

mkdir -p $OLDPWD/eval/results

eval_output=$(python3.12 calculation.py --results_dir answers/$MODEL_NAME 2>&1)
echo "$eval_output"

cd -
# Extract and save result
score_line=$(echo "$eval_output" | grep -E "MME metric:" | tail -1)
if [ -n "$score_line" ]; then
    echo "MME: $score_line" >> eval/results/${MODEL_NAME}_eval.txt
fi

