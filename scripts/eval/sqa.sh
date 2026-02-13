#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

MODEL_PATH=$1
MODEL_NAME=$2
EVAL_DIR="./eval_dataset"

output_file=$EVAL_DIR/sqa/answers/$MODEL_NAME/merge.jsonl

# Check if answer file already exists
if [ -f "$output_file" ]; then
    echo "SQA answer file exists: $output_file"
    echo "Skipping answer generation, proceeding to evaluation..."
else
    echo "SQA answer file not found, generating answers..."
    
    for IDX in $(seq 0 $((CHUNKS-1))); do
        CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python3.12 -m llavakd.eval.model_vqa_science \
            --model-path $MODEL_PATH \
            --question-file $EVAL_DIR/sqa/llava_test_CQM-A.json \
            --image-folder $EVAL_DIR/sqa/images/test \
            --answers-file $EVAL_DIR/sqa/answers/$MODEL_NAME/${CHUNKS}_${IDX}.jsonl \
            --num-chunks $CHUNKS \
            --chunk-idx $IDX \
            --single-pred-prompt \
            --temperature 0 \
            --conv-mode phi &
    done

    wait

    # Clear out the output file if it exists.
    > "$output_file"

    # Loop through the indices and concatenate each file.
    for IDX in $(seq 0 $((CHUNKS-1))); do
        cat $EVAL_DIR/sqa/answers/$MODEL_NAME/${CHUNKS}_${IDX}.jsonl >> "$output_file"
    done
    
    echo "SQA answer generation completed"
fi

mkdir -p eval/results

eval_output=$(python3.12 llavakd/eval/eval_science_qa.py \
    --base-dir $EVAL_DIR/sqa \
    --result-file "$output_file" \
    --output-file $EVAL_DIR/sqa/answers/"$MODEL_NAME"_output.jsonl \
    --output-result $EVAL_DIR/sqa/answers/"$MODEL_NAME"_result.json 2>&1)

echo "$eval_output"

# Extract and save result
result_line=$(echo "$eval_output" | grep -E "Total:.*Accuracy:")
if [ -n "$result_line" ]; then
    echo "SQA: $result_line" >> eval/results/${MODEL_NAME}_eval.txt
fi

