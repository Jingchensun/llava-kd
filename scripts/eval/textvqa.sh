#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

MODEL_PATH=$1
MODEL_NAME=$2
EVAL_DIR="./eval_dataset"

output_file=$EVAL_DIR/textvqa/answers/$MODEL_NAME/merge.jsonl

# Check if answer file already exists
if [ -f "$output_file" ]; then
    echo "TextVQA answer file exists: $output_file"
    echo "Skipping answer generation, proceeding to evaluation..."
else
    echo "TextVQA answer file not found, generating answers..."
    
    for IDX in $(seq 0 $((CHUNKS-1))); do
        CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python3.12 -m llavakd.eval.model_vqa_loader \
            --model-path $MODEL_PATH \
            --question-file $EVAL_DIR/textvqa/llava_textvqa_val_v051_ocr.jsonl \
            --image-folder $EVAL_DIR/textvqa/train_images \
            --answers-file $EVAL_DIR/textvqa/answers/$MODEL_NAME/${CHUNKS}_${IDX}.jsonl \
            --num-chunks $CHUNKS \
            --chunk-idx $IDX \
            --temperature 0 \
            --conv-mode phi &
    done

    wait

    # Clear out the output file if it exists.
    > "$output_file"

    # Loop through the indices and concatenate each file.
    for IDX in $(seq 0 $((CHUNKS-1))); do
        cat $EVAL_DIR/textvqa/answers/$MODEL_NAME/${CHUNKS}_${IDX}.jsonl >> "$output_file"
    done
    
    echo "TextVQA answer generation completed"
fi

mkdir -p eval/results

eval_output=$(python3.12 -m llavakd.eval.eval_textvqa \
    --annotation-file $EVAL_DIR/textvqa/TextVQA_0.5.1_val.json \
    --result-file "$output_file" 2>&1)

echo "$eval_output"

# Extract and save result
samples_line=$(echo "$eval_output" | grep "Samples:")
accuracy_line=$(echo "$eval_output" | grep "Accuracy:")
if [ -n "$samples_line" ] && [ -n "$accuracy_line" ]; then
    echo "TextVQA: $samples_line, $accuracy_line" >> eval/results/${MODEL_NAME}_eval.txt
fi

