#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

SPLIT="mmbench_dev_en_20231003"

MODEL_PATH=$1
MODEL_NAME=$2
EVAL_DIR="./eval_dataset"

output_file=$EVAL_DIR/mmbench/answers/$SPLIT/$MODEL_NAME/merge.jsonl

# Check if answer file already exists
if [ -f "$output_file" ]; then
    echo "MMBench answer file exists: $output_file"
    echo "Skipping answer generation, proceeding to evaluation..."
else
    echo "MMBench answer file not found, generating answers..."
    
    for IDX in $(seq 0 $((CHUNKS-1))); do
        CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python3.12 -m llavakd.eval.model_vqa_mmbench \
            --model-path $MODEL_PATH \
            --question-file $EVAL_DIR/mmbench/$SPLIT.tsv \
            --answers-file $EVAL_DIR/mmbench/answers/$SPLIT/$MODEL_NAME/${CHUNKS}_${IDX}.jsonl \
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
        cat $EVAL_DIR/mmbench/answers/$SPLIT/$MODEL_NAME/${CHUNKS}_${IDX}.jsonl >> "$output_file"
    done
    
    echo "MMBench answer generation completed"
fi

mkdir -p  $EVAL_DIR/mmbench/answers_upload/$SPLIT
mkdir -p eval/results

# Create a symbolic link or copy the merge.jsonl to expected location
cp "$output_file" "$EVAL_DIR/mmbench/answers/$SPLIT/${MODEL_NAME}.jsonl"

eval_output=$(python3.12 scripts/convert_mmbench_for_submission.py \
    --annotation-file $EVAL_DIR/mmbench/$SPLIT.tsv \
    --result-dir $EVAL_DIR/mmbench/answers/$SPLIT \
    --upload-dir $EVAL_DIR/mmbench/answers_upload/$SPLIT \
    --experiment $MODEL_NAME 2>&1)

echo "$eval_output"

# Count results and save
result_file="$EVAL_DIR/mmbench/answers_upload/$SPLIT/${MODEL_NAME}.xlsx"
if [ -f "$result_file" ]; then
    num_answers=$(python3.12 -c "import pandas as pd; df=pd.read_excel('$result_file'); print(len(df))")
    echo ""
    echo "=========================================="
    echo "MMBench Answer Generation Complete"
    echo "Total answers: $num_answers"
    echo "Results saved to: ${result_file}"
    echo "=========================================="
    
    # Evaluate accuracy
    echo ""
    echo "Starting accuracy evaluation..."
    python3.12 $EVAL_DIR/mmbench/eval.py \
        --result "$result_file" \
        --meta $EVAL_DIR/mmbench/$SPLIT.tsv \
        --model $MODEL_NAME
    
    # Extract overall accuracy from the result
    overall_csv="$EVAL_DIR/mmbench/answers_upload/$SPLIT/${MODEL_NAME}_overall.csv"
    if [ -f "$overall_csv" ]; then
        accuracy=$(python3.12 -c "import pandas as pd; df=pd.read_csv('$overall_csv'); print(f'{df.iloc[0][\"overall\"]*100:.2f}')")
        echo ""
        echo "=========================================="
        echo "MMBench Evaluation Results"
        echo "Overall Accuracy: ${accuracy}%"
        echo "Detailed results saved to: $EVAL_DIR/mmbench/answers_upload/$SPLIT/${MODEL_NAME}_*"
        echo "=========================================="
        echo "MMBench: Accuracy: ${accuracy}% | Total answers: $num_answers | Results: ${result_file}" >> eval/results/${MODEL_NAME}_eval.txt
    else
        echo "Warning: Evaluation results not found. Check $EVAL_DIR/mmbench/answers_upload/$SPLIT/ for details."
        echo "MMBench: Total answers: $num_answers | Results: ${result_file}" >> eval/results/${MODEL_NAME}_eval.txt
    fi
fi