#!/bin/bash


MODEL_PATH=$1
MODEL_NAME=$2
EVAL_DIR="./eval_dataset"

python3.12 -m llavakd.eval.model_vqa_science \
    --model-path $MODEL_PATH \
    --question-file $EVAL_DIR/sqa/llava_test_CQM-A.json \
    --image-folder $EVAL_DIR/sqa/images/test \
    --answers-file $EVAL_DIR/sqa/answers/$MODEL_NAME.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode phi

python3.12 llavakd/eval/eval_science_qa.py \
    --base-dir $EVAL_DIR/sqa \
    --result-file $EVAL_DIR/sqa/answers/$MODEL_NAME.jsonl \
    --output-file $EVAL_DIR/sqa/answers/"$MODEL_NAME"_output.jsonl \
    --output-result $EVAL_DIR/sqa/answers/"$MODEL_NAME"_result.json

