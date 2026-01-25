#!/bin/bash


MODEL_PATH=$1
MODEL_NAME=$2
EVAL_DIR="./eval_dataset" # If the evaluation fails, try changing the path to an absolute path

python3.12 -m llavakd.eval.model_vqa_loader \
    --model-path $MODEL_PATH \
    --question-file $EVAL_DIR/mme/llava_mme.jsonl \
    --image-folder $EVAL_DIR/mme/images \
    --answers-file $EVAL_DIR/mme/answers/$MODEL_NAME.jsonl \
    --temperature 0 \
   --conv-mode phi

cd $EVAL_DIR/mme

python3.12 convert_answer_to_mme.py --experiment $MODEL_NAME

cd eval_tool

python3.12 calculation.py --results_dir answers/$MODEL_NAME

