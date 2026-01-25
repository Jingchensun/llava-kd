#!/bin/bash

MODEL_PATH=$1
MODEL_NAME=$2
NUM_CHUNKS=$3
CHUNK_IDX=$4
EVAL_DIR="./eval_dataset"

echo "Running ScienceQA evaluation on GPU $CUDA_VISIBLE_DEVICES - Chunk $CHUNK_IDX/$NUM_CHUNKS"

python3.12 -m llavakd.eval.model_vqa_science \
    --model-path $MODEL_PATH \
    --question-file $EVAL_DIR/sqa/llava_test_CQM-A.json \
    --image-folder $EVAL_DIR/sqa/images/test \
    --answers-file $EVAL_DIR/sqa/answers/${MODEL_NAME}_chunk_${CHUNK_IDX}.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode phi \
    --num-chunks $NUM_CHUNKS \
    --chunk-idx $CHUNK_IDX

echo "Finished chunk $CHUNK_IDX on GPU $CUDA_VISIBLE_DEVICES"
