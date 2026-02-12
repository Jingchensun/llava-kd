#!/bin/bash
# 使用新的加载方式评估 GQA 数据集
# 支持: HuggingFace 模型 和 本地完整 checkpoint

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

SPLIT="llava_gqa_testdev_balanced"
GQADIR="./eval_dataset/gqa"

MODEL_PATH=$1
MODEL_NAME=$2
LOAD_SOURCE=${3:-"auto"}  # 第三个参数：加载来源 (auto/huggingface/local)
EVAL_DIR="./eval_dataset"

echo "=============================================="
echo "GQA 评估配置"
echo "  模型路径: $MODEL_PATH"
echo "  模型名称: $MODEL_NAME"
echo "  加载来源: $LOAD_SOURCE"
echo "  GPU 数量: $CHUNKS"
echo "=============================================="

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llavakd.eval.model_vqa_loader_v2 \
        --model-path $MODEL_PATH \
        --load-source $LOAD_SOURCE \
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

python scripts/convert_gqa_for_eval.py --src $output_file --dst $GQADIR/testdev_balanced_predictions.json

mkdir -p eval/results

cd $GQADIR
eval_output=$(python eval/eval.py --tier testdev_balanced 2>&1)
echo "$eval_output"

cd -
# Extract and save result
accuracy_line=$(echo "$eval_output" | grep -i "accuracy\|score" | head -1)
if [ -n "$accuracy_line" ]; then
    echo "GQA: $accuracy_line" >> eval/results/${MODEL_NAME}_eval.txt
fi

echo "=============================================="
echo "GQA 评估完成！"
echo "=============================================="
