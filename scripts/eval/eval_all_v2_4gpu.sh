#!/bin/bash
export LD_LIBRARY_PATH=/usr/local/lib/python3.12/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/extras/CUPTI/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/python3.12/site-packages/nvidia/nccl/lib

MODEL_PATH="/home/jsun/llava-kd/checkpoints/qwen25_distill_llava_factory/tiny-llava-Qwen2.5-0.5B-siglip-so400m-patch14-384-qwen2-0_5b_base-distill-pretrain/checkpoint-2000"
MODEL_NAME="Qwen25_15B_DFT_250918_v1120"
EVAL_DIR="./eval_dataset"

echo "=========================================="
echo "开始4GPU并行评估 ScienceQA"
echo "模型: $MODEL_PATH"
echo "=========================================="

# 将数据集分成4份，每张GPU处理1/4
CUDA_VISIBLE_DEVICES=0 bash scripts/eval/sqa_parallel.sh "$MODEL_PATH" "$MODEL_NAME" 4 0 &
CUDA_VISIBLE_DEVICES=1 bash scripts/eval/sqa_parallel.sh "$MODEL_PATH" "$MODEL_NAME" 4 1 &
CUDA_VISIBLE_DEVICES=2 bash scripts/eval/sqa_parallel.sh "$MODEL_PATH" "$MODEL_NAME" 4 2 &
CUDA_VISIBLE_DEVICES=3 bash scripts/eval/sqa_parallel.sh "$MODEL_PATH" "$MODEL_NAME" 4 3 &

# 等待所有GPU任务完成
wait

echo "=========================================="
echo "所有GPU任务完成，开始合并结果..."
echo "=========================================="

# 合并4个chunk的结果
cat $EVAL_DIR/sqa/answers/${MODEL_NAME}_chunk_0.jsonl \
    $EVAL_DIR/sqa/answers/${MODEL_NAME}_chunk_1.jsonl \
    $EVAL_DIR/sqa/answers/${MODEL_NAME}_chunk_2.jsonl \
    $EVAL_DIR/sqa/answers/${MODEL_NAME}_chunk_3.jsonl \
    > $EVAL_DIR/sqa/answers/${MODEL_NAME}.jsonl

echo "结果已合并到: $EVAL_DIR/sqa/answers/${MODEL_NAME}.jsonl"

# 运行评估脚本
echo "=========================================="
echo "开始计算准确率..."
echo "=========================================="

python3.12 llavakd/eval/eval_science_qa.py \
    --base-dir $EVAL_DIR/sqa \
    --result-file $EVAL_DIR/sqa/answers/$MODEL_NAME.jsonl \
    --output-file $EVAL_DIR/sqa/answers/"$MODEL_NAME"_output.jsonl \
    --output-result $EVAL_DIR/sqa/answers/"$MODEL_NAME"_result.json

echo "=========================================="
echo "评估完成！"
echo "结果文件: $EVAL_DIR/sqa/answers/${MODEL_NAME}_result.json"
echo "=========================================="
