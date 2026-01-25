#!/usr/bin/env bash

WORK_DIR=$(cd "$(dirname "$0")/.."; pwd)
export PYTHONPATH=${WORK_DIR}
CONV_MODE=v1
cd ${WORK_DIR}

# ✅ 第一个参数为 OUTPUT_DIR（必填），第二个参数为 TARGET_STEP（可选）
OUTPUT_DIR=$1
TARGET_STEP=$2

if [ -z "$OUTPUT_DIR" ]; then
  echo "❌ 必须传入 OUTPUT_DIR，例如：bash scripts/benchmark-new.sh 469k-alignKD [12000]"
  exit 1
fi

CHECKPOINT_ROOT=outputs6/${OUTPUT_DIR}
OUTPUT_ROOT=outputs6/eval-${OUTPUT_DIR}

echo "✅ CHECKPOINT_ROOT: ${CHECKPOINT_ROOT}"
echo "✅ OUTPUT_ROOT: ${OUTPUT_ROOT}"

if [ -n "$TARGET_STEP" ]; then
  echo "✅ 只评估指定的 checkpoint-${TARGET_STEP}"
  CHECKPOINT_LIST=("${CHECKPOINT_ROOT}/checkpoint-${TARGET_STEP}")
else
  echo "✅ 未指定 step，默认评估所有 checkpoint-*"
  CHECKPOINT_LIST=(${CHECKPOINT_ROOT}/)
fi

for CHECKPOINT_PATH in "${CHECKPOINT_LIST[@]}"; do
  [ -d "$CHECKPOINT_PATH" ] || continue

  step=$(basename ${CHECKPOINT_PATH} | sed 's/checkpoint-//')
  # OUTPUT_DIR_EVAL=${OUTPUT_ROOT}/distill-checkpoint-${step}
  OUTPUT_DIR_EVAL=${OUTPUT_ROOT}/results
  mkdir -p ${OUTPUT_DIR_EVAL}

  echo "🧪 Evaluating checkpoint-${step}..."

  # for dataset in mme gqa textvqa pope mmbench sqa; do
  for dataset in sqa ; do
    case $dataset in
      # mme)
      #   MODEL_GENERATOR=mobilevlm.eval.model_vqa_loader
      #   DATA_ROOT=${WORK_DIR}/dataset/benchmark_data/mme
      #   SPLIT_NAME=llava_mme
      #   ;;
      # gqa)
      #   MODEL_GENERATOR=mobilevlm.eval.model_vqa_loader
      #   DATA_ROOT=${WORK_DIR}/dataset/benchmark_data/gqa
      #   SPLIT_NAME=llava_gqa_testdev_balanced
      #   ;;
      # textvqa)
      #   MODEL_GENERATOR=mobilevlm.eval.model_vqa_loader
      #   DATA_ROOT=${WORK_DIR}/dataset/benchmark_data/textvqa
      #   SPLIT_NAME=llava_textvqa_val_v051_ocr
      #   ;;
      # pope)
      #   MODEL_GENERATOR=mobilevlm.eval.model_vqa_loader
      #   DATA_ROOT=${WORK_DIR}/dataset/benchmark_data/pope
      #   SPLIT_NAME=llava_pope_test
      #   ;;
      # mmbench)
      #   MODEL_GENERATOR=mobilevlm.eval.model_vqa_mmbench
      #   DATA_ROOT=${WORK_DIR}/dataset/benchmark_data/mmbench
      #   SPLIT_NAME=mmbench_dev_en_20231003
      #   ;;
      sqa)
        MODEL_GENERATOR=mobilevlm.eval.model_vqa_science
        DATA_ROOT=${WORK_DIR}/dataset/benchmark_data/sqa
        SPLIT_NAME=llava_test_CQM-A
        ;;
    esac

    echo "➡️  Running ${dataset}..."
    CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/benchmark/${dataset}.sh \
      ${MODEL_GENERATOR} ${CHECKPOINT_PATH} ${CONV_MODE} ${SPLIT_NAME} ${DATA_ROOT} ${OUTPUT_DIR_EVAL}/${dataset}
  done
done
