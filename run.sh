#!/bin/bash
set -e

TRAIN_FLAG=""
TEST_FLAG=""
GREEDY_FLAG="--greedy"
FORCE_RESPLIT_FLAG=""

FEW_SHOT_K=0
BATCH_SIZE=4
PROMPT_MODE="ctqrs"     # CTQRS 프롬프트
COT_FLAG="--no_cot"             # "--cot" 또는 "--no_cot" (비우면 기본 cot=True)
ADAPTER_MODE="base"         # "base" 또는 "qlora4" (비우면 전달 안 함=컨트롤러 기본 qlora4)
COT_STYLE=""
# COT_FILE=""

MODELS=("qwen")

# 추론 전용 → HF-only
export UNSLOTH_DISABLE_BACKEND_PATCHING=1

echo "▶ Running: prompt=${PROMPT_MODE}, CoT=${COT_FLAG:-default-ON}, k=${FEW_SHOT_K}, adapter=${ADAPTER_MODE:-<controller-default>}, batch=${BATCH_SIZE}"
python3 controller_agent.py \
  ${TRAIN_FLAG} \
  ${TEST_FLAG} \
  ${GREEDY_FLAG} \
  ${FORCE_RESPLIT_FLAG} \
  --few_shot_k "${FEW_SHOT_K}" \
  --batch_size "${BATCH_SIZE}" \
  --prompt "${PROMPT_MODE}" \
  ${COT_FLAG:+${COT_FLAG}} \
  ${ADAPTER_MODE:+--adapter "${ADAPTER_MODE}"} \
  ${COT_STYLE:+--cot_style "${COT_STYLE}"} \
  ${COT_FILE:+--cot_file "${COT_FILE}"} \
  --models "${MODELS[@]}"
