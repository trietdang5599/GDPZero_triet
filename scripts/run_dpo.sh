#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TRANSFORMERS_NO_TORCHVISION=1

MASTER_PORT="${MASTER_PORT:-4040}"
NUM_GPUS="${NUM_GPUS:-2}"
GPU_IDS="${GPU_IDS:-0,1}"
ACCELERATE_CFG="${ACCELERATE_CFG:-${REPO_ROOT}/config/accelerate_config.yaml}"

MODEL_NAME="${MODEL_NAME:-meta-llama/Meta-Llama-3-8B-Instruct}"
SFT_MODEL_PATH="${SFT_MODEL_PATH:-outputs/${MODEL_NAME//\//_}-sft}"
PREF_PATH="${PREF_PATH:-preference_pairs.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/${MODEL_NAME//\//_}-dpo}"
SEED="${SEED:-42}"
USE_LORA="${USE_LORA:-1}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj}"

mkdir -p "$(dirname "${OUTPUT_DIR}")"

LORA_ARGS=()
if [[ "${USE_LORA}" != "0" ]]; then
  LORA_ARGS=(
    --use-lora
    --lora-r "${LORA_R}"
    --lora-alpha "${LORA_ALPHA}"
    --lora-dropout "${LORA_DROPOUT}"
    --lora-target-modules "${LORA_TARGET_MODULES}"
  )
fi

accelerate launch \
  --main_process_port "${MASTER_PORT}" \
  --gpu_ids "${GPU_IDS}" \
  --num_processes "${NUM_GPUS}" \
  --config_file "${ACCELERATE_CFG}" \
  --multi_gpu \
  "${REPO_ROOT}/train_llm.py" \
  --algorithm dpo \
  --dataset-path "${PREF_PATH}" \
  --model-name "${MODEL_NAME}" \
  --reference-model-name "${SFT_MODEL_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --batch-size "${BATCH_SIZE:-4}" \
  --gradient-accumulation "${GRAD_ACCUM:-16}" \
  --num-train-epochs "${NUM_EPOCHS:-3}" \
  --learning-rate "${LEARNING_RATE:-1e-5}" \
  --max-length "${MAX_LENGTH:-512}" \
  --dpo-beta "${DPO_BETA:-0.1}" \
  --gradient-checkpointing \
  --load-in-4bit \
  --bf16 \
  "${LORA_ARGS[@]}" \
  --seed "${SEED}" \
  "$@"
