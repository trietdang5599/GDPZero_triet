#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TRANSFORMERS_NO_TORCHVISION=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN

PYTHON_BIN="${PYTHON:-python}"
ACCELERATE_BIN="${ACCELERATE_BIN:-accelerate}"

MASTER_PORT="${MASTER_PORT:-4040}"
NUM_GPUS="${NUM_GPUS:-2}"
GPU_IDS="${GPU_IDS:-0,1}"
ACCELERATE_CFG="${ACCELERATE_CFG:-${REPO_ROOT}/config/accelerate_config.yaml}"

MODEL_NAME="${MODEL_NAME:-meta-llama/Meta-Llama-3-8B-Instruct}"
DATASET_TRAIN_PATH="${DATASET_TRAIN_PATH:-${REPO_ROOT}/data/p4g/300_dialog_turn_based-train.jsonl}"
DATASET_VAL_PATH="${DATASET_VAL_PATH:-${REPO_ROOT}/data/p4g/300_dialog_turn_based-val.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/${MODEL_NAME//\//_}-sft}"
SEED="${SEED:-42}"

USE_LORA="${USE_LORA:-1}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj}"

GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-1}"
CHECKPOINT_REENTRANT="${CHECKPOINT_REENTRANT:-1}"
FP16="${FP16:-0}"
BF16="${BF16:-1}"
LOAD_IN_4BIT="${LOAD_IN_4BIT:-1}"
LOAD_IN_8BIT="${LOAD_IN_8BIT:-0}"
DEVICE_MAP="${DEVICE_MAP:-}"

BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-32}"
NUM_EPOCHS="${NUM_EPOCHS:-5}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
MAX_LENGTH="${MAX_LENGTH:-2048}"

# role + field map 
SYSTEM_FIELD="${SYSTEM_FIELD:-er}"      # key chứa lời của Persuader trong data
USER_FIELD="${USER_FIELD:-ee}"            # key chứa lời của Persuadee trong data
SYSTEM_ROLE="${SYSTEM_ROLE:-Persuader}"
USER_ROLE="${USER_ROLE:-Persuadee}"

# Optional dataset presets
USE_CB_DATASET="${USE_CB_DATASET:-0}"
USE_P4G_DATASET="${USE_P4G_DATASET:-0}"
if [[ "${USE_CB_DATASET}" != "0" && "${USE_P4G_DATASET}" != "0" ]]; then
  echo "Only one of USE_CB_DATASET or USE_P4G_DATASET can be enabled." >&2
  exit 1
fi
if [[ "${USE_CB_DATASET}" != "0" ]]; then
  CB_DATA_DIR="${REPO_ROOT}/data/CraigslistBargains"
  DATASET_TRAIN_PATH="${CB_DATA_DIR}/test.json"
  DATASET_VAL_PATH="${CB_DATA_DIR}/val.json"
  SYSTEM_FIELD="seller"
  USER_FIELD="buyer"
  SYSTEM_ROLE="Seller"
  USER_ROLE="Buyer"
elif [[ "${USE_P4G_DATASET}" != "0" ]]; then
  P4G_DATA_DIR="${REPO_ROOT}/data/p4g"
  DATASET_TRAIN_PATH="${P4G_DATA_DIR}/300_dialog_turn_based-train.jsonl"
  DATASET_VAL_PATH="${P4G_DATA_DIR}/300_dialog_turn_based-val.jsonl"
  SYSTEM_FIELD="er"
  USER_FIELD="ee"
  SYSTEM_ROLE="Persuader"
  USER_ROLE="Persuadee"
fi


mkdir -p "${OUTPUT_DIR}"

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

PRECISION_ARGS=()
if [[ "${FP16}" != "0" ]]; then
  PRECISION_ARGS+=(--fp16)
fi
if [[ "${BF16}" != "0" ]]; then
  PRECISION_ARGS+=(--bf16)
fi

QUANT_ARGS=()
if [[ "${LOAD_IN_4BIT}" != "0" ]]; then
  QUANT_ARGS+=(--load-in-4bit)
fi
if [[ "${LOAD_IN_8BIT}" != "0" ]]; then
  QUANT_ARGS+=(--load-in-8bit)
fi
if [[ -n "${DEVICE_MAP}" ]]; then
  QUANT_ARGS+=(--device-map "${DEVICE_MAP}")
fi

GRAD_ARGS=()
if [[ "${GRADIENT_CHECKPOINTING}" != "0" ]]; then
  GRAD_ARGS+=(--gradient-checkpointing)
fi
if [[ "${CHECKPOINT_REENTRANT}" != "0" ]]; then
  GRAD_ARGS+=(--checkpoint-reentrant)
fi

accelerate launch \
  --main_process_port "${MASTER_PORT}" \
  --gpu_ids "${GPU_IDS}" \
  --num_processes "${NUM_GPUS}" \
  --config_file "${ACCELERATE_CFG}" \
  --multi_gpu \
  "${REPO_ROOT}/train_llm.py" \
  --algorithm sft \
  --train-dataset-path "${DATASET_TRAIN_PATH}" \
  --val-dataset-path "${DATASET_VAL_PATH}" \
  --model-name "${MODEL_NAME}" \
  --output-dir "${OUTPUT_DIR}" \
  --batch-size "${BATCH_SIZE}" \
  --gradient-accumulation "${GRAD_ACCUM}" \
  --gradient-checkpointing \
  --num-train-epochs "${NUM_EPOCHS}" \
  --learning-rate "${LEARNING_RATE}" \
  --max-length "${MAX_LENGTH}" \
  --system-field "${SYSTEM_FIELD}" \
  --user-field "${USER_FIELD}" \
  --system-role "${SYSTEM_ROLE}" \
  --user-role "${USER_ROLE}" \
  "${LORA_ARGS[@]}" \
  "${PRECISION_ARGS[@]}" \
  "${QUANT_ARGS[@]}" \
  "${GRAD_ARGS[@]}" \
  --seed "${SEED}" \
  "$@"
