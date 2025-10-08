#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON:-python}"
ACCELERATE_BIN="${ACCELERATE_BIN:-accelerate}"

MODEL_NAME="${MODEL_NAME:-meta-llama/Meta-Llama-3-8B-Instruct}"
DATASET_PATH="${REPO_ROOT}/data/p4g/300_dialog_turn_based.pkl"
PREF_PATH="${REPO_ROOT}/data/p4g/preferences.jsonl"
SFT_OUTPUT="${REPO_ROOT}/outputs/qwen25-sft"
DPO_OUTPUT="${REPO_ROOT}/outputs/qwen25-dpo"

NUM_GPUS="${NUM_GPUS:-1}"
MASTER_PORT="${MASTER_PORT:-29500}"

# tránh phân mảnh VRAM cho model lớn
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:512}"
# env NCCL an toàn single-node
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}"
# export NCCL_BLOCKING_WAIT="${NCCL_BLOCKING_WAIT:-1}"
unset NCCL_BLOCKING_WAIT
export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

echo "=== GDPZero Training Pipeline (accelerate) ==="
echo "Python executable : ${PYTHON_BIN}"
echo "Accelerate bin    : ${ACCELERATE_BIN}"
echo "Model name        : ${MODEL_NAME}"
echo "Dialog dataset    : ${DATASET_PATH}"
echo "Preference output : ${PREF_PATH}"
echo "SFT checkpoint    : ${SFT_OUTPUT}"
echo "DPO checkpoint    : ${DPO_OUTPUT}"
echo "GPUs requested    : ${NUM_GPUS}"

[[ -f "${DATASET_PATH}" ]] || { echo "Dataset not found at ${DATASET_PATH}" >&2; exit 1; }
mkdir -p "$(dirname "${PREF_PATH}")" "${SFT_OUTPUT}" "${DPO_OUTPUT}"

if [[ ! -f "${PREF_PATH}" ]]; then
  echo "[1/3] Building preference dataset..."
  "${PYTHON_BIN}" "${REPO_ROOT}/hf_train.py" build-preference-dataset \
    --dialog-path "${DATASET_PATH}" \
    --output "${PREF_PATH}" \
    --num-negatives 1
else
  echo "[1/3] Preference dataset already exists at ${PREF_PATH}, skipping."
fi

run_training () {
  if (( NUM_GPUS > 1 )); then
    # dùng accelerate launch để spawn DDP đúng chuẩn
	accelerate launch \
	--multi_gpu \
	--num_processes "${NUM_GPUS}" \
	--num_machines 1 \
	--mixed_precision no \
	--dynamo_backend no \
	--main_process_port "${MASTER_PORT}" \
	"${REPO_ROOT}/train_llm.py" "$@"

  else
    "${PYTHON_BIN}" "${REPO_ROOT}/train_llm.py" "$@"
  fi
}

echo "[2/3] Running supervised fine-tuning ..."
run_training \
  --algorithm sft \
  --dataset-path "${DATASET_PATH}" \
  --model-name "${MODEL_NAME}" \
  --output-dir "${SFT_OUTPUT}" \
  --batch-size 4 \
  --gradient-accumulation 16 \
  --num-train-epochs 2 \
  --learning-rate 2e-5 \
  --max-length 512 \
  "$@"

echo "[3/3] Running DPO preference optimization..."
run_training \
  --algorithm dpo \
  --dataset-path "${PREF_PATH}" \
  --model-name "${MODEL_NAME}" \
  --reference-model-name "${SFT_OUTPUT}" \
  --output-dir "${DPO_OUTPUT}" \
  --batch-size 4 \
  --gradient-accumulation 16 \
  --num-train-epochs 2 \
  --learning-rate 1e-5 \
  --max-length 512 \
  --dpo-beta 0.1 \
  "$@"

echo "Training pipeline complete."
echo "SFT checkpoint : ${SFT_OUTPUT}"
echo "DPO checkpoint : ${DPO_OUTPUT}"
