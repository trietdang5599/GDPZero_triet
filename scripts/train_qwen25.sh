#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON:-python}"
MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
DATASET_PATH="${REPO_ROOT}/data/p4g/300_dialog_turn_based.pkl"
PREF_PATH="${REPO_ROOT}/data/p4g/preferences.jsonl"
SFT_OUTPUT="${REPO_ROOT}/outputs/qwen25-sft"
DPO_OUTPUT="${REPO_ROOT}/outputs/qwen25-dpo"

echo "=== GDPZero Qwen2.5-7B Training Pipeline ==="
echo "Python executable : ${PYTHON_BIN}"
echo "Model name        : ${MODEL_NAME}"
echo "Dialog dataset    : ${DATASET_PATH}"
echo "Preference output : ${PREF_PATH}"
echo "SFT checkpoint    : ${SFT_OUTPUT}"
echo "DPO checkpoint    : ${DPO_OUTPUT}"

if [[ ! -f "${DATASET_PATH}" ]]; then
	echo "Dataset not found at ${DATASET_PATH}" >&2
	exit 1
fi

mkdir -p "$(dirname "${PREF_PATH}")"
mkdir -p "${SFT_OUTPUT}" "${DPO_OUTPUT}"

if [[ ! -f "${PREF_PATH}" ]]; then
	echo "[1/3] Building preference dataset..."
	"${PYTHON_BIN}" "${REPO_ROOT}/hf_train.py" build-preference-dataset \
		--dialog-path "${DATASET_PATH}" \
		--output "${PREF_PATH}" \
		--num-negatives 1
else
	echo "[1/3] Preference dataset already exists at ${PREF_PATH}, skipping."
fi

echo "[2/3] Running supervised fine-tuning (QLoRA)..."
"${PYTHON_BIN}" "${REPO_ROOT}/train_llm.py" \
	--algorithm sft \
	--dataset-path "${DATASET_PATH}" \
	--model-name "${MODEL_NAME}" \
	--output-dir "${SFT_OUTPUT}" \
	--batch-size 1 \
	--gradient-accumulation 16 \
	--num-train-epochs 2 \
	--learning-rate 2e-5 \
	--max-length 512 \
	"$@"

echo "[3/3] Running DPO preference optimization..."
"${PYTHON_BIN}" "${REPO_ROOT}/train_llm.py" \
	--algorithm dpo \
	--dataset-path "${PREF_PATH}" \
	--model-name "${MODEL_NAME}" \
	--reference-model-name "${SFT_OUTPUT}" \
	--output-dir "${DPO_OUTPUT}" \
	--batch-size 1 \
	--gradient-accumulation 16 \
	--num-train-epochs 2 \
	--learning-rate 1e-5 \
	--max-length 512 \
	--dpo-beta 0.1 \
	"$@"

echo "Training pipeline complete."
echo "SFT checkpoint : ${SFT_OUTPUT}"
echo "DPO checkpoint : ${DPO_OUTPUT}"
