#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL_NAME="${MODEL_NAME:-meta-llama/Meta-Llama-3-8B-Instruct}"
DATASET_PATH="${DATASET_PATH:-${REPO_ROOT}/data/p4g/300_dialog_turn_based.pkl}"
PREF_PATH="${PREF_PATH:-${REPO_ROOT}/preference_pairs.jsonl}"
SFT_OUTPUT="${SFT_OUTPUT:-${REPO_ROOT}/outputs/${MODEL_NAME//\//_}-sft}"
DPO_OUTPUT="${DPO_OUTPUT:-${REPO_ROOT}/outputs/${MODEL_NAME//\//_}-dpo}"

mkdir -p "$(dirname "${PREF_PATH}")" "${SFT_OUTPUT}" "${DPO_OUTPUT}"

echo "=== Step 1/3: Build preference pairs ==="
echo "Preference pairs will be saved to: ${PREF_PATH}"
OUTPUT_PATH="${PREF_PATH}" \
DATASET_PATH="${DATASET_PATH}" \
bash "${SCRIPT_DIR}/run_build_prefs.sh" "$@"
[[ -f "${PREF_PATH}" ]] || { echo "[error] Preference pairs not found at ${PREF_PATH}" >&2; exit 1; }

echo "=== Step 2/3: Supervised fine-tuning ==="
echo "SFT checkpoint directory: ${SFT_OUTPUT}"
OUTPUT_DIR="${SFT_OUTPUT}" \
MODEL_NAME="${MODEL_NAME}" \
DATASET_PATH="${DATASET_PATH}" \
bash "${SCRIPT_DIR}/run_sft.sh" "$@"
[[ -d "${SFT_OUTPUT}" ]] || { echo "[error] SFT output directory missing at ${SFT_OUTPUT}" >&2; exit 1; }

echo "=== Step 3/3: DPO fine-tuning ==="
echo "DPO checkpoint directory: ${DPO_OUTPUT}"
OUTPUT_DIR="${DPO_OUTPUT}" \
MODEL_NAME="${MODEL_NAME}" \
SFT_MODEL_PATH="${SFT_OUTPUT}" \
PREF_PATH="${PREF_PATH}" \
bash "${SCRIPT_DIR}/run_dpo.sh" "$@"
