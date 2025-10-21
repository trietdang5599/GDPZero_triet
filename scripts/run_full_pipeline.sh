#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL_NAME="${MODEL_NAME:-meta-llama/Meta-Llama-3-8B-Instruct}"
DATASET_PATH="${DATASET_PATH:-${REPO_ROOT}/data/p4g/300_dialog_turn_based.pkl}"
PREF_PATH="${PREF_PATH:-${REPO_ROOT}/preference_pairs.jsonl}"
SFT_OUTPUT="${SFT_OUTPUT:-${REPO_ROOT}/outputs/${MODEL_NAME//\//_}-sft}"
DPO_OUTPUT="${DPO_OUTPUT:-${REPO_ROOT}/outputs/${MODEL_NAME//\//_}-dpo}"

echo "=== Step 1/3: Build preference pairs ==="
OUTPUT_PATH="${PREF_PATH}" \
DATASET_PATH="${DATASET_PATH}" \
bash "${SCRIPT_DIR}/run_build_prefs.sh" "$@"

echo "=== Step 2/3: Supervised fine-tuning ==="
OUTPUT_DIR="${SFT_OUTPUT}" \
MODEL_NAME="${MODEL_NAME}" \
DATASET_PATH="${DATASET_PATH}" \
bash "${SCRIPT_DIR}/run_sft.sh" "$@"

echo "=== Step 3/3: DPO fine-tuning ==="
OUTPUT_DIR="${DPO_OUTPUT}" \
MODEL_NAME="${MODEL_NAME}" \
SFT_MODEL_PATH="${SFT_OUTPUT}" \
PREF_PATH="${PREF_PATH}" \
bash "${SCRIPT_DIR}/run_dpo.sh" "$@"
