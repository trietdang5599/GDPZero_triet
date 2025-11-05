#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON:-python}"

DATASET_PATH="${DATASET_PATH:-${REPO_ROOT}/data/p4g/300_dialog_turn_based.pkl}"
OUTPUT_PATH="${OUTPUT_PATH:-${REPO_ROOT}/preference_pairs.jsonl}"
LLM_MODEL="${LLM_MODEL:-gpt-3.5-turbo}"
NUM_DIALOGS="${NUM_DIALOGS:-30}"
NUM_MCTS_SIMS="${NUM_MCTS_SIMS:-10}"
# ONLY_SUCCESS="${ONLY_SUCCESS:-1}"
# LOG_TURN_DETAILS="${LOG_TURN_DETAILS:-1}"

mkdir -p "$(dirname "${OUTPUT_PATH}")"

CMD=(
  "${PYTHON_BIN}"
  "${REPO_ROOT}/runners/generate_preference_pairs.py"
  --llm "${LLM_MODEL}"
  --num-dialogs "${NUM_DIALOGS}"
  --num-mcts-sims "${NUM_MCTS_SIMS}"
  --output "${OUTPUT_PATH}"
)

# if [[ "${ONLY_SUCCESS}" != "0" ]]; then
#   CMD+=(--only-success)
# fi
# if [[ "${LOG_TURN_DETAILS}" != "0" ]]; then
#   CMD+=(--log-turn-details)
# fi

echo "[build_prefs] Running: ${CMD[*]}"
"${CMD[@]}"
