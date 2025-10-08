#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON:-python}"
MODEL_PATH="${1:-outputs/qwen25-dpo}"
SIM_COUNTS=(10 20)
OUTPUT_PREFIX="outputs/gdpzero_qwen25_dpo"
LOG_DIR="${REPO_ROOT}/logs"

echo "Running GDPZero for simulation counts: ${SIM_COUNTS[*]}"
echo "Using python executable: ${PYTHON_BIN}"
echo "Using local model path: ${MODEL_PATH}"

if [[ ! -d "${MODEL_PATH}" ]]; then
	echo "Model directory ${MODEL_PATH} not found. Pass the path as the first argument." >&2
	exit 1
fi

mkdir -p "${LOG_DIR}"

for sims in "${SIM_COUNTS[@]}"; do
	output_file="${OUTPUT_PREFIX}_${sims}.pkl"
	run_stamp="$(date +%Y%m%d_%H%M%S)"
	log_file="${LOG_DIR}/gdpzero_qwen25_${sims}_${run_stamp}.log"
	echo "\n=== Running with --num_mcts_sims ${sims} (output: ${output_file}) ==="
	"${PYTHON_BIN}" "${REPO_ROOT}/runners/gdpzero.py" \
		--llm local \
		--num_mcts_sims "${sims}" \
		--max_realizations 3 \
		--Q_0 0.25 \
		--output "${output_file}" \
		"$@" | tee "${log_file}"
done

echo "\nAll runs completed."
