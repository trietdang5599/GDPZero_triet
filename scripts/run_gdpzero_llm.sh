#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LLM="${LLM:-qwen2.5-7b}"
EVAL_DIR="${OUTPUT_DIR}/evaluation"
LOG_LEVEL="${LOG_LEVEL:-INFO}"

PYTHON_BIN="${PYTHON:-python}"
SIM_COUNTS=(10 20)
OUTPUT_PREFIX="outputs/gdpzero_base_${LLM}"
JUDGE="${JUDGE:-Qwen/Qwen2.5-0.5B-Instruct}"

echo "Running GDPZero for simulation counts: ${SIM_COUNTS[*]}"
echo "Using python executable: ${PYTHON_BIN}"
echo "Using LLM: ${LLM}"
echo "Log level: ${LOG_LEVEL}"

for sims in "${SIM_COUNTS[@]}"; do
	output_file="${OUTPUT_PREFIX}_${sims}sims.pkl"
	echo "\n=== Running with --num_mcts_sims ${sims} (output: ${output_file}) ==="
	"${PYTHON_BIN}" "${REPO_ROOT}/runners/gdpzero.py" \
		--llm "${LLM}" \
		--num_mcts_sims "${sims}" \
		--max_realizations 3 \
		--num_dialogs 30 \
		--Q_0 0.25 \
		--output "${output_file}" \
		--log-level "${LOG_LEVEL}" \
		"$@"
	eval_output_file="${EVAL_DIR}/$(basename "${output_file}" .pkl)_${run_stamp}_eval.pkl"
	echo "\n--- Evaluating ${output_file} (results -> ${eval_output_file}) ---"
	"${PYTHON_BIN}" "${REPO_ROOT}/test.py" \
		-f "${output_file}" \
		--judge "${JUDGE}" \
		--output "${eval_output_file}"
done

echo "\nAll runs completed."
