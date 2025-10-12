#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON:-python}"
MODEL_PATH="${MODEL_PATH:-outputs/qwen25-dpo}"
SIM_COUNTS=(10 20)
NUM_DIALOGS="${NUM_DIALOGS:-30}"
LOG_DIR="${REPO_ROOT}/logs"
OUTPUT_DIR="${REPO_ROOT}/outputs"
MODEL_NAME="${MODEL_NAME:-qwen2.5-0.5b}"
OUTPUT_PREFIX="${OUTPUT_DIR}/gdpzero_${MODEL_NAME}_local"
EVAL_DIR="${OUTPUT_DIR}/evaluation"
JUDGE="${JUDGE:-Qwen/Qwen2.5-0.5B-Instruct}"

echo "Running GDPZero for simulation counts: ${SIM_COUNTS[*]}"
echo "Using python executable: ${PYTHON_BIN}"
echo "Using local model path: ${MODEL_PATH}"
echo "Using judge model: ${JUDGE}"

if [[ ! -d "${MODEL_PATH}" ]]; then
	echo "Model directory ${MODEL_PATH} not found. Pass the path as the first argument." >&2
	exit 1
fi

mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}" "${EVAL_DIR}"

for sims in "${SIM_COUNTS[@]}"; do
	output_file="${OUTPUT_PREFIX}_${sims}sims_${NUM_DIALOGS}.pkl"
	run_stamp="$(date +%Y%m%d_%H%M%S)"
	log_file="${LOG_DIR}/gdpzero_${MODEL_NAME}_${sims}_${run_stamp}.log"
	echo "\n=== Running with --local-model-path:${MODEL_PATH} --num_mcts_sims ${sims} ==="
	"${PYTHON_BIN}" "${REPO_ROOT}/runners/gdpzero.py" \
		--llm local \
		--local-model-path "${MODEL_PATH}" \
		--output "${output_file}" \
		--num_mcts_sims "${sims}" \
		--num_dialogs 30 \
		--max_realizations 3 \
		--Q_0 0.25 \
		"$@" | tee "${log_file}"

	eval_output_file="${EVAL_DIR}/$(basename "${output_file}" .pkl)_${run_stamp}_eval.pkl"
	echo "\n--- Evaluating ${output_file} (results -> ${eval_output_file}) ---"
	"${PYTHON_BIN}" "${REPO_ROOT}/test.py" \
		-f "${output_file}" \
		--judge "${JUDGE}" \
		--output "${eval_output_file}"
done

echo "\nAll runs completed."
