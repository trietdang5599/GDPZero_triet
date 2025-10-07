#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON:-python}"
SIM_COUNTS=(10 20 30 40)
MODEL_DIRS=("gpt2-sft" "gpt2-dpo")
RUNNER_PATH="${REPO_ROOT}/runners/gdpzero.py"

if [[ ! -f "${RUNNER_PATH}" ]]; then
	echo "Runner script not found at ${RUNNER_PATH}" >&2
	exit 1
fi

echo "Running GDPZero for models: ${MODEL_DIRS[*]}"
echo "Simulation counts: ${SIM_COUNTS[*]}"
echo "Using python executable: ${PYTHON_BIN}"

for model_name in "${MODEL_DIRS[@]}"; do
	model_path="outputs/${model_name}"
	if [[ ! -d "${model_path}" ]]; then
		echo "Warning: model directory ${model_path} not found, skipping." >&2
		continue
	fi
	echo "\n##### Model: ${model_name} #####"
	for sims in "${SIM_COUNTS[@]}"; do
		output_file="outputs/gdpzero_${model_name}_${sims}.pkl"
		echo "\n=== ${model_name}: --num_mcts_sims ${sims} (output: ${output_file}) ==="
		"${PYTHON_BIN}" "${RUNNER_PATH}" \
			--output "${output_file}" \
			--llm local \
			--local-model-path "${model_path}" \
			--num_mcts_sims "${sims}" \
			--max_realizations 3 \
			--Q_0 0.25 \
			--debug \
			"$@"
	done
done

echo "\nAll runs completed."
