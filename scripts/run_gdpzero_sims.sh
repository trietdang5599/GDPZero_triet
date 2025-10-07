#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON:-python}"
SIM_COUNTS=(10 20 30 40)
OUTPUT_PREFIX="outputs/gdpzero"

echo "Running GDPZero for simulation counts: ${SIM_COUNTS[*]}"
echo "Using python executable: ${PYTHON_BIN}"

for sims in "${SIM_COUNTS[@]}"; do
	output_file="${OUTPUT_PREFIX}_${sims}.pkl"
	echo "\n=== Running with --num_mcts_sims ${sims} (output: ${output_file}) ==="
	"${PYTHON_BIN}" "${REPO_ROOT}/runners/gdpzero.py" \
		--llm gpt2 \
		--num_mcts_sims "${sims}" \
		--max_realizations 3 \
		--Q_0 0.25 \
		--debug \
		"$@"
done

echo "\nAll runs completed."
