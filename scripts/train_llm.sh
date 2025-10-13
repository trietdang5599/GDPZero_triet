#!/usr/bin/env bash
set -euo pipefail

NCCL_P2P_DISABLE=1
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON:-python}"
ACCELERATE_BIN="${ACCELERATE_BIN:-accelerate}"

MODEL_NAME="${MODEL_NAME:-meta-llama/Meta-Llama-3-8B-Instruct}"
DATASET_PATH="${REPO_ROOT}/data/p4g/300_dialog_turn_based.pkl"
PREF_PATH="${REPO_ROOT}/data/p4g/preference_pair.jsonl"
SFT_OUTPUT="${REPO_ROOT}/outputs/${MODEL_NAME//\//_}-sft"
DPO_OUTPUT="${REPO_ROOT}/outputs/${MODEL_NAME//\//_}-dpo"
DEFAULT_ACCELERATE_CFG="${REPO_ROOT}/config/accelerate_config.yaml"
if [[ -z "${ACCELERATE_CONFIG:-}" && -f "${DEFAULT_ACCELERATE_CFG}" ]]; then
	ACCELERATE_CONFIG="${DEFAULT_ACCELERATE_CFG}"
fi

USE_LORA="${USE_LORA:-1}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj}"

NUM_GPUS="${NUM_GPUS:-1}"        # số GPU bạn muốn dùng
MASTER_PORT="${MASTER_PORT:-0}"
GPU_IDS="${GPU_IDS:-}"           # ví dụ: "0,3" để dùng GPU 0 và 3

# tránh phân mảnh VRAM cho model lớn
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:512}"
# order ổn định
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# --- Detect available GPUs (theo hệ thống) ---
detect_gpus_py=$'import torch,os; print(torch.cuda.device_count() if torch.cuda.is_available() else 0)'
AVAILABLE_GPUS="$(${PYTHON_BIN} - <<PY
${detect_gpus_py}
PY
)"

# Nếu user chỉ định GPU_IDS thì dùng đúng y chang (và set NUM_GPUS = số id)
if [[ -n "${GPU_IDS}" ]]; then
  # chuẩn hoá bỏ khoảng trắng
  GPU_IDS="$(echo "${GPU_IDS}" | tr -d '[:space:]')"
  export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
  # đếm số id
  IFS=',' read -r -a _gpu_arr <<< "${GPU_IDS}"
  NUM_GPUS="${#_gpu_arr[@]}"
else
  # Không chỉ định GPU cụ thể → tạo dải 0..NUM_GPUS-1 nhưng không vượt AVAILABLE_GPUS
  if (( AVAILABLE_GPUS == 0 )); then
    # Không có GPU → fallback CPU
    unset CUDA_VISIBLE_DEVICES
    NUM_GPUS=0
  else
    # clamp NUM_GPUS vào [1..AVAILABLE_GPUS]
    if (( NUM_GPUS > AVAILABLE_GPUS )); then
      echo "[warn] Requested NUM_GPUS=${NUM_GPUS} > available ${AVAILABLE_GPUS}. Clamping to ${AVAILABLE_GPUS}."
      NUM_GPUS="${AVAILABLE_GPUS}"
    fi
    # build "0,1,2,..."
    ids=()
    for ((i=0; i<NUM_GPUS; i++)); do ids+=("$i"); done
    export CUDA_VISIBLE_DEVICES="$(IFS=,; echo "${ids[*]}")"
  fi
fi

# Nếu MASTER_PORT chưa được gán, chọn ngẫu nhiên một cổng rảnh để tránh xung đột giữa các job
if [[ "${MASTER_PORT}" == "0" || "${MASTER_PORT}" == "auto" ]]; then
	MASTER_PORT="$(${PYTHON_BIN} - <<'PY'
import socket
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind(("", 0))
    print(s.getsockname()[1])
PY
)"
fi

# NCCL env cho single-node
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}"
unset NCCL_BLOCKING_WAIT
export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
# transformers will try to import torchvision; disable to avoid GPU op mismatch
export TRANSFORMERS_NO_TORCHVISION="${TRANSFORMERS_NO_TORCHVISION:-1}"

echo "=== GDPZero Training Pipeline (accelerate) ==="
echo "Python executable : ${PYTHON_BIN}"
echo "Accelerate bin    : ${ACCELERATE_BIN}"
echo "Model name        : ${MODEL_NAME}"
echo "Dialog dataset    : ${DATASET_PATH}"
echo "Preference output : ${PREF_PATH}"
echo "SFT checkpoint    : ${SFT_OUTPUT}"
echo "DPO checkpoint    : ${DPO_OUTPUT}"
echo "AVAILABLE_GPUS    : ${AVAILABLE_GPUS}"
echo "CUDA_VISIBLE_DEVS : ${CUDA_VISIBLE_DEVICES:-<CPU>}"
echo "GPUs requested    : ${NUM_GPUS}"

[[ -f "${DATASET_PATH}" ]] || { echo "Dataset not found at ${DATASET_PATH}" >&2; exit 1; }
mkdir -p "$(dirname "${PREF_PATH}")" "${SFT_OUTPUT}" "${DPO_OUTPUT}"

if [[ ! -f "${PREF_PATH}" ]]; then
echo "[1/3] Building preference dataset..."
"${PYTHON_BIN}" "${REPO_ROOT}/runners/generate_preference_pairs.py" \
	--llm gpt-3.5-turbo \
	--only-success \
	--log-turn-details \
	--num-dialogs "${NUM_DIALOGS:-30}" \
	--num-mcts-sims "${NUM_MCTS_SIMS:-10}" \
	--output "${PREF_PATH}"
else
  echo "[1/3] Preference dataset already exists at ${PREF_PATH}, skipping."
fi

run_training () {
	if (( NUM_GPUS >= 1 )); then
		acc_args=(
			--main_process_port "${MASTER_PORT}"
			--num_machines 1
			--num_processes "${NUM_GPUS}"
			--mixed_precision no
			--dynamo_backend no
		)
		if (( NUM_GPUS >= 2 )); then
			acc_args+=(--multi_gpu)
		fi
		if [[ -n "${ACCELERATE_CONFIG:-}" ]]; then
			acc_args+=(--config_file "${ACCELERATE_CONFIG}")
		fi
		if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
			acc_args+=(--gpu_ids "${CUDA_VISIBLE_DEVICES}")
		fi
		"${ACCELERATE_BIN}" launch "${acc_args[@]}" "${REPO_ROOT}/train_llm.py" "$@"
	else
		# CPU fallback
		echo "[warn] No CUDA GPUs detected. Running on CPU."
		"${PYTHON_BIN}" "${REPO_ROOT}/train_llm.py" "$@"
fi
}

LORA_ARGS=()
if [[ "${USE_LORA}" != "0" ]]; then
	LORA_ARGS=(
		--use-lora
		--lora-r "${LORA_R}"
		--lora-alpha "${LORA_ALPHA}"
		--lora-dropout "${LORA_DROPOUT}"
		--lora-target-modules "${LORA_TARGET_MODULES}"
	)
fi

echo "[2/3] Running supervised fine-tuning ..."
run_training \
  --algorithm sft \
  --dataset-path "${DATASET_PATH}" \
  --model-name "${MODEL_NAME}" \
  --output-dir "${SFT_OUTPUT}" \
  --batch-size 4 \
  --gradient-accumulation 16 \
  --num-train-epochs 10 \
  --learning-rate 2e-5 \
  --max-length 512 \
  "${LORA_ARGS[@]}" \
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
  "${LORA_ARGS[@]}" \
  "$@"

echo "Training pipeline complete."
echo "SFT checkpoint : ${SFT_OUTPUT}"
echo "DPO checkpoint : ${DPO_OUTPUT}"
