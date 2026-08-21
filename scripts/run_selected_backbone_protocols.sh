#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STUDY_PYTHON="${STUDY_PYTHON:-${PROJECT_ROOT}/.venv/bin/python}"
STUDY_DEVICE="${STUDY_DEVICE:-cuda}"
STUDY_OUTPUT_ROOT="${STUDY_OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/runs_selected_backbones}"
STUDY_MIN_AVAILABLE_GIB="${STUDY_MIN_AVAILABLE_GIB:-12}"
STUDY_MIN_DISK_GIB="${STUDY_MIN_DISK_GIB:-50}"
STUDY_MIN_GPU_FREE_MIB="${STUDY_MIN_GPU_FREE_MIB:-10000}"
read -r -a SEEDS <<< "${BACKBONE_SEEDS:-0 1 2 3 4}"

METHOD="replay_distill_fixed_budget_uniform"
SELECTED_PROTOCOLS=(P4 P6 P0 P2 P3)
SELECTED_BACKBONES=(tabm ddi_gcn)

usage() {
  echo "Usage: $0 [all|tabm|ddi_gcn] [all|P4|P6|P0|P2|P3]" >&2
  echo "Pilot example: BACKBONE_SEEDS=0 $0 all all" >&2
}

backbone_arg="${1:-all}"
protocol_arg="${2:-all}"
case "${backbone_arg}" in
  all) backbones=("${SELECTED_BACKBONES[@]}") ;;
  tabm|ddi_gcn) backbones=("${backbone_arg}") ;;
  *) usage; exit 2 ;;
esac
case "${protocol_arg}" in
  all) protocols=("${SELECTED_PROTOCOLS[@]}") ;;
  P4|P6|P0|P2|P3) protocols=("${protocol_arg}") ;;
  *) usage; exit 2 ;;
esac

if [[ ! -x "${STUDY_PYTHON}" ]]; then
  echo "Python environment not found: ${STUDY_PYTHON}" >&2
  exit 1
fi

GRAPH_CACHE="${PROJECT_ROOT}/outputs/graph_cache/ddi_gcn/graphs.npz"
GRAPH_MAPPING="${PROJECT_ROOT}/outputs/graph_cache/ddi_gcn/drug_id_mapping.json"
if [[ " ${backbones[*]} " == *" ddi_gcn "* ]] && \
   { [[ ! -s "${GRAPH_CACHE}" ]] || [[ ! -s "${GRAPH_MAPPING}" ]]; }; then
  echo "DDI-GCN graph cache is missing. Build it using docs/BACKBONE_STUDY.md." >&2
  exit 1
fi

task_file_for_protocol() {
  case "$1" in
    P0) echo "${PROJECT_ROOT}/outputs/tasks/random_seed${2}_tasks.json" ;;
    P2) echo "${PROJECT_ROOT}/outputs/tasks/head_to_tail_tasks.json" ;;
    P3) echo "${PROJECT_ROOT}/outputs/tasks/tail_to_head_tasks.json" ;;
    P4) echo "${PROJECT_ROOT}/outputs/tasks/constrained_mass_balanced_seed${2}_tasks.json" ;;
    P6) echo "${PROJECT_ROOT}/outputs/tasks/difficulty_balanced_seed${2}_tasks.json" ;;
  esac
}

slug_for_protocol() {
  case "$1" in
    P0) echo "p0_random" ;;
    P2) echo "p2_head_to_tail" ;;
    P3) echo "p3_tail_to_head" ;;
    P4) echo "p4_constrained_mass_balanced" ;;
    P6) echo "p6_difficulty_balanced" ;;
  esac
}

preflight_resource_guard() {
  local available_kb disk_kb gpu_free_mib
  available_kb="$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)"
  disk_kb="$(df -Pk "${STUDY_OUTPUT_ROOT}" | awk 'NR == 2 {print $4}')"
  if (( available_kb < STUDY_MIN_AVAILABLE_GIB * 1024 * 1024 )); then
    echo "Resource guard: available RAM is below ${STUDY_MIN_AVAILABLE_GIB} GiB." >&2
    exit 1
  fi
  if (( disk_kb < STUDY_MIN_DISK_GIB * 1024 * 1024 )); then
    echo "Resource guard: free disk is below ${STUDY_MIN_DISK_GIB} GiB." >&2
    exit 1
  fi
  if [[ "${STUDY_DEVICE}" == cuda* ]]; then
    if ! command -v nvidia-smi >/dev/null 2>&1; then
      echo "Resource guard: nvidia-smi is unavailable for a CUDA run." >&2
      exit 1
    fi
    gpu_free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sort -nr | head -1 | tr -d ' ')"
    if [[ -z "${gpu_free_mib}" ]] || (( gpu_free_mib < STUDY_MIN_GPU_FREE_MIB )); then
      echo "Resource guard: free GPU memory is below ${STUDY_MIN_GPU_FREE_MIB} MiB." >&2
      exit 1
    fi
  fi
}

is_complete_run() {
  local run_dir="$1"
  [[ -s "${run_dir}/run_summary.md" ]] && \
    [[ -s "${run_dir}/metrics.csv" ]] && \
    [[ -s "${run_dir}/forgetting.csv" ]]
}

mkdir -p "${STUDY_OUTPUT_ROOT}"

for backbone in "${backbones[@]}"; do
  for protocol in "${protocols[@]}"; do
    protocol_slug="$(slug_for_protocol "${protocol}")"
    for seed in "${SEEDS[@]}"; do
      preflight_resource_guard
      task_file="$(task_file_for_protocol "${protocol}" "${seed}")"
      if [[ ! -s "${task_file}" ]]; then
        echo "Missing task file: ${task_file}" >&2
        exit 1
      fi

      run_dir="${STUDY_OUTPUT_ROOT}/${backbone}_${protocol_slug}_seed${seed}_${METHOD}"
      if is_complete_run "${run_dir}"; then
        echo "[skip] Complete: ${run_dir}"
        continue
      fi
      if [[ -d "${run_dir}" ]] && [[ -n "$(find "${run_dir}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
        echo "Refusing to overwrite incomplete run: ${run_dir}" >&2
        exit 1
      fi

      model_args=()
      if [[ "${backbone}" == "tabm" ]]; then
        model_args+=(--tabm-k 32 --tabm-blocks 3 --tabm-d-block 512 --tabm-dropout 0.1)
      else
        model_args+=(
          --graph-cache "${GRAPH_CACHE}"
          --graph-mapping "${GRAPH_MAPPING}"
          --ddi-gcn-depth 8
          --ddi-gcn-width 128
          --ddi-gcn-attention-dim 65
        )
      fi

      echo "[run] backbone=${backbone} protocol=${protocol} seed=${seed} device=${STUDY_DEVICE}"
      "${STUDY_PYTHON}" "${PROJECT_ROOT}/src/training/train_cil.py" \
        --train "${PROJECT_ROOT}/train_extracted.parquet" \
        --validation "${PROJECT_ROOT}/validation_extracted.parquet" \
        --test "${PROJECT_ROOT}/test_extracted.parquet" \
        --feature-cols "${PROJECT_ROOT}/outputs/audit/feature_columns.json" \
        --scaler "${PROJECT_ROOT}/outputs/preprocess/scaler.pkl" \
        --task-file "${task_file}" \
        --outdir "${run_dir}" \
        --method "${METHOD}" \
        --variant "${backbone}" \
        --batch-size 256 \
        --effective-batch-size 1024 \
        --epochs 20 \
        --lr 0.001 \
        --weight-decay 0.0001 \
        --patience 5 \
        --seed "${seed}" \
        --device "${STUDY_DEVICE}" \
        --total-memory-budget 6800 \
        --replay-draws-per-epoch 6800 \
        --distill-alpha 1.0 \
        --temperature 2.0 \
        --feature-distill-weight 0.5 \
        --focal-gamma 1.0 \
        "${model_args[@]}"
    done
  done
done
