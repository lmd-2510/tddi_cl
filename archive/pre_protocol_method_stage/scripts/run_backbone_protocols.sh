#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKBONE_PYTHON="${BACKBONE_PYTHON:-${PROJECT_ROOT}/.venv/bin/python}"
BACKBONE_DEVICE="${BACKBONE_DEVICE:-auto}"
BACKBONE_OUTPUT_ROOT="${BACKBONE_OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/runs_backbones}"
BACKBONE_METHOD="replay_distill_fixed_budget_uniform"
BACKBONE_SEEDS=(0 1 2 3 4)
BACKBONE_VARIANTS=(small base large)
BACKBONE_MIN_AVAILABLE_GIB="${BACKBONE_MIN_AVAILABLE_GIB:-12}"
BACKBONE_MIN_DISK_GIB="${BACKBONE_MIN_DISK_GIB:-50}"

usage() {
  echo "Usage: $0 [P0|P1|P2|P3|P4|P5|P6|P7|P8|all] [small|base|large|tddi|all]" >&2
}

protocol_arg="${1:-all}"
variant_arg="${2:-all}"

case "${protocol_arg}" in
  P0|P1|P2|P3|P4|P5|P6|P7|P8) protocols=("${protocol_arg}") ;;
  all) protocols=(P0 P1 P2 P3 P4 P5 P6 P7 P8) ;;
  *) usage; exit 2 ;;
esac

case "${variant_arg}" in
  small|base|large|tddi) variants=("${variant_arg}") ;;
  all) variants=("${BACKBONE_VARIANTS[@]}") ;;
  *) usage; exit 2 ;;
esac

if [[ ! -x "${BACKBONE_PYTHON}" ]]; then
  echo "Python environment not found: ${BACKBONE_PYTHON}" >&2
  exit 1
fi

task_file_for_protocol() {
  case "$1" in
    P0) echo "${PROJECT_ROOT}/outputs/tasks/random_seed${2}_tasks.json" ;;
    P1) echo "${PROJECT_ROOT}/outputs/tasks/frequency_balanced_tasks.json" ;;
    P2) echo "${PROJECT_ROOT}/outputs/tasks/head_to_tail_tasks.json" ;;
    P3) echo "${PROJECT_ROOT}/outputs/tasks/tail_to_head_tasks.json" ;;
    P4) echo "${PROJECT_ROOT}/outputs/tasks/constrained_mass_balanced_seed${2}_tasks.json" ;;
    P5) echo "${PROJECT_ROOT}/outputs/tasks/multi_factor_balanced_seed${2}_tasks.json" ;;
    P6) echo "${PROJECT_ROOT}/outputs/tasks/difficulty_balanced_seed${2}_tasks.json" ;;
    P7) echo "${PROJECT_ROOT}/outputs/tasks/confusion_spread_seed${2}_tasks.json" ;;
    P8) echo "${PROJECT_ROOT}/outputs/tasks/controlled_rarity_drift_seed${2}_tasks.json" ;;
  esac
}

slug_for_protocol() {
  case "$1" in
    P0) echo "p0_random" ;;
    P1) echo "p1_frequency_balanced" ;;
    P2) echo "p2_head_to_tail" ;;
    P3) echo "p3_tail_to_head" ;;
    P4) echo "p4_constrained_mass_balanced" ;;
    P5) echo "p5_multi_factor_balanced" ;;
    P6) echo "p6_difficulty_balanced" ;;
    P7) echo "p7_confusion_spread" ;;
    P8) echo "p8_controlled_rarity_drift" ;;
  esac
}

preflight_resource_guard() {
  local available_kb
  local disk_kb
  available_kb="$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)"
  disk_kb="$(df -Pk "${BACKBONE_OUTPUT_ROOT}" | awk 'NR == 2 {print $4}')"
  if (( available_kb < BACKBONE_MIN_AVAILABLE_GIB * 1024 * 1024 )); then
    echo "Resource guard: available RAM is below ${BACKBONE_MIN_AVAILABLE_GIB} GiB." >&2
    exit 1
  fi
  if (( disk_kb < BACKBONE_MIN_DISK_GIB * 1024 * 1024 )); then
    echo "Resource guard: free disk is below ${BACKBONE_MIN_DISK_GIB} GiB." >&2
    exit 1
  fi
}

is_complete_run() {
  local run_dir="$1"
  [[ -s "${run_dir}/run_summary.md" ]] && \
    [[ -s "${run_dir}/metrics.csv" ]] && \
    [[ -s "${run_dir}/forgetting.csv" ]]
}

mkdir -p "${BACKBONE_OUTPUT_ROOT}"

for protocol in "${protocols[@]}"; do
  protocol_slug="$(slug_for_protocol "${protocol}")"
  for variant in "${variants[@]}"; do
    for seed in "${BACKBONE_SEEDS[@]}"; do
      preflight_resource_guard
      task_file="$(task_file_for_protocol "${protocol}" "${seed}")"
      if [[ ! -s "${task_file}" ]]; then
        echo "Missing task file for ${protocol} seed=${seed}: ${task_file}" >&2
        if [[ "${protocol}" =~ ^P[0-4]$ ]]; then
          echo "Build P0-P4 with scripts/build_cil_tasks.py first." >&2
        else
          echo "Build P5-P8 with scripts/prepare_advanced_protocol_signals.py and scripts/build_advanced_protocols.py first." >&2
        fi
        echo "See docs/branches/running_guildes/p05_p08_guide.md for reproducible commands." >&2
        exit 1
      fi
      run_dir="${BACKBONE_OUTPUT_ROOT}/${protocol_slug}_seed${seed}_${BACKBONE_METHOD}_mlp${variant}"
      if is_complete_run "${run_dir}"; then
        echo "[skip] Complete: ${run_dir}"
        continue
      fi
      if [[ -d "${run_dir}" ]] && [[ -n "$(find "${run_dir}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
        echo "Refusing to overwrite incomplete run: ${run_dir}" >&2
        exit 1
      fi

      echo "[run] protocol=${protocol} variant=${variant} seed=${seed} device=${BACKBONE_DEVICE}"
      "${BACKBONE_PYTHON}" "${PROJECT_ROOT}/src/training/train_cil.py" \
        --train "${PROJECT_ROOT}/train_extracted.parquet" \
        --validation "${PROJECT_ROOT}/validation_extracted.parquet" \
        --test "${PROJECT_ROOT}/test_extracted.parquet" \
        --feature-cols "${PROJECT_ROOT}/outputs/audit/feature_columns.json" \
        --scaler "${PROJECT_ROOT}/outputs/preprocess/scaler.pkl" \
        --task-file "${task_file}" \
        --outdir "${run_dir}" \
        --method "${BACKBONE_METHOD}" \
        --variant "${variant}" \
        --batch-size 1024 \
        --epochs 20 \
        --patience 5 \
        --seed "${seed}" \
        --device "${BACKBONE_DEVICE}" \
        --total-memory-budget 6800 \
        --replay-draws-per-epoch 6800
    done
  done
done
