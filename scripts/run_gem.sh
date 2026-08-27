#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GEM_PYTHON="${GEM_PYTHON:-${PROJECT_ROOT}/.venv/bin/python}"
GEM_DEVICE="${GEM_DEVICE:-auto}"
GEM_OUTPUT_ROOT="${GEM_OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/runs_methods}"
GEM_MEMORY_BUDGET="${GEM_MEMORY_BUDGET:-500}"
GEM_EPOCHS="${GEM_EPOCHS:-20}"
GEM_BATCH_SIZE="${GEM_BATCH_SIZE:-1024}"
read -r -a SEEDS <<< "${GEM_SEEDS:-0}"

if [[ ! -x "${GEM_PYTHON}" ]]; then
  echo "Python environment not found: ${GEM_PYTHON}" >&2
  exit 1
fi

mkdir -p "${GEM_OUTPUT_ROOT}"

is_complete_run() {
  local run_dir="$1"
  [[ -s "${run_dir}/run_summary.md" ]] && \
    [[ -s "${run_dir}/metrics.csv" ]] && \
    [[ -s "${run_dir}/forgetting.csv" ]]
}

for seed in "${SEEDS[@]}"; do
  task_file="${GEM_TASK_FILE:-${PROJECT_ROOT}/outputs/tasks/constrained_mass_balanced_seed${seed}_tasks.json}"
  if [[ ! -s "${task_file}" ]]; then
    echo "Missing GEM task file: ${task_file}" >&2
    exit 1
  fi

  run_dir="${GEM_OUTPUT_ROOT}/p4_seed${seed}_gem_mlptddi"
  if [[ -n "${GEM_RUN_DIR:-}" ]]; then
    if (( ${#SEEDS[@]} != 1 )); then
      echo "GEM_RUN_DIR can only be used with one seed." >&2
      exit 1
    fi
    run_dir="${GEM_RUN_DIR}"
  fi
  if is_complete_run "${run_dir}"; then
    echo "[skip] Complete: ${run_dir}"
    continue
  fi
  if [[ -d "${run_dir}" ]] && [[ -n "$(find "${run_dir}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "Refusing to overwrite non-empty run: ${run_dir}" >&2
    exit 1
  fi

  limit_args=()
  if [[ -n "${GEM_MAX_TRAIN_ROWS_PER_TASK:-}" ]]; then
    limit_args+=(--max-train-rows-per-task "${GEM_MAX_TRAIN_ROWS_PER_TASK}")
  fi
  if [[ -n "${GEM_MAX_VALIDATION_ROWS_PER_TASK:-}" ]]; then
    limit_args+=(--max-validation-rows-per-task "${GEM_MAX_VALIDATION_ROWS_PER_TASK}")
  fi
  if [[ -n "${GEM_MAX_TEST_ROWS_PER_TASK:-}" ]]; then
    limit_args+=(--max-test-rows-per-task "${GEM_MAX_TEST_ROWS_PER_TASK}")
  fi

  echo "[run] method=gem task_file=${task_file} seed=${seed} device=${GEM_DEVICE} memory=${GEM_MEMORY_BUDGET}"
  "${GEM_PYTHON}" "${PROJECT_ROOT}/src/training/train_cil.py" \
    --train "${PROJECT_ROOT}/train_extracted.parquet" \
    --validation "${PROJECT_ROOT}/validation_extracted.parquet" \
    --test "${PROJECT_ROOT}/test_extracted.parquet" \
    --feature-cols "${PROJECT_ROOT}/outputs/audit/feature_columns.json" \
    --scaler "${PROJECT_ROOT}/outputs/preprocess/scaler.pkl" \
    --task-file "${task_file}" \
    --outdir "${run_dir}" \
    --method gem \
    --variant tddi \
    --batch-size "${GEM_BATCH_SIZE}" \
    --epochs "${GEM_EPOCHS}" \
    --lr 0.001 \
    --weight-decay 0.0001 \
    --dropout 0.2 \
    --activation gelu \
    --norm layernorm \
    --patience 5 \
    --seed "${seed}" \
    --device "${GEM_DEVICE}" \
    --episodic-memory-budget "${GEM_MEMORY_BUDGET}" \
    --focal-gamma 1.0 \
    "${limit_args[@]}"
done
