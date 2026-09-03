#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AGEM_PYTHON="${AGEM_PYTHON:-${PROJECT_ROOT}/.venv/bin/python}"
AGEM_DEVICE="${AGEM_DEVICE:-auto}"
AGEM_OUTPUT_ROOT="${AGEM_OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/runs_methods}"
AGEM_MEMORY_BUDGET="${AGEM_MEMORY_BUDGET:-500}"
AGEM_REFERENCE_BATCH_SIZE="${AGEM_REFERENCE_BATCH_SIZE:-256}"
AGEM_EPOCHS="${AGEM_EPOCHS:-20}"
AGEM_BATCH_SIZE="${AGEM_BATCH_SIZE:-1024}"
read -r -a SEEDS <<< "${AGEM_SEEDS:-0}"

if [[ ! -x "${AGEM_PYTHON}" ]]; then
  echo "Python environment not found: ${AGEM_PYTHON}" >&2
  exit 1
fi

mkdir -p "${AGEM_OUTPUT_ROOT}"

is_complete_run() {
  local run_dir="$1"
  [[ -s "${run_dir}/run_summary.md" ]] && \
    [[ -s "${run_dir}/metrics.csv" ]] && \
    [[ -s "${run_dir}/forgetting.csv" ]]
}

for seed in "${SEEDS[@]}"; do
  task_file="${AGEM_TASK_FILE:-${PROJECT_ROOT}/outputs/tasks/tail_to_head_tasks.json}"
  if [[ ! -s "${task_file}" ]]; then
    echo "Missing A-GEM task file: ${task_file}" >&2
    exit 1
  fi

  run_dir="${AGEM_OUTPUT_ROOT}/p3_seed${seed}_agem_tddi_paper_member"
  if [[ -n "${AGEM_RUN_DIR:-}" ]]; then
    if (( ${#SEEDS[@]} != 1 )); then
      echo "AGEM_RUN_DIR can only be used with one seed." >&2
      exit 1
    fi
    run_dir="${AGEM_RUN_DIR}"
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
  if [[ -n "${AGEM_MAX_TRAIN_ROWS_PER_TASK:-}" ]]; then
    limit_args+=(--max-train-rows-per-task "${AGEM_MAX_TRAIN_ROWS_PER_TASK}")
  fi
  if [[ -n "${AGEM_MAX_VALIDATION_ROWS_PER_TASK:-}" ]]; then
    limit_args+=(--max-validation-rows-per-task "${AGEM_MAX_VALIDATION_ROWS_PER_TASK}")
  fi
  if [[ -n "${AGEM_MAX_TEST_ROWS_PER_TASK:-}" ]]; then
    limit_args+=(--max-test-rows-per-task "${AGEM_MAX_TEST_ROWS_PER_TASK}")
  fi

  echo "[run] method=agem task_file=${task_file} seed=${seed} device=${AGEM_DEVICE} memory=${AGEM_MEMORY_BUDGET}"
  "${AGEM_PYTHON}" "${PROJECT_ROOT}/src/training/train_cil.py" \
    --train "${PROJECT_ROOT}/train_extracted.parquet" \
    --validation "${PROJECT_ROOT}/validation_extracted.parquet" \
    --test "${PROJECT_ROOT}/test_extracted.parquet" \
    --feature-cols "${PROJECT_ROOT}/outputs/audit/feature_columns.json" \
    --scaler "${PROJECT_ROOT}/outputs/preprocess/scaler.pkl" \
    --task-file "${task_file}" \
    --outdir "${run_dir}" \
    --method agem \
    --variant tddi_paper_member \
    --batch-size "${AGEM_BATCH_SIZE}" \
    --epochs "${AGEM_EPOCHS}" \
    --lr 0.001 \
    --weight-decay 0.0001 \
    --dropout 0.2 \
    --activation gelu \
    --norm layernorm \
    --patience 5 \
    --seed "${seed}" \
    --device "${AGEM_DEVICE}" \
    --episodic-memory-budget "${AGEM_MEMORY_BUDGET}" \
    --agem-reference-batch-size "${AGEM_REFERENCE_BATCH_SIZE}" \
    --focal-gamma 1.0 \
    "${limit_args[@]}"
done
