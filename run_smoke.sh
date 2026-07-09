#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/mnt/data/uyen/data_splits"
ENV_NAME="ddi2025-cil"
DEVICE="${DEVICE:-cuda}"

TRAIN_PARQUET="${PROJECT_ROOT}/train_extracted.parquet"
VALID_PARQUET="${PROJECT_ROOT}/validation_extracted.parquet"
TEST_PARQUET="${PROJECT_ROOT}/test_extracted.parquet"

FEATURE_COLS_JSON="${PROJECT_ROOT}/outputs/audit_smoke/feature_columns.json"
SCALER_PKL="${PROJECT_ROOT}/outputs/preprocess_smoke/scaler.pkl"

SEEDS=(0)

log() {
  printf '\n[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$1"
}

ensure_conda() {
  if ! command -v conda >/dev/null 2>&1; then
    echo "conda not found in PATH." >&2
    exit 1
  fi
}

activate_conda_env() {
  ensure_conda

  local conda_base
  conda_base="$(conda info --base)"
  # shellcheck disable=SC1091
  source "${conda_base}/etc/profile.d/conda.sh"

  if ! conda env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
    log "Creating conda environment ${ENV_NAME}"
    conda create -n "${ENV_NAME}" python=3.11 -y
  else
    log "Conda environment ${ENV_NAME} already exists"
  fi

  conda activate "${ENV_NAME}"
  log "Activated conda environment ${ENV_NAME}"
}

install_requirements() {
  log "Installing Python requirements"
  python -m pip install --upgrade pip setuptools wheel
  pip install -r "${PROJECT_ROOT}/requirements.txt"
}

prepare_directories() {
  mkdir -p \
    "${PROJECT_ROOT}/outputs" \
    "${PROJECT_ROOT}/outputs/audit_smoke" \
    "${PROJECT_ROOT}/outputs/class_distribution_smoke" \
    "${PROJECT_ROOT}/outputs/figures_smoke" \
    "${PROJECT_ROOT}/outputs/leakage_smoke" \
    "${PROJECT_ROOT}/outputs/preprocess_smoke" \
    "${PROJECT_ROOT}/outputs/tasks_smoke" \
    "${PROJECT_ROOT}/outputs/runs_smoke"
}

run_audit() {
  log "Running smoke schema audit"
  python "${PROJECT_ROOT}/scripts/inspect_splits.py" \
    --train "${TRAIN_PARQUET}" \
    --validation "${VALID_PARQUET}" \
    --test "${TEST_PARQUET}" \
    --outdir "${PROJECT_ROOT}/outputs/audit_smoke" \
    --batch-size 256 | tee "${PROJECT_ROOT}/outputs/audit_smoke/run.log"
}

run_class_distribution() {
  log "Running smoke class distribution analysis"
  python "${PROJECT_ROOT}/scripts/analyze_class_distribution.py" \
    --train "${TRAIN_PARQUET}" \
    --validation "${VALID_PARQUET}" \
    --test "${TEST_PARQUET}" \
    --outdir "${PROJECT_ROOT}/outputs/class_distribution_smoke" \
    --figdir "${PROJECT_ROOT}/outputs/figures_smoke" | tee "${PROJECT_ROOT}/outputs/class_distribution_smoke/run.log"
}

run_leakage() {
  log "Running smoke leakage audit"
  python "${PROJECT_ROOT}/scripts/check_leakage.py" \
    --train "${TRAIN_PARQUET}" \
    --validation "${VALID_PARQUET}" \
    --test "${TEST_PARQUET}" \
    --outdir "${PROJECT_ROOT}/outputs/leakage_smoke" | tee "${PROJECT_ROOT}/outputs/leakage_smoke/run.log"
}

run_preprocess() {
  log "Running smoke preprocessing fit"
  python "${PROJECT_ROOT}/scripts/preprocess_features.py" \
    --train "${TRAIN_PARQUET}" \
    --validation "${VALID_PARQUET}" \
    --test "${TEST_PARQUET}" \
    --feature-cols "${FEATURE_COLS_JSON}" \
    --outdir "${PROJECT_ROOT}/outputs/preprocess_smoke" \
    --scaler standard \
    --impute-strategy zero \
    --batch-size 256 \
    --max-batches 8 | tee "${PROJECT_ROOT}/outputs/preprocess_smoke/run.log"
}

run_task_builder() {
  log "Running smoke task builder"
  python "${PROJECT_ROOT}/scripts/build_cil_tasks.py" \
    --class-counts "${PROJECT_ROOT}/outputs/class_distribution_smoke/class_counts_train.csv" \
    --outdir "${PROJECT_ROOT}/outputs/tasks_smoke" \
    --protocol random \
    --num-classes 178 \
    --num-tasks 8 \
    --base-task-classes 38 \
    --increment-classes 20 \
    --seeds "${SEEDS[@]}" | tee "${PROJECT_ROOT}/outputs/tasks_smoke/run.log"
}

run_static_baseline() {
  local outdir="${PROJECT_ROOT}/outputs/runs_smoke/static_mlpbase"
  mkdir -p "${outdir}"

  log "Running smoke static MLP-base baseline"
  python "${PROJECT_ROOT}/src/training/train_static.py" \
    --train "${TRAIN_PARQUET}" \
    --validation "${VALID_PARQUET}" \
    --test "${TEST_PARQUET}" \
    --feature-cols "${FEATURE_COLS_JSON}" \
    --scaler "${SCALER_PKL}" \
    --outdir "${outdir}" \
    --variant small \
    --batch-size 128 \
    --epochs 1 \
    --patience 1 \
    --seed 0 \
    --device "${DEVICE}" \
    --max-train-rows 4096 \
    --max-validation-rows 1024 \
    --max-test-rows 1024
}

run_cil_smoke() {
  local outdir="${PROJECT_ROOT}/outputs/runs_smoke/random_seed0_sequential_mlpbase"
  mkdir -p "${outdir}"

  log "Running smoke CIL sequential baseline"
  python "${PROJECT_ROOT}/src/training/train_cil.py" \
    --train "${TRAIN_PARQUET}" \
    --validation "${VALID_PARQUET}" \
    --test "${TEST_PARQUET}" \
    --feature-cols "${FEATURE_COLS_JSON}" \
    --scaler "${SCALER_PKL}" \
    --task-file "${PROJECT_ROOT}/outputs/tasks_smoke/random_seed0_tasks.json" \
    --outdir "${outdir}" \
    --method sequential \
    --variant small \
    --batch-size 128 \
    --epochs 1 \
    --patience 1 \
    --seed 0 \
    --device "${DEVICE}" \
    --max-train-rows-per-task 4096 \
    --max-validation-rows-per-task 1024 \
    --max-test-rows-per-task 1024
}

main() {
  cd "${PROJECT_ROOT}"
  prepare_directories
  exec > >(tee -a "${PROJECT_ROOT}/outputs/run_smoke.log") 2>&1

  activate_conda_env
  install_requirements
  log "Training device set to ${DEVICE}"

  run_audit
  run_class_distribution
  run_leakage
  run_preprocess
  run_task_builder
  run_static_baseline
  run_cil_smoke

  log "Smoke pipeline completed"
}

main "$@"
