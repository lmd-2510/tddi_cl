#!/usr/bin/env bash
set -euo pipefail

# PROJECT_ROOT="/mnt/data/uyen/data_splits"
# ENV_NAME="ddi2025-cil"
# DEVICE="${DEVICE:-cuda}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEVICE="${DEVICE:-mps}"

TRAIN_PARQUET="${PROJECT_ROOT}/train_extracted.parquet"
VALID_PARQUET="${PROJECT_ROOT}/validation_extracted.parquet"
TEST_PARQUET="${PROJECT_ROOT}/test_extracted.parquet"

FEATURE_COLS_JSON="${PROJECT_ROOT}/outputs/audit/feature_columns.json"
SCALER_PKL="${PROJECT_ROOT}/outputs/preprocess/scaler.pkl"

SEEDS=(0 1 2 3 4)

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
    "${PROJECT_ROOT}/outputs/audit" \
    "${PROJECT_ROOT}/outputs/class_distribution" \
    "${PROJECT_ROOT}/outputs/figures" \
    "${PROJECT_ROOT}/outputs/leakage" \
    "${PROJECT_ROOT}/outputs/preprocess" \
    "${PROJECT_ROOT}/outputs/tasks" \
    "${PROJECT_ROOT}/outputs/runs"
}

run_audit() {
  log "Running schema audit"
  python "${PROJECT_ROOT}/scripts/inspect_splits.py" \
    --train "${TRAIN_PARQUET}" \
    --validation "${VALID_PARQUET}" \
    --test "${TEST_PARQUET}" \
    --outdir "${PROJECT_ROOT}/outputs/audit" \
    --batch-size 1024 | tee "${PROJECT_ROOT}/outputs/audit/run.log"
}

run_class_distribution() {
  log "Running class distribution analysis"
  python "${PROJECT_ROOT}/scripts/analyze_class_distribution.py" \
    --train "${TRAIN_PARQUET}" \
    --validation "${VALID_PARQUET}" \
    --test "${TEST_PARQUET}" \
    --outdir "${PROJECT_ROOT}/outputs/class_distribution" \
    --figdir "${PROJECT_ROOT}/outputs/figures" | tee "${PROJECT_ROOT}/outputs/class_distribution/run.log"
}

run_leakage() {
  log "Running leakage audit"
  python "${PROJECT_ROOT}/scripts/check_leakage.py" \
    --train "${TRAIN_PARQUET}" \
    --validation "${VALID_PARQUET}" \
    --test "${TEST_PARQUET}" \
    --outdir "${PROJECT_ROOT}/outputs/leakage" | tee "${PROJECT_ROOT}/outputs/leakage/run.log"
}

run_preprocess() {
  log "Fitting preprocessing artifacts"
  python "${PROJECT_ROOT}/scripts/preprocess_features.py" \
    --train "${TRAIN_PARQUET}" \
    --validation "${VALID_PARQUET}" \
    --test "${TEST_PARQUET}" \
    --feature-cols "${FEATURE_COLS_JSON}" \
    --outdir "${PROJECT_ROOT}/outputs/preprocess" \
    --scaler standard \
    --impute-strategy zero \
    --batch-size 1024 | tee "${PROJECT_ROOT}/outputs/preprocess/run.log"
}

run_task_builder() {
  log "Building CIL task files"
  python "${PROJECT_ROOT}/scripts/build_cil_tasks.py" \
    --class-counts "${PROJECT_ROOT}/outputs/class_distribution/class_counts_train.csv" \
    --outdir "${PROJECT_ROOT}/outputs/tasks" \
    --protocol all \
    --num-classes 178 \
    --num-tasks 8 \
    --base-task-classes 38 \
    --increment-classes 20 \
    --seeds "${SEEDS[@]}" | tee "${PROJECT_ROOT}/outputs/tasks/run.log"
}

run_static_baseline() {
  local outdir="${PROJECT_ROOT}/outputs/runs/static_mlpbase"
  mkdir -p "${outdir}"

  log "Running static MLP-base baseline"
  python "${PROJECT_ROOT}/src/training/train_static.py" \
    --train "${TRAIN_PARQUET}" \
    --validation "${VALID_PARQUET}" \
    --test "${TEST_PARQUET}" \
    --feature-cols "${FEATURE_COLS_JSON}" \
    --scaler "${SCALER_PKL}" \
    --outdir "${outdir}" \
    --variant base \
    --batch-size 512 \
    --epochs 20 \
    --patience 5 \
    --seed 0 \
    --device "${DEVICE}"
}

run_cil_method_for_all_seeds() {
  local method="$1"
  shift
  local extra_args=("$@")

  for seed in "${SEEDS[@]}"; do
    local task_file="${PROJECT_ROOT}/outputs/tasks/random_seed${seed}_tasks.json"
    local outdir="${PROJECT_ROOT}/outputs/runs/random_seed${seed}_${method}_mlpbase"
    mkdir -p "${outdir}"

    log "Running ${method} for seed ${seed}"
    python "${PROJECT_ROOT}/src/training/train_cil.py" \
      --train "${TRAIN_PARQUET}" \
      --validation "${VALID_PARQUET}" \
      --test "${TEST_PARQUET}" \
      --feature-cols "${FEATURE_COLS_JSON}" \
      --scaler "${SCALER_PKL}" \
      --task-file "${task_file}" \
      --outdir "${outdir}" \
      --method "${method}" \
      --variant base \
      --batch-size 512 \
      --epochs 20 \
      --patience 5 \
      --seed "${seed}" \
      --device "${DEVICE}" \
      "${extra_args[@]}"
  done
}

main() {
  cd "${PROJECT_ROOT}"
  prepare_directories
  exec > >(tee -a "${PROJECT_ROOT}/outputs/run_full.log") 2>&1

  activate_conda_env
  install_requirements
  log "Training device set to ${DEVICE}"

  run_audit
  run_class_distribution
  run_leakage
  run_preprocess
  run_task_builder
  run_static_baseline

  run_cil_method_for_all_seeds "sequential"
  run_cil_method_for_all_seeds "replay" --memory-per-class 50
  run_cil_method_for_all_seeds "replay_distill" --memory-per-class 50 --distill-alpha 1.0 --temperature 2.0

  log "Full pipeline completed"
}

main "$@"
