#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "Usage: $0 SEED DATA_ROOT OUTPUT_ROOT [PYTHON]" >&2
  exit 2
fi

S04_SEED=$1
S04_DATA_ROOT=$2
S04_OUTPUT_ROOT=$3
S04_PYTHON=${4:-.venv/bin/python}
S04_METHOD=replay_distill_fixed_budget_uniform
S04_OUTDIR="${S04_OUTPUT_ROOT}/random_seed${S04_SEED}_${S04_METHOD}_mlpbase"

if [[ ! "${S04_SEED}" =~ ^[0-4]$ ]]; then
  echo "SEED must be one of 0, 1, 2, 3, 4." >&2
  exit 2
fi
if [[ -d "${S04_OUTDIR}" ]] && [[ -n "$(find "${S04_OUTDIR}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "Refusing to overwrite non-empty run directory: ${S04_OUTDIR}" >&2
  exit 1
fi

mkdir -p "${S04_OUTPUT_ROOT}"
S04_AVAILABLE_KB=$(df -Pk "${S04_OUTPUT_ROOT}" | awk 'NR == 2 {print $4}')
if (( S04_AVAILABLE_KB < 35 * 1024 * 1024 )); then
  echo "S04 requires at least 35 GiB free before a full run." >&2
  exit 1
fi

"${S04_PYTHON}" -c \
  'import torch; assert torch.backends.mps.is_available(), "MPS is not available"'

export PYTORCH_ENABLE_MPS_FALLBACK=1
"${S04_PYTHON}" src/training/train_cil.py \
  --train "${S04_DATA_ROOT}/train_extracted.parquet" \
  --validation "${S04_DATA_ROOT}/validation_extracted.parquet" \
  --test "${S04_DATA_ROOT}/test_extracted.parquet" \
  --feature-cols "${S04_DATA_ROOT}/outputs/audit/feature_columns.json" \
  --scaler "${S04_DATA_ROOT}/outputs/preprocess/scaler.pkl" \
  --task-file "${S04_DATA_ROOT}/outputs/tasks/random_seed${S04_SEED}_tasks.json" \
  --outdir "${S04_OUTDIR}" \
  --method "${S04_METHOD}" \
  --variant base \
  --batch-size 1024 \
  --epochs 20 \
  --patience 5 \
  --seed "${S04_SEED}" \
  --device mps \
  --total-memory-budget 6800 \
  --replay-draws-per-epoch 6800 \
  --export-s02
