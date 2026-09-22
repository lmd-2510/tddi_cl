#!/usr/bin/env bash
set -euo pipefail

# Create the P3 task-0-frozen preprocessing assets from an existing 3-fold build.
# This does not train a model and never overwrites an existing namespace.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${P3_PYTHON:-python}"
TRAIN="$ROOT/train_extracted.parquet"
VALIDATION="$ROOT/validation_extracted.parquet"
TEST="$ROOT/test_extracted.parquet"
FEATURES="$ROOT/study_assets/data_schema/feature_columns.json"
TASK="$ROOT/study_assets/task_protocols/tail_to_head_tasks.json"
PREP_ROOT="${P3_PREP_ROOT:-$ROOT/study_assets/preprocessing_p3_seed0_fold42}"

die() { echo "[STOP] $*" >&2; exit 1; }

if [[ -n "${P3_FOLD_ROOT:-}" ]]; then
  FOLD_ROOT="$P3_FOLD_ROOT"
else
  mapfile -t candidates < <(
    find "$ROOT/outputs" -type f -name fold_assignments.parquet \
      -printf '%h\n' 2>/dev/null | sort -u
  )
  (( ${#candidates[@]} == 1 )) || die "Set P3_FOLD_ROOT to the single fold-preparation directory"
  FOLD_ROOT="${candidates[0]}"
fi

for file in "$TRAIN" "$VALIDATION" "$TEST" "$FEATURES" "$TASK" \
  "$FOLD_ROOT/fold_assignments.parquet" "$FOLD_ROOT/fold_manifest.json"; do
  [[ -s "$file" ]] || die "Missing: $file"
done

expected_hash="0d64c465b0c4bd34f66e6c76088b6b73fd60839ade3e56017b5fd36c21a26e79"
actual_hash="$(sha256sum "$TASK" | awk '{print $1}')"
[[ "$actual_hash" == "$expected_hash" ]] || die "P3 task-file SHA256 mismatch: $actual_hash"

echo "[INFO] fold root: $FOLD_ROOT"
echo "[INFO] P3 preprocessing root: $PREP_ROOT"

for member in 0 1 2; do
  out="$PREP_ROOT/member_${member}/B"
  artifact="$out/fold_preprocessing.json"
  if [[ -s "$artifact" ]]; then
    echo "[SKIP] member $member already exists: $artifact"
    continue
  fi
  [[ ! -e "$out" ]] || die "Partial namespace exists; inspect/remove only $out before retrying"
  "$PYTHON_BIN" scripts/prepare_fold_preprocessing.py \
    --assignments "$FOLD_ROOT/fold_assignments.parquet" \
    --manifest "$FOLD_ROOT/fold_manifest.json" \
    --train "$TRAIN" --validation "$VALIDATION" --test "$TEST" \
    --task-file "$TASK" --feature-cols "$FEATURES" \
    --member-id "$member" --validation-fold "$member" \
    --experiment-seed 0 --fold-seed 42 \
    --policy task0_standard_frozen --batch-size 2048 --outdir "$out"
done

for member in 0 1 2; do
  test -s "$PREP_ROOT/member_${member}/B/fold_preprocessing.json" \
    || die "Missing P3 preprocessing member $member"
done
echo "[OK] P3 study assets ready: $PREP_ROOT"
