#!/usr/bin/env bash
# Sequential P3 buffer-allocation pilot, then a clean 3-member Hybrid full run.
# A failed experiment is recorded and packaged; it never prevents the next one.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 2
PYTHON_BIN="${P3_PYTHON:-python}"
GPU_ID="${P3_GPU_ID:-0}"
MODE="${1:-run}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
PILOT_CONFIG="$ROOT/configs/p3_focal_equal_buffer_pilot.json"
FULL_CONFIG="$ROOT/configs/p3_hybrid_full8_e30_mem4_nodistill.json"
PILOT_ROOT="$ROOT/outputs/p3_focal_equal_buffer_pilot_seed0"
FULL_ROOT="$ROOT/outputs/p3_hybrid_full8_e30_mem4_nodistill_seed0"
SEQUENCE_ROOT="$ROOT/outputs/p3_experiments"
mkdir -p "$SEQUENCE_ROOT"
SEQUENCE_LOG="$SEQUENCE_ROOT/sequence_${STAMP}.log"

say() { printf '[%s] %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$SEQUENCE_LOG"; }

find_fold_root() {
  if [[ -n "${P3_FOLD_ROOT:-}" ]]; then printf '%s\n' "$P3_FOLD_ROOT"; return 0; fi
  local preferred="$ROOT/study_assets/stratified_3fold_seed42"
  if [[ -s "$preferred/fold_assignments.parquet" && -s "$preferred/fold_manifest.json" ]]; then
    printf '%s\n' "$preferred"; return 0
  fi
  local -a found=()
  mapfile -t found < <(find "$ROOT/outputs" -type f -name fold_assignments.parquet -printf '%h\n' 2>/dev/null | sort -u)
  if [[ ${#found[@]} -eq 1 && -s "${found[0]}/fold_manifest.json" ]]; then printf '%s\n' "${found[0]}"; return 0; fi
  return 1
}

find_prep_root() {
  if [[ -n "${P3_PREP_ROOT:-}" ]]; then printf '%s\n' "$P3_PREP_ROOT"; return 0; fi
  local preferred="$ROOT/study_assets/preprocessing_p3_seed0_fold42"
  if [[ -s "$preferred/member_0/B/fold_preprocessing.json" && -s "$preferred/member_1/B/fold_preprocessing.json" && -s "$preferred/member_2/B/fold_preprocessing.json" ]]; then
    printf '%s\n' "$preferred"; return 0
  fi
  local -a found=()
  mapfile -t found < <(find "$ROOT/study_assets" -type f -path '*/member_0/B/fold_preprocessing.json' -printf '%h\n' 2>/dev/null | sed 's#/member_0/B$##' | sort -u)
  if [[ ${#found[@]} -eq 1 ]]; then printf '%s\n' "${found[0]}"; return 0; fi
  return 1
}

FOLD_ROOT=""
PREP_ROOT=""
if FOLD_ROOT="$(find_fold_root)"; then say "fold root: $FOLD_ROOT"; else say "[WARN] Fold assets unresolved; pass P3_FOLD_ROOT=/.../folds"; fi
if PREP_ROOT="$(find_prep_root)"; then say "preprocessing root: $PREP_ROOT"; else say "[WARN] P3 preprocessing unresolved; pass P3_PREP_ROOT=/.../preprocessing_p3..."; fi

COMMON_ARGS=()
if [[ -n "$FOLD_ROOT" ]]; then COMMON_ARGS+=(--fold-assignments "$FOLD_ROOT/fold_assignments.parquet" --fold-manifest "$FOLD_ROOT/fold_manifest.json"); fi
if [[ -n "$PREP_ROOT" ]]; then COMMON_ARGS+=(--preprocessing-root "$PREP_ROOT"); fi

run_one() {
  local label="$1" config="$2" output_root="$3" member_scope="$4" with_ensemble="$5"
  local train_status=0 viz_status=0 package_status=0 log="$SEQUENCE_ROOT/${label}_${STAMP}.log"
  local archive="$SEQUENCE_ROOT/${label}_review_${STAMP}.zip"
  local status_file="$SEQUENCE_ROOT/${label}_${STAMP}.status.txt"
  say "START $label; output=$output_root; members=$member_scope"
  local -a command=("$PYTHON_BIN" src/training/fold_ensemble3_full.py --config "$config" --python "$PYTHON_BIN" "${COMMON_ARGS[@]}")
  if [[ "$member_scope" == "0" ]]; then command+=(--member-id 0); fi
  if [[ "$MODE" == "check" ]]; then
    "${command[@]}" > "$log" 2>&1
    train_status=$?
    if [[ $train_status -eq 0 ]]; then say "CHECK OK: $label"; else say "CHECK FAILED ($train_status): $label; continuing."; fi
    return 0
  fi
  command+=(--execute)
  CUDA_VISIBLE_DEVICES="$GPU_ID" PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "${command[@]}" > "$log" 2>&1
  train_status=$?
  if [[ $train_status -eq 0 ]]; then say "TRAIN OK: $label"; else say "TRAIN FAILED ($train_status): $label; continuing to post-run steps."; fi

  local -a viz=("$PYTHON_BIN" scripts/visualize_cil_run.py --run-root "$output_root" \
    --outdir "$output_root/visualizations" --member-id 0)
  if [[ "$member_scope" == "0" ]]; then
    viz+=(--member-ids 0)
  else
    viz+=(--ensemble --member-ids 0 1 2)
  fi
  "${viz[@]}" >> "$log" 2>&1
  viz_status=$?
  if [[ $viz_status -eq 0 ]]; then say "VISUALIZATIONS OK: $label"; else say "VISUALIZATIONS FAILED ($viz_status): $label; continuing."; fi

  printf 'experiment=%s\ntrain_exit=%s\nvisualization_exit=%s\n' "$label" "$train_status" "$viz_status" > "$status_file"
  "$PYTHON_BIN" scripts/package_p3_experiment_review.py --label "$label" --run-root "$output_root" \
    --config "$config" --log "$log" --status "train_exit=$train_status; visualization_exit=$viz_status" \
    --archive "$archive" >> "$log" 2>&1
  package_status=$?
  printf 'experiment=%s\ntrain_exit=%s\nvisualization_exit=%s\npackage_exit=%s\n' \
    "$label" "$train_status" "$viz_status" "$package_status" > "$status_file"
  if [[ $package_status -eq 0 ]]; then say "ZIP: $archive"; else say "PACKAGING FAILED ($package_status): $label; continuing."; fi
  return 0
}

case "$MODE" in
  check)
    run_one focal_equal_buffer_pilot "$PILOT_CONFIG" "$PILOT_ROOT" 0 no
    run_one hybrid_full "$FULL_CONFIG" "$FULL_ROOT" all yes
    say "Preflight checks finished (failures do not stop subsequent checks)."
    ;;
  run)
    run_one focal_equal_buffer_pilot "$PILOT_CONFIG" "$PILOT_ROOT" 0 no
    run_one hybrid_full "$FULL_CONFIG" "$FULL_ROOT" all yes
    say "Both experiments have been attempted. Review each .status.txt and .zip independently."
    ;;
  *) echo "Usage: bash scripts/run_p3_experiments.sh [check|run]" >&2; exit 2 ;;
esac
