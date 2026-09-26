#!/usr/bin/env bash
set -uo pipefail

# Full protocol-diverse experiment: P2/member0 -> P3/member1 -> P4/member2,
# then common task-7 test ensemble, detailed report, visualizations and review ZIP.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 2
PYTHON_BIN="${PDE_PYTHON:-python}"
GPU_ID="${PDE_GPU_ID:-0}"
ACTION="${1:-start}"
TAG="${2:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_HOME="$ROOT/outputs/p3_experiments/protocol_diverse/run_$TAG"
OUT_ROOT="${PDE_OUT:-$ROOT/outputs/protocol_diverse_ensemble_seed0_$TAG}"
MONITOR_ROOT="$ROOT/outputs/p3_experiments/protocol_diverse"
LATEST="$MONITOR_ROOT/latest.txt"

die() { echo "[STOP] $*" >&2; exit 2; }

find_fold_root() {
  if [[ -n "${PDE_FOLD_ROOT:-${P3_FOLD_ROOT:-}}" ]]; then
    printf '%s\n' "${PDE_FOLD_ROOT:-$P3_FOLD_ROOT}"
    return 0
  fi
  local preferred="$ROOT/study_assets/stratified_3fold_seed42"
  if [[ -s "$preferred/fold_assignments.parquet" && -s "$preferred/fold_manifest.json" ]]; then
    printf '%s\n' "$preferred"
    return 0
  fi
  local -a found=()
  mapfile -t found < <(find "$ROOT/outputs" -type f -name fold_assignments.parquet -printf '%h\n' 2>/dev/null | sort -u)
  if [[ ${#found[@]} -eq 1 && -s "${found[0]}/fold_manifest.json" ]]; then
    printf '%s\n' "${found[0]}"
    return 0
  fi
  return 1
}

FOLD_ROOT="$(find_fold_root)" || die "Cannot identify a unique frozen fold root; export PDE_FOLD_ROOT=/path/to/folds."
[[ -s "$ROOT/train_extracted.parquet" ]] || die "Missing train_extracted.parquet"
[[ -s "$ROOT/validation_extracted.parquet" ]] || die "Missing validation_extracted.parquet"
[[ -s "$ROOT/test_extracted.parquet" ]] || die "Missing test_extracted.parquet"
[[ -s "$FOLD_ROOT/fold_assignments.parquet" && -s "$FOLD_ROOT/fold_manifest.json" ]] || die "Incomplete fold assets: $FOLD_ROOT"

COMMON=(
  "$PYTHON_BIN" "$ROOT/scripts/run_protocol_diverse_ensemble.py"
  --output-root "$OUT_ROOT"
  --train "$ROOT/train_extracted.parquet"
  --validation "$ROOT/validation_extracted.parquet"
  --test "$ROOT/test_extracted.parquet"
  --feature-cols "$ROOT/study_assets/data_schema/feature_columns.json"
  --fold-assignments "$FOLD_ROOT/fold_assignments.parquet"
  --fold-manifest "$FOLD_ROOT/fold_manifest.json"
  --python "$PYTHON_BIN"
)

preflight() {
  "${COMMON[@]}" check
}

run_all() {
  mkdir -p "$RUN_HOME" "$OUT_ROOT"
  local train_status=0 report_status=1 visualization_status=1 package_status=1
  "${COMMON[@]}" run > "$RUN_HOME/protocol_diverse.log" 2>&1 || train_status=$?
  [[ -s "$OUT_ROOT/final_results/PROTOCOL_DIVERSE_ENSEMBLE_RESULTS.md" ]] && report_status=0
  local expected_png=30 actual_png=0
  if [[ -d "$OUT_ROOT/visualizations" ]]; then
    actual_png="$(find "$OUT_ROOT/visualizations" -type f -name '*.png' | wc -l)"
  fi
  (( actual_png >= expected_png )) && visualization_status=0
  printf 'training_and_ensemble_exit=%s\nreport_exit=%s\nvisualization_exit=%s\n' \
    "$train_status" "$report_status" "$visualization_status" > "$OUT_ROOT/experiment_status.txt"
  local archive="$OUT_ROOT/review/protocol_diverse_ensemble_review_$TAG.zip"
  if "$PYTHON_BIN" "$ROOT/scripts/package_p3_experiment_review.py" \
      --label "Protocol-diverse ensemble P2/P3/P4 full run" \
      --run-root "$OUT_ROOT" \
      --config "$ROOT/configs/p3_hybrid_equal_buffer_full8_e30_mem4.json" \
      --status "training_and_ensemble_exit=$train_status; report_exit=$report_status; visualization_exit=$visualization_status" \
      --log "$RUN_HOME/protocol_diverse.log" --archive "$archive" \
      > "$RUN_HOME/package.log" 2>&1; then
    package_status=0
    echo "[OK] Review ZIP: $archive"
    echo "[OK] SHA256: $archive.sha256"
  else
    echo "[WARN] Packaging failed; inspect $RUN_HOME/package.log"
  fi
  echo "[OUTPUT] $OUT_ROOT"
  echo "[STATUS] train=$train_status report=$report_status figures=$visualization_status package=$package_status"
  return "$train_status"
}

case "$ACTION" in
  check)
    preflight || die "Protocol assets/config preflight failed."
    ;;
  run)
    preflight || die "Preflight failed; no training started."
    run_all
    ;;
  start)
    mkdir -p "$RUN_HOME"
    preflight > "$RUN_HOME/preflight.log" 2>&1 || {
      tail -n 40 "$RUN_HOME/preflight.log" >&2
      die "Preflight failed; no model training started."
    }
    printf '%s\n' "$RUN_HOME" > "$LATEST"
    printf '%s\n' "$OUT_ROOT" > "$RUN_HOME/output_root.txt"
    nohup env CUDA_VISIBLE_DEVICES="$GPU_ID" PYTHONUNBUFFERED=1 \
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      PDE_FOLD_ROOT="$FOLD_ROOT" PDE_OUT="$OUT_ROOT" PDE_PYTHON="$PYTHON_BIN" \
      bash "$ROOT/scripts/run_protocol_diverse_ensemble.sh" run "$TAG" \
      > "$RUN_HOME/nohup.log" 2>&1 < /dev/null &
    echo $! > "$RUN_HOME/job.pid"
    echo "[STARTED] P2/member0 -> P3/member1 -> P4/member2; full 8-task trajectories. PID=$(<"$RUN_HOME/job.pid")"
    echo "[LOG] $RUN_HOME/nohup.log"
    echo "[OUTPUT] $OUT_ROOT"
    ;;
  status|follow)
    [[ -s "$LATEST" ]] || die "No detached protocol-diverse run registered."
    RUN_HOME="$(<"$LATEST")"
    [[ -s "$RUN_HOME/job.pid" ]] || die "PID file missing: $RUN_HOME/job.pid"
    PID="$(<"$RUN_HOME/job.pid")"
    if [[ "$ACTION" == follow ]]; then
      tail -f "$RUN_HOME/nohup.log"
    else
      if kill -0 "$PID" 2>/dev/null; then echo "[RUNNING] PID=$PID"; else echo "[STOPPED/FINISHED] PID=$PID"; fi
      [[ -s "$RUN_HOME/output_root.txt" ]] && echo "[OUTPUT] $(<"$RUN_HOME/output_root.txt")"
      [[ -s "$RUN_HOME/nohup.log" ]] && tail -n 30 "$RUN_HOME/nohup.log"
      [[ -s "$RUN_HOME/package.log" ]] && tail -n 8 "$RUN_HOME/package.log"
    fi
    ;;
  *)
    echo "Usage: bash scripts/run_protocol_diverse_ensemble.sh [check|start|run|status|follow] [run_tag]" >&2
    exit 2
    ;;
esac
