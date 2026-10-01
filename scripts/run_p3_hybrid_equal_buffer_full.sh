#!/usr/bin/env bash
# Full P3 runner: configurable experiment slug/config; defaults preserve the historical Hybrid run.
# Default action is detached nohup start; training and offline ensemble are sequential.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 2
source "$ROOT/scripts/lib/source_snapshot.sh"
PYTHON_BIN="${P3_PYTHON:-python}"
GPU_ID="${P3_GPU_ID:-0}"
ACTION="${1:-start}"
EXPERIMENT_SLUG="${P3_EXPERIMENT_SLUG:-p3_hybrid_equal_buffer_full8_e30_mem4}"
DISPLAY_LABEL="${P3_DISPLAY_LABEL:-P3 Hybrid + equal-class buffer full8 e30 mem4}"
LOSS_LABEL="${P3_LOSS_LABEL:-Hybrid (Focal current + CE replay)}"
CONFIG="${P3_CONFIG:-$ROOT/configs/p3_hybrid_equal_buffer_full8_e30_mem4.json}"
PROTOCOL_LABEL="${P3_PROTOCOL_LABEL:-P3}"
TASK_FILE="${P3_TASK_FILE:-$ROOT/study_assets/task_protocols/tail_to_head_tasks.json}"
TASK_SHA256="${P3_TASK_SHA256:-0d64c465b0c4bd34f66e6c76088b6b73fd60839ade3e56017b5fd36c21a26e79}"
THRESHOLD="${P3_THRESHOLD_CONFIG:-$ROOT/configs/eval_tddi_p3_ensemble_entropy_threshold.json}"
MONITOR_ROOT="${P3_MONITOR_ROOT:-$ROOT/outputs/p3_experiments/hybrid_equal_buffer}"
LATEST="$MONITOR_ROOT/latest.txt"

die() { echo "[STOP] $*" >&2; exit 2; }

latest_home() {
  [[ -s "$LATEST" ]] || die "No detached run registered under $MONITOR_ROOT"
  local value
  value="$(<"$LATEST")"
  [[ -d "$value" && "$value" == "$MONITOR_ROOT"/run_* ]] \
    || die "Invalid latest run pointer: $value"
  printf '%s\n' "$value"
}

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
  local preferred="${P3_PREP_PREFERRED:-$ROOT/study_assets/preprocessing_p3_seed0_fold42}"
  if [[ -s "$preferred/member_0/B/fold_preprocessing.json" && -s "$preferred/member_1/B/fold_preprocessing.json" && -s "$preferred/member_2/B/fold_preprocessing.json" ]]; then
    printf '%s\n' "$preferred"; return 0
  fi
  local -a found=()
  mapfile -t found < <(find "$ROOT/study_assets" -type f -path '*/member_0/B/fold_preprocessing.json' -printf '%h\n' 2>/dev/null | sed 's#/member_0/B$##' | sort -u)
  if [[ ${#found[@]} -eq 1 ]]; then printf '%s\n' "${found[0]}"; return 0; fi
  return 1
}

FOLD_ROOT="$(find_fold_root)" || die "Cannot uniquely find fold assignments; set P3_FOLD_ROOT to the folds directory."
PREP_ROOT="$(find_prep_root)" || die "Cannot uniquely find P3 preprocessing; set P3_PREP_ROOT."

make_common_args() {
  local out_root="$1"
  COMMON_ARGS=(
    --config "$CONFIG" --python "$PYTHON_BIN"
    --train "$ROOT/train_extracted.parquet"
    --validation "$ROOT/validation_extracted.parquet"
    --test "$ROOT/test_extracted.parquet"
    --feature-cols "$ROOT/study_assets/data_schema/feature_columns.json"
    --fold-assignments "$FOLD_ROOT/fold_assignments.parquet"
    --fold-manifest "$FOLD_ROOT/fold_manifest.json"
    --task-file "$TASK_FILE"
    --preprocessing-root "$PREP_ROOT"
    --output-root "$out_root"
    --threshold-config "$THRESHOLD"
    --device cuda
  )
}

check_inputs() {
  local f
  for f in "$CONFIG" "$TASK_FILE" "$THRESHOLD" \
    "$ROOT/train_extracted.parquet" "$ROOT/validation_extracted.parquet" "$ROOT/test_extracted.parquet" \
    "$ROOT/study_assets/data_schema/feature_columns.json" \
    "$FOLD_ROOT/fold_assignments.parquet" "$FOLD_ROOT/fold_manifest.json"; do
    [[ -s "$f" ]] || die "Missing required input: $f"
  done
  for member in 0 1 2; do
    [[ -s "$PREP_ROOT/member_${member}/B/fold_preprocessing.json" ]] || die "Missing preprocessing member $member"
  done
  [[ "$(sha256sum "$TASK_FILE" | awk '{print $1}')" == "$TASK_SHA256" ]] || die "$PROTOCOL_LABEL task protocol hash mismatch"
  echo "[OK] $PROTOCOL_LABEL inputs; 3 members sequential; 4% memory/member (27,778 slots); equal-class buffer; loss=$LOSS_LABEL; replay=12.5%, cap=3."
  echo "[OK] Fold root: $FOLD_ROOT"
  echo "[OK] Preprocessing root: $PREP_ROOT"
}

preflight() {
  local out_root="$1"
  check_inputs
  make_common_args "$out_root"
  "$PYTHON_BIN" src/training/fold_ensemble3_full.py "${COMMON_ARGS[@]}"
}

run_visualizations() {
  local out_root="$1" member task split scope viz_status=0
  local outdir
  for member in 0 1 2; do
    scope="member_${member}"
    for task in 6 7; do
      for split in validation test; do
        outdir="$out_root/diagnostics/$scope/task_${task}_${split}"
        "$PYTHON_BIN" scripts/visualize_cil_run.py \
          --run-root "$out_root" --outdir "$outdir" --task-id "$task" \
          --member-id "$member" --member-ids "$member" --split "$split" \
          --task-file "$TASK_FILE" --train "$ROOT/train_extracted.parquet" \
          --validation "$ROOT/validation_extracted.parquet" --test "$ROOT/test_extracted.parquet" \
          >> "$RUN_HOME/postprocess.log" 2>&1 || viz_status=1
      done
    done
  done
  for task in 6 7; do
    for split in validation test; do
      outdir="$out_root/diagnostics/offline_ensemble/task_${task}_${split}"
      "$PYTHON_BIN" scripts/visualize_cil_run.py \
        --run-root "$out_root" --outdir "$outdir" --task-id "$task" \
        --member-id 0 --member-ids 0 1 2 --ensemble --split "$split" \
        --task-file "$TASK_FILE" --train "$ROOT/train_extracted.parquet" \
        --validation "$ROOT/validation_extracted.parquet" --test "$ROOT/test_extracted.parquet" \
        >> "$RUN_HOME/postprocess.log" 2>&1 || viz_status=1
    done
  done
  return "$viz_status"
}

run_experiment() {
  local run_tag="$1"
  RUN_HOME="${P3_RUN_HOME:-$MONITOR_ROOT/run_$run_tag}"
  OUT_ROOT="${P3_OUT:-$ROOT/outputs/${EXPERIMENT_SLUG}_seed0_$run_tag}"
  mkdir -p "$RUN_HOME" "$OUT_ROOT"
  for provenance in source_snapshot_manifest.json source_commit.txt source_tree.txt \
    source_archive_sha256.txt source_snapshot_files_sha256.txt; do
    if [[ -s "$RUN_HOME/$provenance" ]]; then
      cp "$RUN_HOME/$provenance" "$OUT_ROOT/$provenance"
    fi
  done
  make_common_args "$OUT_ROOT"
  echo "[RUN] Starting three-member full $PROTOCOL_LABEL run ($DISPLAY_LABEL): $OUT_ROOT"
  if "$PYTHON_BIN" src/training/fold_ensemble3_full.py "${COMMON_ARGS[@]}" --execute \
    > "$RUN_HOME/training_and_offline.log" 2>&1; then
    TRAIN_STATUS=0
    echo "[OK] Training and offline OOF/test ensemble completed."
  else
    TRAIN_STATUS=$?
    echo "[WARN] Training/offline pipeline failed (exit=$TRAIN_STATUS); preserving partial artifacts and continuing packaging."
  fi

  REPORT_STATUS=1
  VIZ_STATUS=1
  DIAGNOSTIC_STATUS=1
  if [[ $TRAIN_STATUS -eq 0 && -s "$OUT_ROOT/full_manifest.json" ]]; then
    if "$PYTHON_BIN" src/eval/report.py --full-root "$OUT_ROOT" --task-file "$TASK_FILE" \
      --outdir "$OUT_ROOT/final_results" > "$RUN_HOME/final_report.log" 2>&1; then
      REPORT_STATUS=0
      echo "[OK] Full $PROTOCOL_LABEL final report generated."
    else
      REPORT_STATUS=$?
      echo "[WARN] Final report failed (exit=$REPORT_STATUS)."
    fi
    if run_visualizations "$OUT_ROOT"; then
      VIZ_STATUS=0
      echo "[OK] Task 6/7 member and offline-ensemble visualizations generated for validation and test."
    else
      VIZ_STATUS=$?
      echo "[WARN] One or more visualizations failed (exit=$VIZ_STATUS)."
    fi
    if "$PYTHON_BIN" scripts/analyze_p3_task6_task7.py \
      --full-root "$OUT_ROOT" --task-file "$TASK_FILE" \
      --outdir "$OUT_ROOT/final_results/boundary_task6_task7" \
      > "$RUN_HOME/task6_task7_analysis.log" 2>&1; then
      DIAGNOSTIC_STATUS=0
      echo "[OK] Aligned task-6/task-7 class and cross-group diagnostics generated."
    else
      DIAGNOSTIC_STATUS=$?
      echo "[WARN] Task 6/7 diagnostic failed (exit=$DIAGNOSTIC_STATUS)."
    fi
  else
    echo "[WARN] Offline report/visualization skipped because the verified full manifest is absent."
  fi

  STATUS_FILE="$OUT_ROOT/experiment_status.txt"
  printf 'train_and_offline_exit=%s\nfinal_report_exit=%s\nvisualization_exit=%s\ntask6_task7_analysis_exit=%s\n' \
    "$TRAIN_STATUS" "$REPORT_STATUS" "$VIZ_STATUS" "$DIAGNOSTIC_STATUS" > "$STATUS_FILE"
  ARCHIVE="$OUT_ROOT/review/${EXPERIMENT_SLUG}_review_${run_tag}.zip"
  if "$PYTHON_BIN" scripts/package_p3_experiment_review.py \
    --label "$DISPLAY_LABEL" \
    --run-root "$OUT_ROOT" --config "$CONFIG" --status \
    "train_and_offline_exit=$TRAIN_STATUS; final_report_exit=$REPORT_STATUS; visualization_exit=$VIZ_STATUS; task6_task7_analysis_exit=$DIAGNOSTIC_STATUS" \
    --log "$RUN_HOME/training_and_offline.log" --archive "$ARCHIVE" \
    > "$RUN_HOME/package.log" 2>&1; then
    echo "[OK] Review ZIP: $ARCHIVE"
    echo "[OK] SHA256: $ARCHIVE.sha256"
  else
    echo "[WARN] Review packaging failed; see $RUN_HOME/package.log"
  fi
  printf '%s\n' "$OUT_ROOT" > "$RUN_HOME/output_root.txt"
  if [[ $TRAIN_STATUS -ne 0 ]]; then return "$TRAIN_STATUS"; fi
  return 0
}

launch() {
  local run_home="$1" run_tag="$2" out_root="$3" execution_root="$4" commit
  local execution_config execution_task execution_threshold
  commit="$(<"$run_home/source_commit.txt")"
  execution_config="$(cil_snapshot_path "$ROOT" "$execution_root" "$CONFIG")"
  execution_task="$(cil_snapshot_path "$ROOT" "$execution_root" "$TASK_FILE")"
  execution_threshold="$(cil_snapshot_path "$ROOT" "$execution_root" "$THRESHOLD")"
  nohup env CUDA_VISIBLE_DEVICES="$GPU_ID" PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    P3_PYTHON="$PYTHON_BIN" P3_GPU_ID="$GPU_ID" \
    P3_EXPERIMENT_SLUG="$EXPERIMENT_SLUG" P3_DISPLAY_LABEL="$DISPLAY_LABEL" \
    P3_LOSS_LABEL="$LOSS_LABEL" P3_CONFIG="$execution_config" \
    P3_PROTOCOL_LABEL="$PROTOCOL_LABEL" P3_TASK_FILE="$execution_task" \
    P3_TASK_SHA256="$TASK_SHA256" P3_THRESHOLD_CONFIG="$execution_threshold" \
    P3_FOLD_ROOT="$FOLD_ROOT" P3_PREP_ROOT="$PREP_ROOT" \
    P3_MONITOR_ROOT="$MONITOR_ROOT" P3_RUN_HOME="$run_home" \
    P3_OUT="$out_root" P3_RUN_TAG="$run_tag" \
    CIL_SOURCE_COMMIT="$commit" CIL_SOURCE_SNAPSHOT="$execution_root" \
    bash "$execution_root/scripts/run_p3_hybrid_equal_buffer_full.sh" run "$run_tag" \
    > "$run_home/nohup.log" 2>&1 < /dev/null &
  printf '%s\n' "$!" > "$run_home/job.pid"
  echo "[STARTED] $DISPLAY_LABEL; full 8 tasks, 3 members; PID=$(<"$run_home/job.pid")"
  echo "[LOG] $run_home/nohup.log"
  echo "[OUTPUT] $out_root"
}

case "$ACTION" in
  check)
    check_inputs
    TAG="${2:-$(date -u +%Y%m%dT%H%M%SZ)}"
    OUT_ROOT="${P3_OUT:-$ROOT/outputs/${EXPERIMENT_SLUG}_seed0_$TAG}"
    preflight "$OUT_ROOT"
    echo "[OK] Dry-run/preflight complete; no model created."
    ;;
  run)
    TAG="${2:-${P3_RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}}"
    check_inputs
    run_experiment "$TAG"
    ;;
  start)
    TAG="${2:-$(date -u +%Y%m%dT%H%M%SZ)}"
    RUN_HOME="$MONITOR_ROOT/run_$TAG"
    OUT_ROOT="$ROOT/outputs/${EXPERIMENT_SLUG}_seed0_$TAG"
    [[ ! -e "$RUN_HOME" && ! -e "$OUT_ROOT" ]] || die "Run ID already exists: $TAG"
    cil_assert_snapshot_source_clean "$ROOT" || exit $?
    mkdir -p "$RUN_HOME"
    preflight "$OUT_ROOT" > "$RUN_HOME/preflight.log" 2>&1 || {
      tail -n 30 "$RUN_HOME/preflight.log" >&2
      die "Preflight failed; no training started."
    }
    SNAPSHOT_ROOT="$(cil_create_source_snapshot "$ROOT" "$RUN_HOME")" || exit $?
    printf '%s\n' "$OUT_ROOT" > "$RUN_HOME/output_root.txt"
    printf '%s\n' "$RUN_HOME" > "$LATEST"
    launch "$RUN_HOME" "$TAG" "$OUT_ROOT" "$SNAPSHOT_ROOT"
    ;;
  resume)
    RUN_HOME="$(latest_home)"
    OUT_ROOT="$(<"$RUN_HOME/output_root.txt")"
    TAG="${RUN_HOME##*/run_}"
    if [[ -s "$RUN_HOME/job.pid" ]] && kill -0 "$(<"$RUN_HOME/job.pid")" 2>/dev/null; then
      die "Run is already live: PID=$(<"$RUN_HOME/job.pid")"
    fi
    if [[ -s "$OUT_ROOT/full_manifest.json" ]] \
      && grep -q '"final_status": "completed"' "$OUT_ROOT/full_manifest.json" \
      && compgen -G "$OUT_ROOT/review/${EXPERIMENT_SLUG}_review_*.zip" > /dev/null; then
      echo "[OK] Run is already completed: $OUT_ROOT"
      exit 0
    fi
    SNAPSHOT_ROOT="$(cil_load_source_snapshot "$RUN_HOME")" || exit $?
    launch "$RUN_HOME" "$TAG" "$OUT_ROOT" "$SNAPSHOT_ROOT"
    ;;
  status|follow)
    RUN_HOME="$(latest_home)"
    [[ -s "$RUN_HOME/job.pid" ]] || die "PID file missing: $RUN_HOME/job.pid"
    PID="$(<"$RUN_HOME/job.pid")"
    if [[ "$ACTION" == "follow" ]]; then
      tail -f "$RUN_HOME/nohup.log"
    else
      if kill -0 "$PID" 2>/dev/null; then echo "[RUNNING] PID=$PID"; else echo "[STOPPED/FINISHED] PID=$PID"; fi
      [[ -s "$RUN_HOME/output_root.txt" ]] && echo "[OUTPUT] $(<"$RUN_HOME/output_root.txt")"
      [[ -s "$RUN_HOME/nohup.log" ]] && tail -n 30 "$RUN_HOME/nohup.log"
      [[ -s "$RUN_HOME/package.log" ]] && tail -n 5 "$RUN_HOME/package.log"
    fi
    ;;
  *)
    echo "Usage: bash scripts/run_p3_hybrid_equal_buffer_full.sh [start|run|check|resume|status|follow] [run_tag]" >&2
    exit 2
    ;;
esac
