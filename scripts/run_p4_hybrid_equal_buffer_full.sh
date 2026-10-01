#!/usr/bin/env bash
# One-command P4 full run using the selected best-historical-derived recipe:
# Hybrid loss + equal-class buffer, three sequential members, offline ensemble,
# task-6/task-7 diagnostics, visualizations, and compact review ZIP.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 2
source "$ROOT/scripts/lib/source_snapshot.sh"
ACTION="${1:-start}"
PYTHON_BIN="${P4_HYBRID_PYTHON:-python}"
GPU_ID="${P4_HYBRID_GPU_ID:-0}"
CONFIG="$ROOT/configs/p4_hybrid_equal_buffer_full8_e30_mem4.json"
TASK_FILE="$ROOT/study_assets/task_protocols/constrained_mass_balanced_seed0_tasks.json"
TASK_SHA256="9d22af8618fe03c40b92e2aabbab8b68a89de83bc0de4f51e24932951cf298f4"
EXPERIMENT_SLUG="p4_hybrid_equal_buffer_full8_e30_mem4"
DISPLAY_LABEL="P4 Hybrid + equal-class buffer full8 e30 mem4"
MONITOR_ROOT="${P4_HYBRID_MONITOR_ROOT:-$ROOT/outputs/p4_experiments/hybrid_equal_buffer}"
LATEST="$MONITOR_ROOT/latest.txt"
PREP_ROOT="${P4_HYBRID_PREP_ROOT:-$ROOT/study_assets/preprocessing_p4_seed0_fold42}"

die() { echo "[STOP] $*" >&2; exit 2; }

find_fold_root() {
  if [[ -n "${P4_HYBRID_FOLD_ROOT:-}" ]]; then
    printf '%s\n' "$P4_HYBRID_FOLD_ROOT"
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

latest_home() {
  [[ -s "$LATEST" ]] || die "No P4 Hybrid equal-buffer run is registered."
  local value
  value="$(<"$LATEST")"
  [[ -d "$value" && "$value" == "$MONITOR_ROOT"/run_* ]] || die "Invalid latest run pointer: $value"
  printf '%s\n' "$value"
}

FOLD_ROOT="$(find_fold_root)" || die "Cannot resolve one fold root; export P4_HYBRID_FOLD_ROOT=/absolute/path/to/folds."

check_static_inputs() {
  local path
  for path in "$CONFIG" "$TASK_FILE" \
    "$ROOT/train_extracted.parquet" "$ROOT/validation_extracted.parquet" "$ROOT/test_extracted.parquet" \
    "$ROOT/study_assets/data_schema/feature_columns.json" \
    "$FOLD_ROOT/fold_assignments.parquet" "$FOLD_ROOT/fold_manifest.json"; do
    [[ -s "$path" ]] || die "Missing required input: $path"
  done
  [[ "$(sha256sum "$TASK_FILE" | awk '{print $1}')" == "$TASK_SHA256" ]] \
    || die "P4 task protocol hash mismatch."
  echo "[OK] P4 constrained-mass-balanced protocol and frozen inputs validated."
  echo "[OK] Recipe: Hybrid (Focal current + CE replay), equal-class buffer, replay=12.5%, cap=3, 30 epochs, 27,778 slots/member."
}

prepare_preprocessing() {
  local log="$1" member target target_dir
  mkdir -p "$(dirname "$log")"
  for member in 0 1 2; do
    target="$PREP_ROOT/member_${member}/B/fold_preprocessing.json"
    target_dir="$(dirname "$target")"
    if [[ -s "$target" ]]; then
      echo "[SKIP] P4 preprocessing member $member already exists." | tee -a "$log"
      continue
    fi
    if [[ -d "$target_dir" ]] && find "$target_dir" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
      die "Partial preprocessing namespace requires inspection: $target_dir"
    fi
    # prepare_fold_preprocessing.py deliberately rejects an existing outdir.
    # A failed earlier attempt may have left the directory present but empty.
    if [[ -d "$target_dir" ]]; then
      rmdir "$target_dir" || die "Cannot remove empty preprocessing directory: $target_dir"
    fi
    mkdir -p "$(dirname "$target_dir")"
    echo "[RUN] Preparing frozen P4 task-0 scaler for member $member." | tee -a "$log"
    if ! "$PYTHON_BIN" "$ROOT/scripts/prepare_fold_preprocessing.py" \
      --assignments "$FOLD_ROOT/fold_assignments.parquet" \
      --manifest "$FOLD_ROOT/fold_manifest.json" \
      --train "$ROOT/train_extracted.parquet" \
      --validation "$ROOT/validation_extracted.parquet" \
      --test "$ROOT/test_extracted.parquet" \
      --task-file "$TASK_FILE" \
      --feature-cols "$ROOT/study_assets/data_schema/feature_columns.json" \
      --member-id "$member" --validation-fold "$member" \
      --experiment-seed 0 --fold-seed 42 \
      --policy task0_standard_frozen --batch-size 2048 \
      --outdir "$target_dir" >> "$log" 2>&1; then
      tail -n 40 "$log" >&2
      die "P4 preprocessing failed for member $member; see $log"
    fi
    [[ -s "$target" ]] || die "Preprocessing command completed without artifact: $target"
    echo "[OK] P4 preprocessing member $member completed." | tee -a "$log"
  done
}

export_generic_environment() {
  export P3_PYTHON="$PYTHON_BIN"
  export P3_GPU_ID="$GPU_ID"
  export P3_EXPERIMENT_SLUG="$EXPERIMENT_SLUG"
  export P3_DISPLAY_LABEL="$DISPLAY_LABEL"
  export P3_LOSS_LABEL="Hybrid (Focal current + CE replay)"
  export P3_CONFIG="$CONFIG"
  export P3_PROTOCOL_LABEL="P4"
  export P3_TASK_FILE="$TASK_FILE"
  export P3_TASK_SHA256="$TASK_SHA256"
  export P3_FOLD_ROOT="$FOLD_ROOT"
  export P3_PREP_ROOT="$PREP_ROOT"
  export P3_MONITOR_ROOT="$MONITOR_ROOT"
}

launch() {
  local run_home="$1" tag="$2" out_root="$3" execution_root="$4" commit
  commit="$(<"$run_home/source_commit.txt")"
  nohup env CUDA_VISIBLE_DEVICES="$GPU_ID" PYTHONUNBUFFERED=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    P4_HYBRID_FOLD_ROOT="$FOLD_ROOT" P4_HYBRID_PREP_ROOT="$PREP_ROOT" \
    P4_HYBRID_PYTHON="$PYTHON_BIN" P4_HYBRID_MONITOR_ROOT="$MONITOR_ROOT" \
    P4_HYBRID_RUN_HOME="$run_home" P4_HYBRID_OUT_ROOT="$out_root" \
    PYTHONDONTWRITEBYTECODE=1 \
    CIL_SOURCE_COMMIT="$commit" CIL_SOURCE_SNAPSHOT="$execution_root" \
    bash "$execution_root/scripts/run_p4_hybrid_equal_buffer_full.sh" run "$tag" \
    > "$run_home/nohup.log" 2>&1 < /dev/null &
  printf '%s\n' "$!" > "$run_home/job.pid"
  echo "[STARTED] $DISPLAY_LABEL; PID=$(<"$run_home/job.pid")"
  echo "[OUTPUT] $out_root"
  echo "[LOG] $run_home/nohup.log"
}

case "$ACTION" in
  status|follow)
    RUN_HOME="$(latest_home)"
    OUT_ROOT="$(<"$RUN_HOME/output_root.txt")"
    if [[ "$ACTION" == "follow" ]]; then
      touch "$RUN_HOME/nohup.log"
      tail -f "$RUN_HOME/nohup.log"
    else
      PID="$(<"$RUN_HOME/job.pid")"
      if kill -0 "$PID" 2>/dev/null; then
        echo "[RUNNING] PID=$PID"
      elif [[ -s "$RUN_HOME/exit_code.txt" ]]; then
        echo "[FINISHED] exit_code=$(<"$RUN_HOME/exit_code.txt")"
      else
        echo "[STOPPED/UNKNOWN] PID=$PID"
      fi
      echo "[OUTPUT] $OUT_ROOT"
      [[ -s "$RUN_HOME/nohup.log" ]] && tail -n 30 "$RUN_HOME/nohup.log"
      [[ -s "$RUN_HOME/package.log" ]] && tail -n 5 "$RUN_HOME/package.log"
    fi
    exit 0
    ;;
esac

TAG="${2:-${P4_HYBRID_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}}"
RUN_HOME="${P4_HYBRID_RUN_HOME:-$MONITOR_ROOT/run_$TAG}"
OUT_ROOT="${P4_HYBRID_OUT_ROOT:-$ROOT/outputs/${EXPERIMENT_SLUG}_seed0_$TAG}"

case "$ACTION" in
  check)
    check_static_inputs
    mkdir -p "$RUN_HOME"
    prepare_preprocessing "$RUN_HOME/preprocessing.log"
    export_generic_environment
    P3_OUT="$OUT_ROOT" P3_RUN_HOME="$RUN_HOME" \
      bash "$ROOT/scripts/run_p3_hybrid_equal_buffer_full.sh" check "$TAG"
    ;;
  run)
    mkdir -p "$RUN_HOME" "$OUT_ROOT"
    printf '%s\n' "$OUT_ROOT" > "$RUN_HOME/output_root.txt"
    check_static_inputs
    status=0
    prepare_preprocessing "$RUN_HOME/preprocessing.log" || status=$?
    if [[ $status -eq 0 ]]; then
      export_generic_environment
      export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU_ID}"
      P3_OUT="$OUT_ROOT" P3_RUN_HOME="$RUN_HOME" \
        bash "$ROOT/scripts/run_p3_hybrid_equal_buffer_full.sh" run "$TAG" || status=$?
    fi
    printf '%s\n' "$status" > "$RUN_HOME/exit_code.txt"
    exit "$status"
    ;;
  start)
    [[ ! -e "$RUN_HOME" && ! -e "$OUT_ROOT" ]] || die "Run ID already exists: $TAG"
    check_static_inputs
    cil_assert_snapshot_source_clean "$ROOT" || exit $?
    mkdir -p "$RUN_HOME"
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
  *)
    die "Usage: bash scripts/run_p4_hybrid_equal_buffer_full.sh [check|start|run|resume|status|follow]"
    ;;
esac
