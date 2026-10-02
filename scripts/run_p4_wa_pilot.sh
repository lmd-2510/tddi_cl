#!/usr/bin/env bash
set -Eeuo pipefail

# P4 / pure Weight Alignment / member-0 pilot.  No replay or exemplar buffer
# is used: this isolates the classifier-row norm correction from EWC.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG="$ROOT/configs/p4_wa_pilot_member0.json"
OUT_BASE="$ROOT/outputs/p4_wa_pilot_member0"
MONITOR_BASE="$ROOT/outputs/p4_experiments/wa_pilot_member0_monitor"
TASK_FILE="$ROOT/study_assets/task_protocols/constrained_mass_balanced_seed0_tasks.json"
FEATURE_COLS="$ROOT/study_assets/data_schema/feature_columns.json"
TRAIN="$ROOT/train_extracted.parquet"
VALIDATION="$ROOT/validation_extracted.parquet"
TEST="$ROOT/test_extracted.parquet"

die() { echo "[STOP] $*" >&2; exit 1; }

find_prep() {
  if [[ -n "${P4_PREP_ROOT:-}" && -s "$P4_PREP_ROOT/member_0/B/fold_preprocessing.json" ]]; then
    printf '%s\n' "$P4_PREP_ROOT/member_0/B/fold_preprocessing.json"; return 0
  fi
  local preferred="$ROOT/study_assets/preprocessing_p4_seed0_fold42/member_0/B/fold_preprocessing.json"
  if [[ -s "$preferred" ]]; then printf '%s\n' "$preferred"; return 0; fi
  mapfile -t found < <(find "$ROOT/study_assets" "$ROOT/outputs" -type f \
    -path '*preprocessing_p4*/*/B/fold_preprocessing.json' 2>/dev/null | sort -u)
  (( ${#found[@]} == 1 )) || die "Set P4_PREP_ROOT to the current P4 preprocessing root (found ${#found[@]} candidates)."
  printf '%s\n' "${found[0]}"
}

validate_inputs() {
  [[ -s "$TASK_FILE" && -s "$FEATURE_COLS" ]] || die "Missing P4 task/schema input."
  [[ -s "$TRAIN" && -s "$VALIDATION" && -s "$TEST" ]] || die "Missing extracted parquet input(s)."
  local task_sha; task_sha="$(sha256sum "$TASK_FILE" | awk '{print $1}')"
  [[ "$task_sha" == "9d22af8618fe03c40b92e2aabbab8b68a89de83bc0de4f51e24932951cf298f4" ]] || die "P4 task-file SHA256 mismatch: $task_sha"
  echo "[OK] P4 inputs validated."
}

new_run_id() { date -u +%Y%m%dT%H%M%SZ; }

package_review() {
  local out="$1" monitor="$2"
  local review="$out/p4_wa_pilot_member0_${out##*/}_review.zip"
  cp "$monitor/training.log" "$out/training.log" 2>/dev/null || true
  (cd "$out" && zip -qr "$review" . -x '*.npz' '*.npy' 'task0_frozen_scaler.pkl' "$(basename "$review")")
  sha256sum "$review" > "$review.sha256"
  echo "$review" > "$out/review_package.txt"
}

run_training() {
  validate_inputs
  local run_id out monitor prep scaler
  run_id="${RUN_ID:-$(new_run_id)}"
  # Keep one timestamp for both output and monitor paths.
  out="$OUT_BASE/$run_id"
  monitor="$MONITOR_BASE/run_$run_id"
  prep="$(find_prep)"
  mkdir -p "$out" "$monitor"
  scaler="$out/task0_frozen_scaler.pkl"
  "$PYTHON_BIN" scripts/convert_fold_preprocessing_to_scaler.py --input "$prep" --output "$scaler" | tee "$monitor/preprocessing.log"
  cp "$CONFIG" "$out/run_config.json"
  printf '%s\n' "$prep" > "$out/preprocessing_source.txt"
  set +e
  {
    echo "[RUN] P4 pure WA pilot; member=0; CE; policy=new_class_mean_norm_v1; replay=none"
    "$PYTHON_BIN" src/training/train_cil.py \
      --train "$TRAIN" --validation "$VALIDATION" --test "$TEST" \
      --feature-cols "$FEATURE_COLS" --scaler "$scaler" --task-file "$TASK_FILE" \
      --outdir "$out/member_0" --method wa --variant tddi_paper_member \
      --batch-size 64 --effective-batch-size 1024 --epochs 30 --patience 5 \
      --lr 0.001 --weight-decay 0.0001 --dropout 0.2 --activation gelu --norm layernorm \
      --ewc-classification-loss ce --weight-alignment new_class_mean_norm_v1 \
      --seed 0 --member-id 0 --ensemble-mode stratified_3fold --fold-id 0 \
      --fold-count 3 --fold-seed 42 --device cuda
  } 2>&1 | tee "$monitor/training.log"
  local status_code=${PIPESTATUS[0]}
  set -e
  printf '%s\n' "$status_code" > "$monitor/exit_code.txt"
  (( status_code == 0 )) || die "WA pilot failed; see $monitor/training.log"
  package_review "$out" "$monitor"
  echo "[DONE] P4 WA pilot completed: $out"
}

status() {
  local latest
  latest="$(find "$MONITOR_BASE" -mindepth 1 -maxdepth 1 -type d -name 'run_*' -printf '%T@ %p\n' 2>/dev/null | sort -nr | awk 'NR==1{sub(/^[^ ]+ /,""); print}')"
  [[ -n "$latest" ]] || { echo "[NONE] No P4 WA pilot run found."; return 0; }
  echo "[RUN] $latest"
  [[ -s "$latest/exit_code.txt" ]] && echo "exit_code=$(cat "$latest/exit_code.txt")" || echo "still running or not started"
  pgrep -af "src/training/train_cil.py.*--method wa" || true
}

start() {
  mkdir -p "$MONITOR_BASE"
  local run_id monitor
  run_id="$(new_run_id)"; monitor="$MONITOR_BASE/run_$run_id"
  mkdir -p "$monitor"
  echo "$run_id" > "$monitor/run_id.txt"
  nohup bash "$0" run --run-id "$run_id" > "$monitor/nohup.log" 2>&1 &
  echo $! > "$monitor/job.pid"
  echo "[STARTED] PID=$!"
  echo "[LOG] $monitor/nohup.log"
}

cmd="${1:-status}"
case "$cmd" in
  check) validate_inputs; prep_path="$(find_prep)"; echo "[OK] P4 preprocessing: $prep_path"; echo "[OK] WA pilot inputs ready." ;;
  start) start ;;
  run) shift; [[ "${1:-}" == "--run-id" ]] && { RUN_ID="$2"; } || true; run_training ;;
  status) status ;;
  follow) latest="$(find "$MONITOR_BASE" -name nohup.log -type f -printf '%T@ %p\n' 2>/dev/null | sort -nr | awk 'NR==1{sub(/^[^ ]+ /,""); print}')"; [[ -n "$latest" ]] && tail -f "$latest" || die "No nohup log found." ;;
  *) echo "Usage: $0 {check|start|run|status|follow}"; exit 2 ;;
esac
