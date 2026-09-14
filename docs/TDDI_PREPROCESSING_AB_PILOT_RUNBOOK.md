# Runbook GPU — preprocessing A/B, member 0, P3 task 0–1

Prompt 12, ngày 2026-09-14. **Chỉ chạy các lệnh training trên server GPU riêng.**
Lượt triển khai code không chạy dữ liệu thật. Chưa chọn preprocessing/ranking cuối.

Mục tiêu: so A `raw_identity` với B `task0_standard_frozen`, cùng member 0/seed 0,
assignment fold seed 42, cùng current/exemplar IDs và sampler order ở epoch chung.
Chỉ validation được dùng đánh giá hai phương án; không chạy test metrics, ba member,
ensemble UE, threshold hoặc full8 để quyết định preprocessing.

## 1. Bản đồ file và thứ tự thực hiện

| File | Vai trò |
| --- | --- |
| `scripts/prepare_fold_preprocessing.py` | Tạo artifact A hoặc fit B chỉ từ training task 0 của member 0, đóng băng. |
| `src/training/fold_ab_study.py` | Dry-run, điều phối tuần tự, kiểm tra/skip/resume hợp lệ. |
| `src/training/fold_pilot_training.py` | Trainer validation-only và checkpoint task boundary. |
| `configs/smoke_tddi_p3_fold_ab_A_raw_seed0.json` / `...B_task0_scaler_seed0.json` | Smoke 3 epoch/task, chỉ task 0–1. |
| `configs/pilot_tddi_p3_fold_ab_A_raw_seed0.json` / `...B_task0_scaler_seed0.json` | Pilot tối đa 20 epoch/task, patience 5, chỉ task 0–1. |
| `scripts/compare_preprocessing_pilots.py` | So audit/validation, xuất JSON/Markdown và optional review bundle; không import trainer/PyTorch hay đọc dataset. |

Trình tự: kiểm tra máy/data → chuẩn bị A/B → dry-run → smoke A rồi B → kiểm tra
kỹ thuật → pilot A rồi B → báo cáo validation → gửi review → **dừng chờ Prompt 13**.

## 2. Vào đúng repo, kiểm tra môi trường

```bash
conda activate ai_env
cd "$HOME/DrugDrug/cil-tddi/cil-tddi"
set -o pipefail
export REPO_ROOT="$PWD"
git status --short
git rev-parse HEAD
which python
python --version
python -m pip check
command -v nohup
command -v nvidia-smi
command -v /usr/bin/time
free -h
df -h .
nvidia-smi
```

Nếu cần cập nhật code từ GitHub: kiểm tra working tree trước rồi `git pull --ff-only`.
**Không reset hoặc xóa file local** để ép pull. Máy code phải commit/push bản mới
trước thì server mới pull được; tài liệu này không có nghĩa code đã được push.
Không update code/requirements giữa A và B hoặc giữa một run đang resume.

Nếu môi trường thiếu dependencies, cài trước khi bắt đầu và dùng cùng môi trường
cho cả A/B:

```bash
python -m pip install -r requirements.txt
python -m pip install pytest
python -m pip check
CUDA_VISIBLE_DEVICES=0 python - <<'PY'
import torch
print('torch:', torch.__version__, 'CUDA build:', torch.version.cuda)
assert torch.cuda.is_available(), 'CUDA unavailable'
free, total = torch.cuda.mem_get_info(0)
print('GPU:', torch.cuda.get_device_name(0))
print('free/total GiB:', free / 2**30, total / 2**30)
PY
```

Điều kiện trước smoke:

- GPU 0 được phép sử dụng, không có job của mình đang train trùng namespace.
  Không dừng process của người khác để lấy GPU.
- Card 24 GB đã dùng trước đây là ứng viên phù hợp để thử, **không bảo đảm không OOM**.
  Ưu tiên GPU còn phần lớn VRAM trống; peak thật lấy từ smoke.
- Với máy RAM 62 GiB trước đây, kiểm tra `available`, không chỉ cột `free`.
  Cần dư RAM cho source arrays, model/teacher/optimizer và buffer; theo dõi swap.
- Dự trù tối thiểu khoảng 20 GiB disk trống cho output hai scope A/B, ngoài dataset;
  đây là dự phòng vận hành, không phải ngưỡng bảo đảm. Dùng checkpoint size smoke
  để ước lượng lại; interrupted attempts và log sẽ tăng dung lượng.
- `pip check` và các test liên quan pass. Không lấy việc test nhanh là bằng chứng
  đã train được model thật hoặc có đủ VRAM.

## 3. Khai báo path — sửa đúng nơi đang lưu data/fold

Không tạo lại folds. Giữ `fold_assignments.parquet` và `fold_manifest.json` đã audit.
Các path sau là ví dụ, **đổi `FOLD_ROOT` theo vị trí thật trên server**:

```bash
export TRAIN="$REPO_ROOT/train_extracted.parquet"
export VALIDATION="$REPO_ROOT/validation_extracted.parquet"
export TEST="$REPO_ROOT/test_extracted.parquet"
export FEATURES="$REPO_ROOT/study_assets/data_schema/feature_columns.json"
export TASK_FILE="$REPO_ROOT/study_assets/task_protocols/tail_to_head_tasks.json"
export FOLD_ROOT="$REPO_ROOT/study_assets/stratified_3fold_seed42"
export ASSIGNMENTS="$FOLD_ROOT/fold_assignments.parquet"
export FOLD_MANIFEST="$FOLD_ROOT/fold_manifest.json"
export PREP_ROOT="$REPO_ROOT/study_assets/preprocessing_ab_seed0_fold42"
export AB_ROOT="$REPO_ROOT/outputs/preprocessing_ab_v1"
export MONITOR_ROOT="$REPO_ROOT/outputs/preprocessing_ab_monitor_v1"
mkdir -p "$MONITOR_ROOT"

for file in "$TRAIN" "$VALIDATION" "$TEST" "$FEATURES" "$TASK_FILE" "$ASSIGNMENTS" "$FOLD_MANIFEST"; do
  if [ ! -s "$file" ]; then echo "[MISSING] $file"; else echo "[OK] $file"; fi
done
```

Nếu có `[MISSING]`, **dừng**, tìm lại path; không chạy tiếp hoặc dùng nhầm smoke task
file. Không `mkdir` trước các `.../A/member_0`, `.../B/member_0`, preprocessing A/B
outdir hoặc comparison outdir: công cụ cần tự tạo namespace mới để chống ghi đè.
Chỉ thư mục monitor/log được tạo trước.

### Ghi revision, thông số máy và kiểm tra hashes

```bash
export PREFLIGHT_DIR="$(mktemp -d "$MONITOR_ROOT/preflight.XXXXXX")"
git rev-parse HEAD > "$PREFLIGHT_DIR/git_commit.txt"
git status --short > "$PREFLIGHT_DIR/git_status.txt"
python -m pip freeze > "$PREFLIGHT_DIR/requirements_frozen.txt"
nvidia-smi > "$PREFLIGHT_DIR/nvidia_smi.txt"
free -h > "$PREFLIGHT_DIR/ram.txt"
df -h . > "$PREFLIGHT_DIR/disk.txt"
python - <<'PY' | tee "$PREFLIGHT_DIR/protocol_fold_check.txt"
import os, json, hashlib
from pathlib import Path
def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()
task_path = Path(os.environ['TASK_FILE'])
spec = json.loads(task_path.read_text())
tasks = spec['tasks']
assert spec['protocol'] == 'tail_to_head'
assert [t['task_id'] for t in tasks] == list(range(8))
assert [len(t['classes']) for t in tasks] == [38,20,20,20,20,20,20,20]
assert len({c for t in tasks for c in t['classes']}) == 178
assert sha(task_path) == '0d64c465b0c4bd34f66e6c76088b6b73fd60839ade3e56017b5fd36c21a26e79'
manifest = json.loads(Path(os.environ['FOLD_MANIFEST']).read_text())
assert sha(os.environ['ASSIGNMENTS']) == manifest['assignment_sha256']
assert manifest['assignment_sha256'] == 'c84dc034431ed27907cdfd7217fa3460e0a9480bbf8343e5c91e68927de79a43'
assert manifest['assignment_row_count'] == 694455 and manifest['fold_seed'] == 42
print('P3 full8 /178 classes: PASS; execution will stop at task1')
print('task SHA256:', sha(task_path))
print('assignment SHA256:', manifest['assignment_sha256'])
print('global budget=27778; members=9260/9259/9259')
PY
```

Dùng `set -o pipefail` trong shell để pipeline `python | tee` không che lỗi Python.
Nếu assert fail, dừng kiểm tra revision/bytes/line endings/assignment gốc; không sửa
hash tùy tiện để bỏ qua integrity. Bước chuẩn bị dưới và dry-run sẽ kiểm tra cả
source hashes, source rows/labels/IDs. Chỉ kiểm tra file có tồn tại là chưa đủ.

## 4. Unit tests và chuẩn bị preprocessing A/B

```bash
python -m pytest tests/test_fold_ab_study.py \
  tests/test_compare_preprocessing_pilots.py \
  tests/test_fold_pilot_training.py tests/test_fold_replay_checkpoint.py -q
python scripts/prepare_fold_preprocessing.py --help
python src/training/fold_ab_study.py --help
python scripts/compare_preprocessing_pilots.py --help
```

Chạy tuần tự hai lệnh chuẩn bị, không cần GPU. B chỉ fit trên training task 0;
không fit toàn bộ hai folds và không fit validation/test:

```bash
python scripts/prepare_fold_preprocessing.py \
  --assignments "$ASSIGNMENTS" --manifest "$FOLD_MANIFEST" \
  --train "$TRAIN" --validation "$VALIDATION" --test "$TEST" \
  --task-file "$TASK_FILE" --feature-cols "$FEATURES" \
  --member-id 0 --validation-fold 0 --experiment-seed 0 --fold-seed 42 \
  --policy raw_identity --batch-size 2048 \
  --outdir "$PREP_ROOT/member_0/A"

python scripts/prepare_fold_preprocessing.py \
  --assignments "$ASSIGNMENTS" --manifest "$FOLD_MANIFEST" \
  --train "$TRAIN" --validation "$VALIDATION" --test "$TEST" \
  --task-file "$TASK_FILE" --feature-cols "$FEATURES" \
  --member-id 0 --validation-fold 0 --experiment-seed 0 --fold-seed 42 \
  --policy task0_standard_frozen --batch-size 2048 \
  --outdir "$PREP_ROOT/member_0/B"
```

Nếu đã có đúng artifacts, bỏ hai lệnh tạo lại; dry-run sẽ load/validate, không refit.
Giữ toàn bộ thư mục preprocessing. Namespace có file dở thì kiểm tra log và chọn
namespace mới nếu cần, không xóa/ghi đè. Không dùng scaler legacy đã fit toàn train.

## 5. Dry-run smoke A/B

Trong **cùng phiên Bash**, tạo array dùng chung:

```bash
COMMON=(
  --train "$TRAIN" --validation "$VALIDATION" --test "$TEST"
  --feature-cols "$FEATURES" --task-file "$TASK_FILE"
  --fold-assignments "$ASSIGNMENTS" --fold-manifest "$FOLD_MANIFEST"
  --preprocessing-root "$PREP_ROOT" --output-root "$AB_ROOT"
  --member-id 0 --device cuda
)
python src/training/fold_ab_study.py \
  --config configs/smoke_tddi_p3_fold_ab_A_raw_seed0.json \
           configs/smoke_tddi_p3_fold_ab_B_task0_scaler_seed0.json \
  "${COMMON[@]}" | tee "$PREFLIGHT_DIR/smoke_dry_run.txt"
```

Mong đợi `fresh` cho lần đầu, budget 9260, chỉ member 0, task 0–1. `UNVERIFIED`
nghĩa còn thiếu inputs, chưa được chạy thật. Sau lần chạy trước có thể thấy
`complete`/`resume` khi checkpoint hợp lệ. Đọc dữ liệu và hash có thể mất vài phút,
không có log epoch ngay không đồng nghĩa bị treo.

## 6. Hàm chạy nohup trên CUDA 0 và ghi telemetry

Copy hàm này vào Bash trong repo. Nó chỉ chạy khi bạn gọi `launch_ab ...`.
Log nằm ngoài member outdir, tránh conflict guard. GPU monitor là một process
riêng, không phải member train thứ hai.

```bash
launch_ab() {
  local label="$1"
  shift
  local log_dir
  log_dir="$(mktemp -d "$MONITOR_ROOT/${label}.XXXXXX")" || return 1
  nohup env CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    bash -c '
      set -uo pipefail
      log_dir="$1"
      shift
      nvidia-smi --id=0 \
        --query-gpu=timestamp,index,memory.used,memory.free,utilization.gpu,power.draw \
        --format=csv -l 2 > "$log_dir/gpu.csv" 2> "$log_dir/gpu_monitor.err" &
      monitor_pid=$!
      trap "kill ${monitor_pid} 2>/dev/null || true; wait ${monitor_pid} 2>/dev/null || true" EXIT
      /usr/bin/time -v python src/training/fold_ab_study.py "$@" --execute
      status=$?
      printf "%s\n" "$status" > "$log_dir/exit_code.txt"
      exit "$status"
    ' ab-job "$log_dir" --config "$@" "${COMMON[@]}" \
    > "$log_dir/nohup.log" 2>&1 < /dev/null &
  local job_pid=$!
  printf "%s\n" "$job_pid" > "$log_dir/job.pid"
  echo "PID=$job_pid LOG_DIR=$log_dir"
}

launch_ab smoke \
  configs/smoke_tddi_p3_fold_ab_A_raw_seed0.json \
  configs/smoke_tddi_p3_fold_ab_B_task0_scaler_seed0.json
```

Smoke chạy **A rồi B**, mỗi case 3 epoch/task, dừng sau task 1. Không đồng thời
khởi chạy thêm một `launch_ab` khác. `nohup` cho phép tắt laptop/ngắt SSH; server
phải còn chạy. Nó không tự khôi phục sau server reboot/OOM.

## 7. Theo dõi và acceptance sau smoke

Thay path `LOG_DIR` bằng dòng in từ hàm:

```bash
LOG_DIR='/absolute/path/printed/by/launch_ab'
ps -p "$(cat "$LOG_DIR/job.pid")" -o pid,ppid,etime,stat,%cpu,%mem,args
pgrep -afu "$USER" 'fold_ab_study.py|src/training/train_cil.py'
tail -n 80 "$LOG_DIR/nohup.log"
tail -f "$AB_ROOT/smoke/A/member_0/stdout.log"
```

`Ctrl+C` khi `tail -f` chỉ thoát xem log, không dừng job. Sau A, đọc
`$AB_ROOT/smoke/B/member_0/stdout.log`. Dùng `nvidia-smi` xem GPU; khi còn đang load
CPU/data, GPU utilization có thể thấp. `memory.used` của nvidia-smi **không phải**
PyTorch peak allocated/reserved của task; báo cáo giữ hai loại riêng.

Sau job:

```bash
cat "$LOG_DIR/exit_code.txt"
find "$AB_ROOT/smoke" -path '*/checkpoints/task_*.pt' -type f -printf '%s bytes %p\n'
python src/training/fold_ab_study.py \
  --config configs/smoke_tddi_p3_fold_ab_A_raw_seed0.json \
           configs/smoke_tddi_p3_fold_ab_B_task0_scaler_seed0.json \
  "${COMMON[@]}"
export SMOKE_REPORT="$AB_ROOT/reviews/smoke_$(date -u +%Y%m%dT%H%M%SZ)"
python scripts/compare_preprocessing_pilots.py \
  --run-a "$AB_ROOT/smoke/A/member_0" --run-b "$AB_ROOT/smoke/B/member_0" \
  --task-file "$TASK_FILE" --outdir "$SMOKE_REPORT" --bundle
```

**Go kỹ thuật** khi exit 0, dry-run xác nhận cả A/B `complete` đúng scope, report
alignment PASS; head task0/1 là 38/58, không có task2; losses/validation finite;
current đầy đủ, repeat cap ≤3, replay task0=0; có buffer/task checkpoint và VRAM
không OOM. Task1 phải có distillation hoạt động theo audit (loss có thể rất nhỏ,
không bắt buộc từng batch dương). Chưa kết luận A/B thắng theo smoke 3 epoch.

Thiếu telemetry: report ghi `unavailable`; không suy thành 0 MB/0 giây. Xem thêm
`gpu_monitor.err` và `/usr/bin/time -v` trong nohup log. Telemetry thiếu chưa đủ để
khẳng định máy dư VRAM. Nếu có mismatch/nonfinite/OOM thì **no-go**, chưa chạy pilot.

## 8. Pilot A rồi B — chỉ sau smoke đạt go

Giữ nguyên mọi dữ liệu/assignment/preprocessing/hyper ngoài max epochs từ smoke sang
pilot. Namespace pilot tự tách smoke; **không resume checkpoint smoke để thành pilot**.

```bash
python src/training/fold_ab_study.py \
  --config configs/pilot_tddi_p3_fold_ab_A_raw_seed0.json \
           configs/pilot_tddi_p3_fold_ab_B_task0_scaler_seed0.json \
  "${COMMON[@]}" | tee "$PREFLIGHT_DIR/pilot_dry_run.txt"

launch_ab pilot_A configs/pilot_tddi_p3_fold_ab_A_raw_seed0.json
```

Đợi A kết thúc, kiểm tra exit code/log và dry-run một config A thấy `complete`.
**Sau đó** mới gọi:

```bash
launch_ab pilot_B configs/pilot_tddi_p3_fold_ab_B_task0_scaler_seed0.json
```

Có thể chủ động đưa cả hai config vào **một** `launch_ab pilot_AB ...A... ...B...`
để chạy tuần tự tự động. Không gọi cách đó đồng thời với hai lệnh riêng ở trên.
Đây vẫn chỉ member 0/task0–1; max20 epoch/patience5, không phải full8.

## 9. Resume và OOM

### Interruption nhưng không đổi config/code

Nếu đăng nhập SSH lại, khai báo lại các biến path ở mục 2–3, array COMMON ở mục 5
và hàm launch_ab ở mục 6. Giữ đúng AB_ROOT cũ; không chạy lại lệnh tạo preprocessing.

Trước hết kiểm tra PID và command để chắc job cũ đã dừng. Chạy lại chính command
`launch_ab` cùng config/COMMON: task0 checkpoint hợp lệ → resume task1; task1 hợp
lệ → skip. Dữ liệu được hash/validate lại. Không chỉ dựa vào tên `.pt` hoặc PID cũ
(PID có thể được tái sử dụng). Không dùng `kill -9` mặc định.

Checkpoint chỉ ở task boundary: task đang dở có thể phải học lại cả task. Artifact
attempt dở được giữ; attempt tiếp theo nằm trong `attempts/`. Sau resume, summary
root có thể cũ; dùng manifest mới hoặc task artifacts, không đọc root summary rồi
kết luận sai rằng task1 chưa xong. Comparator tự tìm completed task directory;
nếu có nhiều bản completed mâu thuẫn thì fail, cần kiểm tra chứ không đoán bản mới.

Không có boundary checkpoint hợp lệ: giữ output/log dở để điều tra, tạo một
`AB_ROOT` mới nếu cần restart cả đối chứng. Không xóa output để che lỗi.
Hash implementation nằm trong resume contract: pull code mới giữa run sẽ không
được tự bỏ qua mismatch. Đổi epochs/patience/microbatch cũng không resume chung.

### Nếu OOM

Đã hỗ trợ **explicit** microbatch 64 → 32 → 16 → 8; accumulation lần lượt
16 → 32 → 64 → 128, luôn effective target 1024. Training và validation dùng
microbatch đã chọn. Không tự fallback trong trainer, không đổi lr/loss/epochs.

Tạo **bốn bản config mới**, áp cùng batch cho smoke/pilot và A/B; giữ bản gốc:

```bash
export OOM_MICROBATCH=32
export OOM_CONFIG_DIR="$REPO_ROOT/configs/local_ab_mb32"
python - <<'PY'
import os, json
from pathlib import Path
mb = int(os.environ['OOM_MICROBATCH'])
assert mb in (8,16,32)
folder = Path(os.environ['OOM_CONFIG_DIR'])
folder.mkdir(parents=True, exist_ok=False)
for phase in ('smoke', 'pilot'):
    for case in ('A_raw', 'B_task0_scaler'):
        name = f'{phase}_tddi_p3_fold_ab_{case}_seed0.json'
        c = json.loads((Path('configs') / name).read_text())
        c['training']['batch_size'] = mb
        c['training']['gradient_accumulation_steps'] = 1024 // mb
        c['output_root'] = f'outputs/preprocessing_ab_mb{mb}/{phase}/{c["case"]}'
        with (folder / name).open('x') as f: json.dump(c, f, indent=2)
print(folder)
PY
```

Đổi `AB_ROOT` sang namespace mới, **tạo lại COMMON ở mục 5** để cập nhật array đã
expand giá trị cũ. Dùng path `$OOM_CONFIG_DIR/...json` thay bốn config mặc định
trong dry-run/launch. Chạy lại smoke **cả A/B** dưới cùng điều kiện trước pilot.
Không so A batch64 với B batch32 như một đối chứng preprocessing.

Giảm microbatch không giảm đáng kể bộ nhớ model/teacher/optimizer. Nếu vẫn OOM
ở batch8, dừng gửi log; không tự giảm width, AMP, buffer hoặc đổi method. Cùng
effective batch không bảo đảm bitwise-identical dropout/kernels sau đổi microbatch;
vì vậy cần chạy lại A/B và ghi config.

## 10. So sánh pilot và gửi kết quả

Sau A/B đã hoàn tất task1 và dry-run xác nhận hợp lệ:

```bash
export PILOT_REPORT="$AB_ROOT/reviews/pilot_$(date -u +%Y%m%dT%H%M%SZ)"
python scripts/compare_preprocessing_pilots.py \
  --run-a "$AB_ROOT/pilot/A/member_0" --run-b "$AB_ROOT/pilot/B/member_0" \
  --task-file "$TASK_FILE" --outdir "$PILOT_REPORT" --bundle
cat "$PILOT_REPORT/comparison.md"
tar -tzf "$PILOT_REPORT/review.tar.gz"
```

Comparator không cần GPU, dataset, assignment hoặc checkpoint weights để đọc
report-only bundle. Nó kiểm tra JSON contract/provenance, class map và ID/order
audits, **không thay thế** checkpoint validation của dry-run hoặc data audit gốc.
Lỗi mismatch được báo trước khi xuất so sánh; không xuất bảng điểm không hợp lệ.

Báo cáo có:

- `final_task1_seen_all`: model cuối task1 trên toàn bộ validation class đã thấy
  (58 class), không phải mean task0+task1, không phải full178/task7.
- Task0/task1 `seen_all`, `old`, `current`, số mẫu, Accuracy/Macro-F1/Balanced Accuracy.
  Old tại task1 là class task0; new/current là class task1; argmax trên toàn seen head.
- Epoch curves, best epoch, losses focal/logit KD/feature MSE raw và scaled, total,
  optimizer steps, epoch seconds, runtime từng task, peak allocated/reserved bytes,
  model-only và boundary checkpoint size (nếu có).
- Replay fraction thực tế, class/exemplar coverage, max repeat/cap; JSON còn giữ
  per-class draws, histogram và sampler hashes.
- Delta B−A, **không tự kết luận winner**. Các giá trị telemetry thiếu là null/
  `unavailable`, không bịa hoặc train lại để lấp số.

Task runtime lấy từ summary đến lúc tạo summary, không bao gồm toàn bộ preflight
hoặc flush cuối boundary checkpoint. Tổng wall-clock xem `/usr/bin/time` của job.
Peak checkpoint size trong summary là model-only; boundary còn buffer/state nên
lớn hơn. Peak VRAM phụ thuộc môi trường/job khác, cần gửi GPU log cùng báo cáo.

### Review gọn so với backup đầy đủ

Gửi `review.tar.gz` là gói **report-only**: JSON/Markdown, run_config/provenance,
full P3 JSON, completed-task/input/buffer/epoch/metrics audits và member `.log`.
Không gồm dataset, checkpoint, assignment; không dùng gói này để resume.

Gửi thêm log/config/preflight và manifest điều phối (không đóng gói dataset):

```bash
REVIEW_EXTRAS="$(mktemp -d "$MONITOR_ROOT/review_extras.XXXXXX")"
mkdir "$REVIEW_EXTRAS/default_configs"
cp configs/{smoke,pilot}_tddi_p3_fold_ab_{A_raw,B_task0_scaler}_seed0.json "$REVIEW_EXTRAS/default_configs/"
if [ -n "${OOM_CONFIG_DIR:-}" ]; then
  mkdir "$REVIEW_EXTRAS/oom_configs"
  cp "$OOM_CONFIG_DIR"/*.json "$REVIEW_EXTRAS/oom_configs/"
fi
cp -a "$PREFLIGHT_DIR" "$REVIEW_EXTRAS/preflight"
cp -a "$PREP_ROOT/member_0" "$REVIEW_EXTRAS/preprocessing_member0"
cp -a "$AB_ROOT/pilot/manifests" "$REVIEW_EXTRAS/pilot_manifests"
# Copy các log_dir smoke/pilot thực tế được launch_ab in ra, gồm gpu.csv/nohup/exit_code.
# Ví dụ: cp -a /absolute/path/to/pilot_A.XXXXXX "$REVIEW_EXTRAS/"
tar -czf "$REVIEW_EXTRAS.tar.gz" -C "$REVIEW_EXTRAS" .
```

Không copy cả `MONITOR_ROOT` vào thư mục con của chính nó. Không cần gửi weights
nặng để so điểm; không đưa riêng test report vào để chọn A/B.

**Backup fold đầy đủ** riêng, nếu muốn có thể phục hồi partition:

```bash
FOLD_BACKUP="$(mktemp -d "$MONITOR_ROOT/fold_backup.XXXXXX")"
tar -czf "$FOLD_BACKUP/folds_full.tar.gz" -C "$FOLD_ROOT" \
  fold_assignments.parquet fold_manifest.json
tar -tzf "$FOLD_BACKUP/folds_full.tar.gz"
```

Gói này đủ assignment+manifest, không thay dataset gốc. Muốn backup để **resume
training**, cần thêm toàn bộ member output/checkpoints/task/attempt artifacts,
preprocessing, đúng config/code/environment và vẫn phải giữ ba source Parquets.
Không xóa các file trên server chỉ vì đã tạo gói review nhỏ.

## 11. Điểm dừng

Gửi review pilot + extras; nếu smoke lỗi, gửi smoke logs/audits trước. Chờ người
dùng và Prompt 13 đánh giá validation/trade-off. Không tự chuyển sang full8,
member1/2, đổi exemplar ranking, ensemble hoặc threshold sau khi report hoàn tất.

## 12. Kiểm chứng triển khai trên máy code

Ngày 2026-09-14:

- `python -m pytest -q --junitxml=.pytest_cache/prompt12_regression.xml`:
  **470 passed**, không có failure cũ hoặc failure còn lại.
- 18 test mới bao gồm synthetic A/B → JSON/Markdown → report-only bundle → load
  lại không cần dataset/checkpoint, mismatch guards, missing telemetry, resumed
  attempt directory, OOM config/tiny training, CLI help và cú pháp runbook.
- Các khối Bash được kiểm tra bằng `bash -n`; Python nhúng chỉ parse syntax,
  không thực thi training hoặc preprocessing server.
- `git diff --check`: pass. Không train dataset thật, không tự chọn winner,
  không thay artifact lịch sử và không commit/push trong Prompt 12.

Ngoài script/report/tests và tài liệu, thay đổi trainer chỉ phục vụ microbatch OOM
explicit (32/16/8, accumulation tương ứng); mặc định vẫn 64/16/1024. Logic EWC và
replay legacy không đổi. Contract resume vẫn chặn đổi batch/code giữa run.
