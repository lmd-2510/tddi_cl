# Runbook: P0–P8 với T-DDI

## 1. Kiểm tra môi trường

```bash
cd /media/neeyu/hien_3/uyen/data_splits
test -x .venv/bin/python
.venv/bin/python -c "import torch, pandas, pyarrow, sklearn; print(torch.__version__, torch.cuda.is_available())"
df -h .
free -h
```

Nếu phải cài lại môi trường:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

## 2. Kiểm tra artifact đầu vào

```bash
test -s train_extracted.parquet
test -s validation_extracted.parquet
test -s test_extracted.parquet
test -s outputs/audit/feature_columns.json
test -s outputs/preprocess/scaler.pkl
.venv/bin/python -m unittest discover -s tests -p 'test_*.py' -v
bash -n scripts/run_protocol_study.sh
```

Task files chuẩn phải nằm trong `outputs/tasks/`. Runner sẽ báo lỗi và dừng nếu file cần dùng bị thiếu.

## 3. Tạo lại P0–P4 khi cần

Lệnh này tạo ở thư mục tạm để audit/diff trước khi thay artifact chuẩn.

```bash
protocol_tmpdir="$(mktemp -d)"
.venv/bin/python scripts/build_cil_tasks.py \
  --class-counts outputs/class_distribution/class_counts_train.csv \
  --validation-counts outputs/class_distribution/class_counts_validation.csv \
  --test-counts outputs/class_distribution/class_counts_test.csv \
  --outdir "$protocol_tmpdir" \
  --protocol all \
  --seeds 0 1 2 3 4
```

## 4. Tạo lại P5–P8 khi cần

P5–P8 cần static T-DDI reference và signals. Không dùng test trong bước này. Artifact hiện tại nằm ở `outputs/advanced_protocols/`.

```bash
.venv/bin/python scripts/prepare_advanced_protocol_signals.py \
  --train train_extracted.parquet \
  --validation validation_extracted.parquet \
  --feature-cols outputs/audit/feature_columns.json \
  --scaler outputs/preprocess/scaler.pkl \
  --checkpoint outputs/advanced_protocols/static_tddi_reference/best_model.pt \
  --class-map outputs/advanced_protocols/static_tddi_reference/global_class_map.json \
  --outdir outputs/advanced_protocols/signals_new \
  --variant tddi --batch-size 1024 --device auto
```

Sau khi kiểm tra signals, build task vào thư mục mới để tránh ghi đè:

```bash
.venv/bin/python scripts/build_advanced_protocols.py \
  --class-counts outputs/class_distribution/class_counts_train.csv \
  --class-stats outputs/advanced_protocols/signals/class_protocol_stats.csv \
  --difficulty outputs/advanced_protocols/signals/validation_class_difficulty.csv \
  --confusion-edges outputs/advanced_protocols/signals/validation_confusion_edges.csv \
  --outdir outputs/tasks_new \
  --protocol all --seeds 0 1 2 3 4
```

## 5. Chạy trong tmux

Một protocol:

```bash
tmux new-session -d -s ddi_p4 \
  "cd /media/neeyu/hien_3/uyen/data_splits && bash scripts/run_protocol_study.sh P4 2>&1 | tee outputs/runs_backbones/p4_tddi_driver.log"
```

Tất cả protocol:

```bash
tmux new-session -d -s ddi_protocols \
  "cd /media/neeyu/hien_3/uyen/data_splits && bash scripts/run_protocol_study.sh all 2>&1 | tee outputs/runs_backbones/protocol_study_tddi_driver.log"
```

Kiểm tra session/log:

```bash
tmux ls
tmux attach -t ddi_protocols
tail -f /media/neeyu/hien_3/uyen/data_splits/outputs/runs_backbones/protocol_study_tddi_driver.log
```

Nhấn `Ctrl-b`, sau đó `d` để detach. Không dùng đường dẫn `~/outputs/...`; output nằm dưới root của repo.

## 6. Cơ chế bảo vệ máy

Runner kiểm tra trước mỗi run:

- RAM khả dụng tối thiểu 12 GiB (`PROTOCOL_MIN_AVAILABLE_GIB` có thể override có chủ đích);
- ổ đĩa trống tối thiểu 50 GiB (`PROTOCOL_MIN_DISK_GIB` có thể override);
- task file tồn tại và không rỗng;
- run hoàn tất thì skip;
- run directory tồn tại nhưng chưa hoàn tất thì dừng, không overwrite;
- chạy tuần tự từng protocol/seed, không khởi chạy nhiều GPU job song song.

Ví dụ chỉ khi đã tự kiểm tra máy và cần hạ ngưỡng:

```bash
PROTOCOL_MIN_AVAILABLE_GIB=8 PROTOCOL_MIN_DISK_GIB=30 \
  bash scripts/run_protocol_study.sh P4
```

## 7. Điều kiện một run hoàn tất

Một run chỉ được tính khi cùng tồn tại và không rỗng:

```text
run_summary.md
metrics.csv
forgetting.csv
```

`run_config.json` ghi argument, protocol, seed, git state và checksum implementation để truy vết.
