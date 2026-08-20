# Reproducibility and Reporting

## Objective

Đảm bảo mọi run của DDI2025-CIL có thể tái tạo, audit, và tổng hợp thành báo cáo mà không mơ hồ về config, schema, task order, hay artifact.

## Reproducibility Principles

- fixed seeds
- config-driven behavior
- feature schema saved
- task JSON saved
- command log saved
- run outputs self-describing
- test set không dùng cho tuning

## Planned Config Layout

```text
configs/
  data.yaml
  model_mlp_base.yaml
  cil_random_8tasks.yaml
  cil_long_tail.yaml
  method_sequential.yaml
  method_replay_50.yaml
  method_replay_distill_50.yaml
```

## Run Artifact Layout

Mỗi run phải lưu:

```text
outputs/runs/{run_id}/config.yaml
outputs/runs/{run_id}/metrics.csv
outputs/runs/{run_id}/task_matrix.csv
outputs/runs/{run_id}/forgetting.csv
outputs/runs/{run_id}/checkpoint_paths.txt
outputs/runs/{run_id}/run_summary.md
outputs/runs/{run_id}/train.log
outputs/runs/{run_id}/stdout.log
outputs/runs/{run_id}/events.csv
```

## Run ID Convention

Run ID nên chứa:

- protocol
- seed
- method
- model
- memory size nếu có
- timestamp

Ví dụ:

```text
random_seed0_replay50_mlpbase_20260706T120000
```

## Required Metadata to Save

- dataset path
- file names used
- feature schema hash
- task JSON hash
- random seeds
- model config
- preprocessing config
- git commit hash nếu repo đã tồn tại về sau
- environment file hoặc package snapshot

## Logging and Progress Capture

Mỗi run phải lưu đồng thời:

- log ra console/stdout cho người chạy theo dõi trực tiếp
- log file để audit sau khi run xong hoặc crash giữa chừng

### Minimum Console Fields

- `run_id`
- `task_id`
- `epoch`
- `train_loss`
- `val_macro_f1`
- `best_metric_so_far`
- learning rate
- elapsed time
- checkpoint save event

### Minimum Event Types

`events.csv` nên có ít nhất các loại event:

- `run_started`
- `task_started`
- `epoch_completed`
- `best_model_updated`
- `checkpoint_saved`
- `early_stopped`
- `task_completed`
- `run_completed`
- `run_failed`

### Failure Logging

Nếu run lỗi giữa chừng, vẫn phải lưu:

- exception message
- task đang chạy
- epoch gần nhất
- checkpoint cuối cùng đã lưu thành công
- config của run

## Reporting Rules

- báo cáo mean ± std qua nhiều seeds
- tách result chính và ablation
- không báo test-selected best config như main result nếu tuning trên test
- ghi rõ metric chính là gì
- ghi rõ benchmark chính là Random-CIL hay Long-tail CIL

## Recommended Summary Files

```text
reports/ddi2025_cil_report.md
reports/tables/
reports/figures/
```

`run_summary.md` nên có:

1. objective của run
2. task file dùng
3. config chính
4. metric theo task
5. lỗi hoặc warning
6. artifact paths

## Reproducibility Checklist

- [ ] Dataset path recorded
- [ ] Parquet files used
- [ ] Feature columns saved
- [ ] Meta columns excluded
- [ ] Label column excluded from `X`
- [ ] Task split JSON saved
- [ ] Random seeds fixed
- [ ] Config saved
- [ ] Model checkpoint saved
- [ ] Metrics saved per task
- [ ] Console progress được mirror vào file log
- [ ] Checkpoint save events được log lại
- [ ] Mean ± std reported across seeds
- [ ] Test set not used for hyperparameter tuning

## Command Logging

Mỗi run nên lưu command line đầy đủ hoặc resolved config snapshot để:

- rerun chính xác
- audit khác biệt giữa 2 run
- tránh “same name, different config”

Nếu dùng shell redirect hay job scheduler, vẫn cần đảm bảo nội dung progress chính được ghi vào `stdout.log` hoặc `train.log`, không chỉ hiện tạm trên terminal.

## Planned Environment Artifacts

- `requirements.txt` hoặc lock file
- `environment.yaml` nếu dùng conda
- `system_info.txt` hoặc `torch_env.txt` nếu cần

## Acceptance Criteria

- [ ] Có quy ước artifact nhất quán
- [ ] Có checklist chống test leakage
- [ ] Có schema để tổng hợp mean ± std theo seed
- [ ] Người khác có thể truy ngược một kết quả về đúng config và task file đã dùng
- [ ] Người khác có thể xem lại toàn bộ tiến trình run từ log file mà không cần terminal gốc

## Definition of Done

- [ ] File này đủ để làm chuẩn lưu trữ run và viết report reproducible
