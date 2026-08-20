# Execution Roadmap

## Objective

Biến toàn bộ kế hoạch thành roadmap triển khai theo phase với deliverable, Definition of Done, và command dự kiến.

## Proposed Repo Structure

```text
ddi2025-cil/
  configs/
    data.yaml
    model_mlp_base.yaml
    cil_random_8tasks.yaml
    cil_long_tail.yaml
    method_sequential.yaml
    method_replay_50.yaml
    method_replay_distill_50.yaml

  plans/
    00_project_overview.md
    01_data_audit_and_schema_qc.md
    02_feature_schema_and_preprocessing.md
    03_class_distribution_and_long_tail_analysis.md
    04_leakage_and_overlap_checks.md
    05_class_incremental_protocol.md
    06_model_and_baseline_design.md
    07_replay_and_distillation_design.md
    08_metrics_and_evaluation_protocol.md
    09_experiment_matrix.md
    10_execution_roadmap.md
    11_risks_limitations_and_ablation.md
    12_reproducibility_and_reporting.md

  scripts/
    inspect_splits.py
    analyze_class_distribution.py
    check_leakage.py
    build_cil_tasks.py
    preprocess_features.py
    summarize_results.py

  src/
    data/
      ddi_dataset.py
      feature_schema.py
      class_mapping.py
      task_sampler.py
      replay_buffer.py
    models/
      mlp.py
    methods/
      joint_seen.py
      sequential.py
      replay.py
      replay_distill.py
      ewc.py
    training/
      train_static.py
      train_cil.py
    eval/
      classification_metrics.py
      continual_metrics.py
      calibration_metrics.py
      rare_class_metrics.py
    utils/
      io.py
      seed.py
      logging.py

  outputs/
    audit/
    preprocess/
    class_distribution/
    leakage/
    tasks/
    memory/
    checkpoints/
    runs/
    results/
    figures/

  reports/
    ddi2025_cil_report.md
```

## Phase 1: Planning and Audit

### Deliverables

```text
plans/*.md
scripts/inspect_splits.py
scripts/analyze_class_distribution.py
scripts/check_leakage.py
outputs/audit/*
outputs/class_distribution/*
outputs/leakage/*
```

### Definition of Done

- [ ] Có `feature_columns.json`
- [ ] Xác nhận `3780` feature columns
- [ ] Xác nhận `178` classes
- [ ] Có class distribution report
- [ ] Có leakage report
- [ ] Có task protocol draft

## Phase 2: Preprocessing and Task Construction

### Deliverables

```text
scripts/build_cil_tasks.py
scripts/preprocess_features.py
outputs/preprocess/*
outputs/tasks/*
```

### Definition of Done

- [ ] Tạo đủ random task seeds `0-4`
- [ ] Tạo frequency-balanced tasks
- [ ] Tạo long-tail tasks
- [ ] Mỗi class xuất hiện đúng một lần
- [ ] Có task sample count summary

## Phase 3: Static Baseline

### Deliverables

```text
src/models/mlp.py
src/data/ddi_dataset.py
src/training/train_static.py
outputs/results/static_baseline/*
```

### Definition of Done

- [ ] MLP chạy được trên full train/validation/test
- [ ] Không có label leakage
- [ ] Có static baseline metrics
- [ ] Có per-class metrics
- [ ] Có console log theo epoch và file log của run

## Phase 4: CIL Baselines

### Deliverables

```text
src/training/train_cil.py
src/methods/sequential.py
src/methods/replay.py
src/methods/replay_distill.py
src/eval/continual_metrics.py
outputs/results/cil/*
```

### Definition of Done

- [ ] Chạy được sequential trên 1 seed
- [ ] Chạy được replay_50 trên 1 seed
- [ ] Chạy được replay_distill_50 trên 1 seed
- [ ] Có performance matrix
- [ ] Có forgetting metrics
- [ ] Có log theo task, theo epoch, và checkpoint events

## Phase 5: Full Experiment

### Deliverables

```text
outputs/results/main_random_cil_5seeds/*
outputs/results/long_tail/*
outputs/results/memory_ablation/*
outputs/figures/*
```

### Definition of Done

- [ ] Có kết quả 5 seeds Random-CIL
- [ ] Có mean ± std
- [ ] Có forgetting analysis
- [ ] Có rare-class analysis
- [ ] Có calibration analysis
- [ ] Có log file đầy đủ để audit các run lỗi hoặc dừng sớm

## Phase 6: Report

### Deliverables

```text
reports/ddi2025_cil_report.md
reports/tables/
reports/figures/
```

### Definition of Done

- [ ] Có abstract ngắn
- [ ] Có dataset section
- [ ] Có method section
- [ ] Có experiment section
- [ ] Có result tables
- [ ] Có limitations
- [ ] Có reproducibility checklist

## Proposed Commands

### Audit

```bash
python scripts/inspect_splits.py \
  --train /mnt/data/uyen/data_splits/train_extracted.parquet \
  --validation /mnt/data/uyen/data_splits/validation_extracted.parquet \
  --test /mnt/data/uyen/data_splits/test_extracted.parquet \
  --outdir outputs/audit
```

### Distribution

```bash
python scripts/analyze_class_distribution.py \
  --train /mnt/data/uyen/data_splits/train_extracted.parquet \
  --validation /mnt/data/uyen/data_splits/validation_extracted.parquet \
  --test /mnt/data/uyen/data_splits/test_extracted.parquet \
  --outdir outputs/class_distribution
```

### Leakage

```bash
python scripts/check_leakage.py \
  --train /mnt/data/uyen/data_splits/train_extracted.parquet \
  --validation /mnt/data/uyen/data_splits/validation_extracted.parquet \
  --test /mnt/data/uyen/data_splits/test_extracted.parquet \
  --outdir outputs/leakage
```

### Task Builder

```bash
python scripts/build_cil_tasks.py \
  --class-counts outputs/class_distribution/class_counts_train.csv \
  --num-classes 178 \
  --protocol random \
  --num-tasks 8 \
  --base-task-classes 38 \
  --increment-classes 20 \
  --seeds 0 1 2 3 4 \
  --outdir outputs/tasks
```

### Static Training

```bash
python src/training/train_static.py \
  --config configs/model_mlp_base.yaml \
  --data-config configs/data.yaml \
  --outdir outputs/runs/static_mlpbase
```

### Continual Training

```bash
python src/training/train_cil.py \
  --data-config configs/data.yaml \
  --model-config configs/model_mlp_base.yaml \
  --task-file outputs/tasks/random_seed0_tasks.json \
  --method replay_distill \
  --memory-per-class 50 \
  --temperature 2.0 \
  --distill-alpha 1.0 \
  --outdir outputs/runs/random_seed0_replaydistill50_mlpbase
```

## Sequencing Rules

- không qua Phase 2 nếu Phase 1 chưa khóa xong feature schema
- không qua Phase 4 nếu static baseline chưa chạy đúng
- không làm Phase 5 khi phase 4 chưa có result matrix hợp lệ

## Logging Deliverables

Mỗi training run nên tạo thêm:

```text
outputs/runs/{run_id}/train.log
outputs/runs/{run_id}/stdout.log
outputs/runs/{run_id}/events.csv
```

`events.csv` nên ghi các mốc:

- run start
- task start/end
- epoch end
- best metric update
- checkpoint saved
- early stopping triggered
- run end

## Logging Definition of Done

- [ ] Console có progress bar hoặc progress lines rõ ràng
- [ ] Có log theo task và theo epoch
- [ ] Có dòng log khi save checkpoint
- [ ] Có warning log khi gặp `NaN/inf` hoặc thiếu exemplar replay
- [ ] Có file log lưu cùng artifact của run

## Acceptance Criteria

- [ ] Có deliverable rõ cho từng phase
- [ ] Có command dự kiến cho các bước chính
- [ ] Có Definition of Done đủ cụ thể để dùng như checklist triển khai
- [ ] Logging/progress không còn là phần mơ hồ

## Definition of Done

- [ ] File này đủ để làm bảng điều phối công việc theo phase cho dự án
