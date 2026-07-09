# Experiment Matrix

## Objective

Chia dự án thành các giai đoạn tăng dần độ khó để tránh chạy full continual benchmark khi pipeline nền còn chưa được audit xong.

## Stage 0: Data Sanity

### Scope

- audit schema
- audit feature count
- class distribution
- leakage/overlap check
- chưa train model

### Outputs

- `outputs/audit/*`
- `outputs/class_distribution/*`
- `outputs/leakage/*`

### Exit Criteria

- [ ] xác nhận `3780` feature columns
- [ ] xác nhận `178` classes
- [ ] có leakage report
- [ ] có task protocol draft

## Stage 1: Static Supervised Sanity Check

### Scope

Train `MLP-base` trên full train split, validate trên validation, test trên test.

### Goal

- kiểm tra pipeline đọc dữ liệu đúng
- kiểm tra feature/label không bị leakage
- có baseline static để so sánh với continual runs

### Metrics

- Accuracy
- Macro-F1
- Balanced Accuracy
- Per-class F1

### Exit Criteria

- [ ] model train/eval chạy end-to-end
- [ ] metric hợp lý, không có dấu hiệu leakage bất thường
- [ ] có report per-class

## Stage 2: Main Random-CIL

### Scope

```text
protocol = random
seeds = 0, 1, 2, 3, 4
tasks = 8
model = MLP-base
methods = sequential, replay_50, replay_distill_50, joint_seen
```

### Goal

Thiết lập benchmark chính cho báo cáo đầu tiên.

### Exit Criteria

- [ ] đủ 5 seeds
- [ ] có mean ± std
- [ ] có performance matrix và forgetting metrics

## Stage 3: Memory Ablation

### Scope

```text
memory_per_class = 20, 50, 100
method = replay
protocol = random seed 0 first
```

### Goal

Định lượng trade-off giữa bộ nhớ và hiệu năng.

### Exit Criteria

- [ ] có kết quả seed 0 cho 3 mức memory
- [ ] nếu ổn mới mở rộng 5 seeds

## Stage 4: Long-Tail CIL

### Scope

```text
protocol = long_tail
methods = sequential, replay_50, replay_distill_50, joint_seen
```

### Goal

- đo tác động của class hiếm xuất hiện muộn
- phân tích rare-class forgetting

### Exit Criteria

- [ ] có so sánh với Random-CIL
- [ ] có rare-class analysis

## Stage 5: Calibration Drift

### Scope

Chạy trên best methods:

- `sequential`
- `replay_50`
- `replay_distill_50`
- `joint_seen`

### Metrics

- ECE by task
- Brier by task
- reliability diagram
- selective risk/coverage nếu khả thi

### Exit Criteria

- [ ] có calibration report theo task
- [ ] có final reliability diagram

## Optional Stage 6: EWC/SI

Chỉ chạy nếu:

- baseline replay đã ổn
- còn thời gian và compute

## Optional Stage 7: Pair-Disjoint or Drug-Incremental Split

Chỉ làm sau khi main CIL protocol đã ổn và leakage report cho thấy cần mở rộng benchmark.

## Experiment Table

| Stage | Protocol | Method | Seed Policy | Purpose |
|---|---|---|---|---|
| 0 | none | audit only | none | khóa schema và leakage |
| 1 | static | MLP-base | 1 seed ban đầu | sanity check |
| 2 | random | sequential/replay/replay_distill/joint_seen | 5 seeds | main benchmark |
| 3 | random | replay | seed 0 trước | memory ablation |
| 4 | long_tail | sequential/replay/replay_distill/joint_seen | 1-5 seeds tùy compute | tail analysis |
| 5 | random/long_tail | selected methods | best methods | calibration drift |

## Acceptance Criteria

- [ ] Có thứ tự chạy rõ ràng từ audit đến full benchmark
- [ ] Không nhảy vào full CIL khi static sanity chưa ổn
- [ ] Mỗi stage có exit criteria riêng
- [ ] Optional stages được đánh dấu rõ là không thuộc main scope ban đầu

## Definition of Done

- [ ] Người triển khai có thể dùng file này làm checklist thực thi theo stage
- [ ] Có thể trả lời “bước tiếp theo là gì” ở mọi thời điểm của dự án
