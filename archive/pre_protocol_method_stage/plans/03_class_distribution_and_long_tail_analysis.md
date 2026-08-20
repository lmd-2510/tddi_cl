# Class Distribution and Long-Tail Analysis

## Objective

Đo mức độ mất cân bằng class, khóa chặt thống kê nền cho protocol CIL, và chuẩn bị logic chia task không làm benchmark méo do class order quá lệch.

## Known Facts to Re-Verify by Script

Theo mô tả đã đọc đúng từ Parquet trước đó:

- cả 3 split có `178` classes
- `class` có kiểu `int64`
- long-tail mạnh

Các thống kê cụ thể như top classes và số lượng lớp hiếm phải được script xuất lại để đảm bảo reproducibility.

## Why Long-Tail Matters for CIL

- lớp lớn dễ thống trị gradient
- lớp hiếm dễ bị quên sau các task sau
- task split ngẫu nhiên có thể tạo task quá dễ hoặc quá khó
- accuracy tổng dễ che khuất thất bại ở tail classes
- replay buffer cần cân bằng theo lớp, không lấy random toàn cục

## Required Script

```text
scripts/analyze_class_distribution.py
```

## Planned Outputs

```text
outputs/class_distribution/class_counts_train.csv
outputs/class_distribution/class_counts_validation.csv
outputs/class_distribution/class_counts_test.csv
outputs/class_distribution/class_distribution_summary.md
outputs/class_distribution/rare_classes.csv
outputs/figures/class_distribution_train.png
outputs/figures/class_distribution_logscale.png
```

## Required Statistics

- `num_classes`
- `min_count`
- `max_count`
- `median_count`
- `mean_count`
- `num_classes_<=5`
- `num_classes_<=10`
- `num_classes_<=20`
- `num_classes_<=50`
- `imbalance_ratio = max_count / min_count`

## Planned Tables

1. Class count table cho từng split
2. Top-10 class counts
3. Tail-class table cho các ngưỡng `<=5`, `<=10`, `<=20`, `<=50`
4. Head / medium / tail bin summary dùng cho evaluation sau này

## Planned Figures

- histogram hoặc bar chart class counts cho train
- plot log-scale để nhìn đuôi phân bố rõ hơn
- optional cumulative coverage plot: top-k classes bao phủ bao nhiêu phần trăm sample

## Suggested Head/Medium/Tail Bins

Bin proposal ban đầu, cần audit trước khi cố định:

- `head`: top frequency bin hoặc count `> 1000`
- `medium`: khoảng giữa
- `tail`: count `<= 20` hoặc `<= 50` tùy mức chi tiết

Ngưỡng cuối cùng phải được lưu trong config/báo cáo, không hardcode âm thầm.

## Rare-Class Policy

Main benchmark phải **giữ toàn bộ 178 classes**, kể cả các lớp train chỉ có vài mẫu.

Tuy nhiên cần policy diễn giải rõ:

- báo cáo riêng rare classes `<=5`, `<=10`, `<=20`
- không overclaim per-class F1 của lớp chỉ có 3-4 mẫu
- macro metrics phải luôn đi kèm context về tail counts
- có thể thêm ablation phụ `min-count filtered benchmark`, ví dụ chỉ giữ classes với `train_count >= 20`, nhưng không thay main benchmark

Artifact summary nên nêu rõ lớp nào quá hiếm để metric ở level từng lớp dễ nhiễu.

## Pseudocode

```python
for split in ["train", "validation", "test"]:
    y = read_parquet(split_path, columns=["class"])["class"]
    counts = y.value_counts().sort_index()
    save_counts(split, counts)

summary = summarize_distribution(train_counts, val_counts, test_counts)
rare = build_rare_class_table(train_counts, thresholds=[5, 10, 20, 50])
plot_counts(train_counts)
plot_counts_logscale(train_counts)
```

## Decision Points Supported by This Plan

- protocol nào dùng làm main benchmark
- task order nào công bằng hơn cho long-tail dataset
- memory per class có đủ để giữ old classes không
- macro metrics nào cần ưu tiên

## Planned Command

```bash
python scripts/analyze_class_distribution.py \
  --train /mnt/data/uyen/data_splits/train_extracted.parquet \
  --validation /mnt/data/uyen/data_splits/validation_extracted.parquet \
  --test /mnt/data/uyen/data_splits/test_extracted.parquet \
  --outdir outputs/class_distribution
```

## Checklist

- [ ] Xác nhận `178` classes ở cả 3 split
- [ ] Xuất class count cho train/validation/test
- [ ] Có top classes và rare classes
- [ ] Có imbalance ratio
- [ ] Có biểu đồ thường và log-scale
- [ ] Có policy diễn giải cho classes quá hiếm
- [ ] Có phần giải thích tác động lên CIL metrics và replay

## Acceptance Criteria

- [ ] Người triển khai có thể dùng file này để chọn protocol class order hợp lý
- [ ] Báo cáo nêu rõ vì sao `Macro-F1` và `Balanced Accuracy` quan trọng hơn accuracy tổng
- [ ] Tail-class risk được liên kết trực tiếp với forgetting analysis sau này

## Definition of Done

- [ ] Có `scripts/analyze_class_distribution.py`
- [ ] Có đủ artifact trong `outputs/class_distribution/` và `outputs/figures/`
- [ ] Có `class_distribution_summary.md` làm đầu vào cho task builder
