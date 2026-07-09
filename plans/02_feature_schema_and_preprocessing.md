# Feature Schema and Preprocessing

## Objective

Định nghĩa feature schema an toàn, tách biệt I/O khỏi transform logic, và cố định pipeline preprocessing để không tạo leakage giữa train/validation/test.

## Input/Output Definition

- **Input X**: descriptor features sau khi loại explicit 6 meta columns và `class`
- **Target y**: `class`
- **Expected feature count**: `3780` sau audit

## Excluded Columns

```python
meta_cols = [
    "drugid-drug_a",
    "drugid-drug_b",
    "drugname-drug_a",
    "drugname-drug_b",
    "drugsmiles-drug_a",
    "drugsmiles-drug_b",
]
label_col = "class"
```

## Feature Column Policy

```python
feature_cols = [c for c in columns if c not in meta_cols + [label_col]]
```

Checklist bắt buộc:

- [ ] `class` không thuộc `feature_cols`
- [ ] không có `drugid-*` trong `feature_cols`
- [ ] không có `drugname-*` trong `feature_cols`
- [ ] không có `drugsmiles-*` trong `feature_cols`
- [ ] `feature_cols` có đúng 3780 cột sau audit

## Preprocessing Strategy

### Baseline First Pass

1. Đọc từ Parquet.
2. Chọn `feature_cols` theo explicit names.
3. Convert features sang `float32`.
4. Kiểm tra NaN/inf.
5. Impute theo strategy rõ ràng nếu cần.
6. Fit scaler **chỉ trên train split**.
7. Transform train/validation/test bằng cùng scaler.
8. Lưu đầy đủ artifact và báo cáo.

### Default Baseline Recommendation

- `float32`
- `StandardScaler`
- fit trên train only
- không log-transform ở vòng đầu
- không clipping ở vòng đầu nếu chưa có bằng chứng outlier gây vỡ training

### Planned Ablations

- `RobustScaler`
- quantile clipping `0.1% - 99.9%`
- `log1p` nếu audit xác nhận feature không âm và việc biến đổi có ý nghĩa
- class-balanced sampler được xem là thành phần training, không gộp vào preprocessing

## Out-of-Core and Memory-Aware Option

Train Parquet vẫn đủ lớn để việc đọc full bằng pandas rồi materialize toàn bộ `train + validation + test` cùng lúc có thể nặng RAM. Vì vậy cần có phương án out-of-core ngay trong plan triển khai.

### Preferred Fallbacks When RAM Is Tight

- `Polars` lazy scan
- `PyArrow Dataset`
- chunked/batched reading theo row groups
- chỉ materialize split đang cần xử lý

### Scaler Strategy Under Memory Pressure

- fit scaler theo incremental mean/variance nếu dùng chuẩn hóa kiểu standard
- hoặc fit trên sample train đại diện có seed cố định, nhưng phải log rõ sampling policy
- không materialize toàn bộ ma trận train nếu không cần

### Rules

- không đọc đồng thời full `train`, `validation`, `test` nếu không có lý do rõ
- không tạo thêm bản sao float32/full matrix ngoài nhu cầu hiện tại
- mọi fallback out-of-core phải cho kết quả deterministic hoặc log rõ nguồn sai khác

## Missing/Inf Strategy

Ưu tiên conservative:

1. audit số lượng và tỷ lệ missing/inf theo cột
2. nếu hiếm và nằm ở descriptor numeric thì ưu tiên median imputation theo train
3. chỉ dùng fill `0` nếu descriptor semantics hợp lý và được ghi rõ trong report
4. không fit imputer trên validation/test

## Sparse Zeros and Long-Tail Considerations

- nhiều descriptor có thể chứa nhiều số 0 hoặc phân phối lệch
- không xóa cột zero-heavy chỉ vì hiếm xuất hiện nếu chưa audit variance và ảnh hưởng mô hình
- nếu có near-constant columns, đánh dấu để ablation sau; không tự động loại ngay ở baseline đầu tiên trừ khi rõ ràng vô ích hoặc gây lỗi số

## Required Artifacts

```text
outputs/preprocess/feature_columns.json
outputs/preprocess/meta_columns.json
outputs/preprocess/scaler_config.json
outputs/preprocess/scaler.pkl
outputs/preprocess/preprocessing_report.md
```

## Required Script

```text
scripts/preprocess_features.py
```

## Pseudocode

```python
train = read_parquet(train_path)
valid = read_parquet(valid_path)
test = read_parquet(test_path)

feature_cols = load_or_build_feature_columns(train.columns)
X_train = train[feature_cols].astype("float32")
X_valid = valid[feature_cols].astype("float32")
X_test = test[feature_cols].astype("float32")

check_missing_inf(X_train, X_valid, X_test)
imputer = fit_imputer_on_train(X_train)
X_train = imputer.transform(X_train)
X_valid = imputer.transform(X_valid)
X_test = imputer.transform(X_test)

scaler = StandardScaler().fit(X_train)
save(scaler)
```

Nếu RAM không đủ, pseudocode tương đương phải chuyển sang batched processing:

```python
for batch in scan_parquet(train_path, columns=feature_cols):
    X_batch = batch.to_numpy().astype("float32")
    incremental_scaler.partial_fit(X_batch)
```

## Anti-Leakage Rules

- không suy ra `feature_cols` bằng kiểu dữ liệu
- không fit scaler hoặc imputer trên validation/test
- không dùng thống kê của test set để chọn preprocessing config
- không đưa meta text vào pipeline numeric

## Planned Commands

```bash
python scripts/preprocess_features.py \
  --train /mnt/data/uyen/data_splits/train_extracted.parquet \
  --validation /mnt/data/uyen/data_splits/validation_extracted.parquet \
  --test /mnt/data/uyen/data_splits/test_extracted.parquet \
  --feature-cols outputs/audit/feature_columns.json \
  --scaler standard \
  --outdir outputs/preprocess
```

## Acceptance Criteria

- [ ] Có artifact lưu rõ `feature_cols`
- [ ] Có artifact lưu rõ scaler config
- [ ] Báo cáo ghi rõ scaler chỉ fit trên train
- [ ] Có nhánh ablation rõ ràng cho `RobustScaler`
- [ ] Có phương án out-of-core rõ ràng nếu RAM không đủ
- [ ] Không có bước nào dùng test để tuning preprocessing

## Definition of Done

- [ ] Có `scripts/preprocess_features.py`
- [ ] Có đủ artifact trong `outputs/preprocess/`
- [ ] Có `preprocessing_report.md` nêu assumption, risk, và quyết định baseline
