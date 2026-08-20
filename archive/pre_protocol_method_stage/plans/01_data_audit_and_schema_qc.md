# Data Audit and Schema QC

## Objective

Khóa chặt schema, row/column counts, feature boundaries, kiểu dữ liệu, và các rủi ro parse trước khi viết bất kỳ code preprocess hoặc training nào.

## Input Files

| Split | Preferred Input | Fallback |
|---|---|---|
| train | `/mnt/data/uyen/data_splits/train_extracted.parquet` | `/mnt/data/uyen/data_splits/train_extracted.csv` |
| validation | `/mnt/data/uyen/data_splits/validation_extracted.parquet` | `/mnt/data/uyen/data_splits/validation_extracted.csv` |
| test | `/mnt/data/uyen/data_splits/test_extracted.parquet` | `/mnt/data/uyen/data_splits/test_extracted.csv` |

## Why Parquet First

- file nhỏ hơn CSV
- schema rõ ràng hơn
- tránh lỗi parser do text field có quote/comma
- đọc chọn cột hiệu quả hơn
- phù hợp cho audit và training lặp lại

**Rule:** không dùng `awk -F,` để kiểm tra schema CSV.

## Required Schema Facts

Đã xác nhận từ Parquet metadata:

- mỗi split có `3787` cột
- `train` có `520841` dòng
- `validation` có `173614` dòng
- `test` có `173614` dòng
- `class` ở vị trí `3248`
- `drugid-drug_a`, `drugid-drug_b`, `drugname-drug_a`, `drugname-drug_b`, `drugsmiles-drug_a`, `drugsmiles-drug_b` tồn tại

Các fact còn lại phải được audit bằng script, không kết luận bằng suy đoán.

## Audit Questions

1. Có đúng 3787 cột ở cả 3 split không?
2. Có đúng 6 cột meta và 1 cột nhãn cần loại khỏi feature space không?
3. Sau khi loại 7 cột này, số feature có đúng `3780` không?
4. Có duplicate column names không?
5. Có NaN, inf, hoặc constant columns đáng kể không?
6. `class` có phải integer và nhất quán giữa các split không?
7. Có cột nào mang kiểu object/string nhưng lại lọt vào candidate feature space không?

## Class ID Policy

Không được giả định class IDs liên tục kiểu `0..177`.

Vì dataset có `178` lớp nhưng giá trị class thực có thể không liên tục, code audit và training không được viết:

```python
num_classes = max(class_values) + 1
```

Phải audit và lưu explicit mapping:

```python
unique_classes = sorted(train["class"].unique())
global_class_map = {cls: i for i, cls in enumerate(unique_classes)}
```

Artifact tối thiểu cần thêm:

```text
outputs/audit/global_class_map.json
outputs/audit/class_id_summary.csv
```

## Explicit Column Policy

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

feature_cols = [c for c in columns if c not in meta_cols + [label_col]]
```

Không lấy feature bằng heuristic “giữ tất cả cột số”.

## Required Script

```text
scripts/inspect_splits.py
```

## Planned Outputs

```text
outputs/audit/schema_summary.json
outputs/audit/row_col_counts.csv
outputs/audit/column_positions.csv
outputs/audit/meta_columns_check.json
outputs/audit/feature_columns.json
outputs/audit/global_class_map.json
outputs/audit/class_id_summary.csv
outputs/audit/missing_inf_report.csv
outputs/audit/constant_columns.csv
outputs/audit/audit_summary.md
```

## Minimum Checks

- [ ] Mỗi Parquet có 3787 cột
- [ ] train có 520841 dòng
- [ ] validation có 173614 dòng
- [ ] test có 173614 dòng
- [ ] `class` tồn tại trong cả 3 split
- [ ] `class` dtype là integer hoặc convert an toàn sang integer
- [ ] `178` lớp được xác định từ `nunique`, không suy ra từ `max(class)+1`
- [ ] class IDs thực được lưu trong `global_class_map.json`
- [ ] 6 meta columns tồn tại đầy đủ
- [ ] `class` không nằm trong `feature_cols`
- [ ] 6 meta columns không nằm trong `feature_cols`
- [ ] `feature_cols` có đúng 3780 cột
- [ ] không có duplicate column names
- [ ] báo cáo NaN/inf theo cột
- [ ] báo cáo constant và near-constant columns

## Suggested Audit Flow

1. Mở Parquet metadata và lưu row/column counts.
2. Đọc schema và cột tên từ Parquet.
3. Xác nhận vị trí của `class` và các meta columns.
4. Sinh `feature_cols` bằng explicit exclusion.
5. Quét từng split để tính missing, inf, nunique, zero-ratio nếu cần.
6. Gộp thành báo cáo Markdown và JSON.

## Pseudocode

```python
for split in ["train", "validation", "test"]:
    table = read_parquet(split_path)
    assert label_col in table.columns
    for c in meta_cols:
        assert c in table.columns

    feature_cols = [c for c in table.columns if c not in meta_cols + [label_col]]
    assert label_col not in feature_cols
    assert len(feature_cols) == 3780

    summarize_schema(table, feature_cols)
    check_missing_inf(table, feature_cols)
    check_constant_columns(table, feature_cols)

unique_classes = sorted(train[label_col].unique())
global_class_map = {cls: i for i, cls in enumerate(unique_classes)}
assert len(unique_classes) == 178
save_json(global_class_map, "outputs/audit/global_class_map.json")
```

## Acceptance Criteria

- [ ] `feature_columns.json` được tạo ra từ explicit exclusion, không phải heuristic
- [ ] báo cáo chỉ ra rõ `class` nằm giữa bảng nên có nguy cơ leakage
- [ ] báo cáo chỉ ra rõ class IDs có thể không liên tục
- [ ] audit report tách riêng phần “đã xác nhận” và phần “cần verify”
- [ ] không có bước nào sửa dữ liệu gốc

## Definition of Done

- [ ] Có `scripts/inspect_splits.py`
- [ ] Có đủ artifact audit trong `outputs/audit/`
- [ ] Có `audit_summary.md` tóm tắt các phát hiện quan trọng
- [ ] Có kết luận rõ ràng về số feature kỳ vọng `3780`
