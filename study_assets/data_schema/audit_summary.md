# Audit Summary

## Key Facts

- Canonical label column: `class`
- Canonical meta columns: `drugid-drug_a, drugid-drug_b, drugname-drug_a, drugname-drug_b, drugsmiles-drug_a, drugsmiles-drug_b`
- Canonical feature count from explicit exclusion: `3780`
- Train class cardinality from `nunique`: `178`
- Partial scan mode: `False`

## Checklist

- [x] Mỗi Parquet có 3787 cột
- [x] train có 520841 dòng
- [x] validation có 173614 dòng
- [x] test có 173614 dòng
- [x] `class` tồn tại trong cả 3 split
- [x] `class` là integer-like
- [x] Có 178 class từ `nunique(train['class'])`
- [x] 6 meta columns tồn tại đầy đủ
- [x] `feature_cols` có đúng 3780 cột
- [x] Không có duplicate column names
- [x] Column order nhất quán giữa các split

## Class ID Policy

- `num_classes` phải lấy từ `sorted(unique_classes)` hoặc `nunique`, không dùng `max(class)+1`.
- `global_class_map.json` được xây từ train split với `178` class IDs thực.

## Missing / Inf Overview

- Số cột có `non_finite_count > 0`: `0`
- Số cột có `null_count > 0`: `0`

## Constant / Near-Constant Overview

- Số cột `constant`: `2767`
- Số cột `zero_heavy_candidate`: `232`

## Notes

- Script này là Parquet-first audit; không dùng `awk -F,` để suy luận schema CSV.
- `zero_heavy_candidate` là cờ near-constant bảo thủ, không đồng nghĩa cột vô dụng.

## Output Artifacts

- `schema_summary.json`
- `row_col_counts.csv`
- `column_positions.csv`
- `meta_columns_check.json`
- `feature_columns.json`
- `global_class_map.json`
- `class_id_summary.csv`
- `missing_inf_report.csv`
- `constant_columns.csv`

## Sample Class IDs

| class_id | global_index | count_train | present_train | count_validation | present_validation | count_test | present_test |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0 | 590 | True | 197 | True | 196 | True |
| 2 | 1 | 50 | True | 17 | True | 17 | True |
| 3 | 2 | 82 | True | 27 | True | 27 | True |
| 4 | 3 | 14 | True | 5 | True | 5 | True |
| 5 | 4 | 41 | True | 14 | True | 14 | True |
| 6 | 5 | 65314 | True | 21771 | True | 21771 | True |
| 7 | 6 | 18245 | True | 6082 | True | 6082 | True |
| 8 | 7 | 739 | True | 246 | True | 246 | True |
| 9 | 8 | 7622 | True | 2540 | True | 2541 | True |
| 10 | 9 | 23051 | True | 7683 | True | 7684 | True |
