# Leakage and Overlap Checks

## Objective

Phát hiện sớm mọi nguồn leakage hoặc overlap có thể làm benchmark DDI2025-CIL lạc quan giả tạo.

## Threat Model

### 1. Label Leakage

- `class` nằm giữa bảng
- nếu lấy feature bằng heuristic không an toàn, `class` có thể lọt vào `X`

### 2. Meta Text Leakage

- `drugid-*`
- `drugname-*`
- `drugsmiles-*`

Các cột này không được dùng trong descriptor baseline đầu tiên.

### 3. Pair Leakage

Cùng một cặp thuốc có thể xuất hiện ở nhiều split hoặc đảo chiều A-B/B-A.

### 4. Duplicate or Conflicting Rows

- exact duplicate rows
- cùng `pair_key` nhưng nhiều `class`
- reverse pair map về cùng cặp chuẩn hóa

## Pair Key Policy

Không được mặc định DDI hoàn toàn symmetric.

Phải tạo đồng thời:

```python
ordered_pair_key = (drugid_drug_a, drugid_drug_b)
unordered_pair_key = tuple(sorted([drugid_drug_a, drugid_drug_b]))
```

Trong đó:

- `ordered_pair_key` dùng để kiểm tra overlap đúng thứ tự A->B
- `unordered_pair_key` dùng để kiểm tra reverse overlap A-B/B-A
- báo cáo phải tách riêng 2 loại overlap này

## Canonical Pair Key

```python
pair_key = tuple(sorted([drugid_drug_a, drugid_drug_b]))
```

Nếu dùng thêm tên thuốc để kiểm tra phụ thì vẫn coi `drugid-*` là khóa chính.

## Required Script

```text
scripts/check_leakage.py
```

## Planned Outputs

```text
outputs/leakage/pair_overlap_report.csv
outputs/leakage/ordered_pair_overlap_report.csv
outputs/leakage/reverse_pair_overlap_report.csv
outputs/leakage/drug_overlap_report.csv
outputs/leakage/duplicate_pair_report.csv
outputs/leakage/multilabel_pair_report.csv
outputs/leakage/leakage_summary.md
```

## Minimum Checks

- [ ] exact duplicate rows
- [ ] duplicate `ordered_pair_key` trong cùng split
- [ ] duplicate `unordered_pair_key` trong cùng split
- [ ] `ordered_pair_key` overlap train-validation
- [ ] `ordered_pair_key` overlap train-test
- [ ] `ordered_pair_key` overlap validation-test
- [ ] reverse pair A-B/B-A overlap
- [ ] cùng `unordered_pair_key` có nhiều class hay không
- [ ] drug ID overlap train/validation/test
- [ ] `class` hoặc meta columns có lọt vào `feature_cols` không

## Interpretation Rules

- nếu có pair overlap giữa train/test thì static supervised performance có thể optimistic
- protocol CIL đầu tiên **có thể giữ split gốc** để so sánh với T-DDI-style setup
- pair-disjoint split chỉ nên thêm như ablation nâng cao, không âm thầm thay split gốc
- không gọi benchmark là “real temporal continual learning” nếu chỉ đang chia class order nhân tạo

## Suggested Procedure

1. Đọc các cột:
   - `drugid-drug_a`
   - `drugid-drug_b`
   - `class`
2. Tạo `ordered_pair_key` và `unordered_pair_key`
3. Tính overlap cùng chiều và sau canonicalization
4. Kiểm tra exact duplicate toàn hàng hoặc theo subset cột chính
5. Tạo report và mức độ nghiêm trọng

## Pseudocode

```python
def build_ordered_pair_key(df):
    return list(zip(df["drugid-drug_a"], df["drugid-drug_b"]))

def build_unordered_pair_key(df):
    return df.apply(
        lambda r: tuple(sorted([r["drugid-drug_a"], r["drugid-drug_b"]])),
        axis=1,
    )

train["ordered_pair_key"] = build_ordered_pair_key(train)
train["unordered_pair_key"] = build_unordered_pair_key(train)
valid["ordered_pair_key"] = build_ordered_pair_key(valid)
valid["unordered_pair_key"] = build_unordered_pair_key(valid)
test["ordered_pair_key"] = build_ordered_pair_key(test)
test["unordered_pair_key"] = build_unordered_pair_key(test)

check_overlap(train["ordered_pair_key"], valid["ordered_pair_key"])
check_overlap(train["unordered_pair_key"], valid["unordered_pair_key"])
check_multilabel_unordered_pairs(train)
```

## Planned Command

```bash
python scripts/check_leakage.py \
  --train /mnt/data/uyen/data_splits/train_extracted.parquet \
  --validation /mnt/data/uyen/data_splits/validation_extracted.parquet \
  --test /mnt/data/uyen/data_splits/test_extracted.parquet \
  --outdir outputs/leakage
```

## Acceptance Criteria

- [ ] Có `leakage_summary.md` phân loại rủi ro theo mức độ
- [ ] Báo cáo ghi rõ cái gì là “observed” và cái gì là “future protocol option”
- [ ] Không đề xuất thay split gốc nếu chưa ghi rõ lý do
- [ ] Mọi kiểm tra leakage đều reproducible bằng script
- [ ] Báo cáo tách riêng ordered overlap và unordered/reverse overlap

## Definition of Done

- [ ] Có `scripts/check_leakage.py`
- [ ] Có đủ artifact trong `outputs/leakage/`
- [ ] Có kết luận rõ ràng về pair overlap và reverse-pair risk
