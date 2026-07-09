# Risks, Limitations, and Ablation Plan

## Objective

Liệt kê các rủi ro kỹ thuật và diễn giải có thể làm benchmark sai lệch, đồng thời xác định các ablation cần thiết để tránh overclaim.

## Major Risks

1. **Label leakage** vì `class` nằm giữa bảng.
2. **Meta text leakage** nếu vô tình đưa `drugid`, `drugname`, `drugsmiles` vào feature.
3. **CSV parse lỗi** nếu dùng shell split theo comma.
4. **Long-tail mạnh** làm `Macro-F1` biến động lớn và replay dễ lệch.
5. **Một số class quá ít mẫu** khiến validation/test cho lớp hiếm không ổn định.
6. **Pair leakage** giữa train/validation/test.
7. **Reverse pair leakage** A-B/B-A.
8. **Static split không phản ánh thời gian thật**.
9. **Random class order không mang ngữ nghĩa thời gian**.
10. **Compute/RAM bottleneck** do file lớn và nhiều task.

## Interpretation Limits

Không được viết:

```text
real temporal continual learning benchmark
```

Nếu chưa có timestamp hoặc version ordering.

Được viết:

```text
class-incremental protocol constructed from DDI2025 classes
```

Và có thể viết:

```text
long-tail emerging CIL simulates late-arriving rare DDI event classes
```

## Planned Ablations

- `MLP-small` vs `MLP-base` vs `MLP-large`
- `CrossEntropy` vs `FocalLoss`
- `StandardScaler` vs `RobustScaler`
- replay memory `20/50/100` per class
- Random-CIL vs frequency-balanced CIL vs long-tail CIL
- with/without distillation
- with/without class-balanced sampler
- optional pair-disjoint evaluation
- optional drug-incremental split

## Prioritization

### Must-Have

- replay memory size
- protocol comparison random vs long-tail
- with/without distillation

### Nice-to-Have

- focal loss
- robust scaler
- MLP width sweep

### Optional Extension

- pair-disjoint benchmark
- drug-incremental split
- EWC/SI full comparison

## Risk Mitigations

### Leakage

- explicit column exclusion
- audit script trước preprocessing
- feature schema saved as artifact

### Long-Tail

- macro metrics
- class-balanced replay
- tail-class reporting

### Split Quality

- overlap/leakage reports
- giữ split gốc nhưng nói rõ giới hạn
- pair-disjoint là ablation, không thay thế âm thầm

### Compute

- Parquet-first
- stage-wise rollout
- seed 0 sanity trước khi mở rộng 5 seeds

## What Must Be Reported Plainly

- uncertainty ở rare classes
- mọi overlap giữa split nếu có
- calibration drift nếu model quá tự tin
- gap giữa `joint_seen` và continual methods

## Acceptance Criteria

- [ ] Mọi rủi ro chính đều có mitigation cụ thể
- [ ] Ablation có thứ tự ưu tiên, không chỉ là wishlist
- [ ] Giới hạn diễn giải được ghi rõ, tránh overclaim

## Definition of Done

- [ ] File này đủ để chặn các claim quá mạnh khi viết report
- [ ] Có danh sách ablation rõ ràng để triển khai sau benchmark chính
