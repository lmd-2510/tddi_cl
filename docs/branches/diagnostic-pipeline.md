# Branch: feat/diagnostic-pipeline

## 1. Thông tin chung

| Thuộc tính | Giá trị |
| --- | --- |
| Branch | `feat/diagnostic-pipeline` |
| Base branch | `main` |
| Trạng thái | In progress |
| Ngày bắt đầu | YYYY-MM-DD |
| Người thực hiện | |
| Tài liệu liên quan | [`../diagnostics.md`](../diagnostics.md) |

## 2. Mục tiêu

Xây dựng diagnostic evaluation pipeline để:

1. đo class-wise performance và forgetting;
2. lưu predictions và latent representations;
3. đánh giá probability calibration;
4. xây fixed-budget replay-distillation baseline;
5. thực hiện các thí nghiệm E01–E03.

## 3. Phạm vi

### Trong phạm vi

- [ ] **S01** Class-wise evaluation
- [ ] **S02** Prediction và latent representation
- [ ] **S03** Calibration
- [ ] **S04** Fixed-budget replay baseline
- [ ] **E01** Rare-class forgetting
- [ ] **E02** Multi-prototype diagnostics
- [ ] **E03** Uncertainty và future forgetting

### Ngoài phạm vi

- M1–M5 trong roadmap phương pháp.
- Full rarity-aware replay.
- Full uncertainty-aware replay.
- Thay đổi dataset hoặc task protocol hiện tại.

## 4. Kế hoạch thực hiện

| ID | Công việc | Trạng thái | Commit/PR | Ghi chú |
| --- | --- | --- | --- | --- |
| S01 | Class-wise evaluation | Todo | | |
| S02 | Prediction và latent export | Todo | | |
| S03 | Calibration pipeline | Todo | | |
| S04 | Fixed-budget baseline | Todo | | |
| E01 | Rare-class forgetting | Todo | | |
| E02 | Prototype diagnostics | Todo | | |
| E03 | Uncertainty analysis | Todo | | |

Trạng thái sử dụng: `Todo`, `In progress`, `Blocked`, `Done`.

## 5. Output files

| ID | File | Trạng thái | Ghi chú |
| --- | --- | --- | --- |
| O01 | `class_trajectory.csv` | Todo | |
| O02 | `class_forgetting.csv` | Todo | |
| O03 | `predictions.parquet` | Todo | |
| O04 | `latent_features.npz` | Todo | |
| O05 | `calibration_by_task.csv` | Todo | |
| O06 | `replay_budget_audit.csv` | Todo | |
| O07 | `fixed_budget_baseline.csv` | Todo | |
| O08 | `rarity_forgetting.csv` | Todo | |
| O09 | `prototype_diagnostics.csv` | Todo | |
| O10 | `uncertainty_forgetting.csv` | Todo | |

## 6. Thay đổi mã nguồn

| File/Module | Thay đổi | Liên kết |
| --- | --- | --- |
| | | S01–S04 |

## 7. Quyết định kỹ thuật

### D01 — Tên quyết định

- **Ngày:** YYYY-MM-DD
- **Bối cảnh:**
- **Quyết định:**
- **Lý do:**
- **Ảnh hưởng:**

## 8. Vấn đề và rủi ro

| ID | Vấn đề | Mức độ | Hướng xử lý | Trạng thái |
| --- | --- | --- | --- | --- |
| R01 | | | | |

## 9. Kiểm thử và xác minh

### Automated tests

- [ ] Unit tests
- [ ] Integration tests
- [ ] Smoke run
- [ ] Reproducibility check

### Lệnh kiểm tra

```bash
# Thêm các lệnh test hoặc smoke run tại đây
```

### Kết quả

| Ngày | Lệnh/Experiment | Kết quả | Output |
| --- | --- | --- | --- |
| | | | |

## 10. Nhật ký công việc

### YYYY-MM-DD

- Đã làm:
- Kết quả:
- Vướng mắc:
- Bước tiếp theo:

## 11. Điều kiện merge

- [ ] Hoàn thành phạm vi đã chọn.
- [ ] Các test liên quan đều pass.
- [ ] Output schema đã được kiểm tra.
- [ ] Không sử dụng test set cho calibration hoặc replay allocation.
- [ ] Memory và replay budget được audit.
- [ ] Tài liệu được cập nhật.
- [ ] Không commit dữ liệu hoặc artifact lớn ngoài chủ đích.

## 12. Tổng kết

- **Kết quả chính:**
- **Phần chưa hoàn thành:**
- **Quyết định cho bước M1–M5:**
- **Follow-up branch/issue:**
