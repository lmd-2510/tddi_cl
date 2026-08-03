# Branch: feat/diagnostic-pipeline

## 1. Thông tin chung

| Thuộc tính | Giá trị |
| --- | --- |
| Branch | `feat/diagnostic-pipeline` |
| Base branch | `main` |
| Trạng thái | In progress |
| Ngày bắt đầu | 2026-08-02 |
| Người thực hiện | |
| Tài liệu liên quan | [`../diagnostics.md`](../diagnostics.md), [`../results/s01_results.md`](../results/s01_results.md) |

## 2. Mục tiêu

Xây dựng diagnostic evaluation pipeline để:

1. đo class-wise performance và forgetting;
2. lưu predictions và latent representations;
3. đánh giá probability calibration;
4. xây fixed-budget replay-distillation baseline;
5. thực hiện các thí nghiệm E01–E03.

## 3. Phạm vi

### Trong phạm vi

- [x] **S01** Class-wise evaluation
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
| S01 | Class-wise evaluation | Done | | 20/20 full MPS runs và output validation pass |
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
| O01 | `class_trajectory.csv` | Done | 20 files, tổng 17.280 dòng |
| O02 | `class_forgetting.csv` | Done | 20 files, tổng 17.280 dòng |
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
| `src/eval/classwise_metrics.py` | Class-wise metrics, trajectory và forgetting tracker | S01 |
| `src/eval/classification_metrics.py` | Dùng explicit labels cho task-group metrics | S01 |
| `src/training/train_cil.py` | Thu thập class-wise metrics tại `test_seen_all` | S01 |
| `src/data/ddi_dataset.py` | Đếm class trên full train split từ cột label | S01 |

## 7. Quyết định kỹ thuật

### D01 — Giới hạn probability trong S01

- **Ngày:** 2026-08-02
- **Bối cảnh:** S01 cần confidence và entropy, nhưng S02–S03 mới phụ trách raw probabilities và calibration.
- **Quyết định:** Chỉ tính aggregate mean confidence và mean entropy trong memory ở S01.
- **Lý do:** Hoàn thành class trajectory mà không chồng phạm vi với S02–S03.
- **Ảnh hưởng:** Không lưu logits/probabilities và chưa tính calibration error trong S01.

### D02 — Explicit labels cho task-group evaluation

- **Ngày:** 2026-08-02
- **Bối cảnh:** Model dự đoán trên toàn bộ seen classes trong khi một task-group chỉ chứa labels của task đang được đánh giá.
- **Quyết định:** Truyền explicit local class indices vào Macro-F1, Weighted-F1 và balanced accuracy; balanced accuracy được tính bằng macro recall trên cùng label set.
- **Lý do:** Tránh đưa prediction-only classes vào mẫu số Macro-F1 và loại warning của sklearn mà không che lỗi bằng warning filter.
- **Ảnh hưởng:** `task_matrix.csv` và task-level `forgetting.csv` có semantics nhất quán theo class set của từng eval task.

## 8. Vấn đề và rủi ro

| ID | Vấn đề | Mức độ | Hướng xử lý | Trạng thái |
| --- | --- | --- | --- | --- |
| R01 | MPS không khả dụng trong agent runtime | Thấp | CPU smoke; full experiment chạy MPS trong user runtime | Resolved |

## 9. Kiểm thử và xác minh

### Automated tests

- [x] Unit tests
- [x] Integration tests
- [x] Smoke run
- [x] Full 5-seed experiment

### Lệnh kiểm tra

```bash
.venv/bin/python -m unittest discover -s tests -v

.venv/bin/python src/training/train_cil.py \
  --train train_extracted.parquet \
  --validation validation_extracted.parquet \
  --test test_extracted.parquet \
  --feature-cols outputs/audit_smoke/feature_columns.json \
  --scaler outputs/preprocess_smoke/scaler.pkl \
  --task-file outputs/tasks_smoke/random_seed0_tasks.json \
  --outdir /tmp/ddi-s01-smoke \
  --method sequential \
  --variant small \
  --batch-size 128 \
  --epochs 1 \
  --patience 1 \
  --seed 0 \
  --device cpu \
  --max-train-rows-per-task 256 \
  --max-validation-rows-per-task 256 \
  --max-test-rows-per-task 256
```

### Kết quả

| Ngày | Lệnh/Experiment | Kết quả | Output |
| --- | --- | --- | --- |
| 2026-08-02 | `unittest` | 9/9 tests pass | Class-wise và explicit-label metric tests |
| 2026-08-02 | Direct 8-task CPU smoke | Pass: 864 unique rows/file, 178 final classes | O01, O02 và existing CIL artifacts |
| 2026-08-02 | Explicit-label 8-task CPU smoke | Pass: không còn sklearn label warning | Task-group và seen-all metrics |
| 2026-08-03 | Full S01 MPS experiment | Pass: 20/20 runs, toàn bộ invariants hợp lệ | 17.280 O01 rows và 17.280 O02 rows |

## 10. Nhật ký công việc

### 2026-08-02

- Đã làm: Hoàn thành S01.1–S01.5, explicit-label metrics, unit tests và direct smoke run.
- Kết quả: O01/O02 đúng schema, full-split train count và class-wise forgetting.
- Vướng mắc: MPS không khả dụng; CPU smoke hoàn tất bình thường.
- Bước tiếp theo: S02 — lưu prediction và latent representation.

### 2026-08-03

- Đã làm: Hoàn thành 20 full MPS runs cho 4 methods × 5 seeds.
- Kết quả: Tất cả runs có `run_completed`, đúng schema và row counts; xem [`s01_results.md`](../results/s01_results.md).
- Nhận xét: Replay methods giảm forgetting mạnh so với sequential; replay-distillation giảm forgetting hơn replay nhưng final Macro-F1 thấp hơn nhẹ.
- Bước tiếp theo: S02 — lưu prediction và latent representation; sau đó E01 dùng O01/O02 để phân tích rare-class forgetting.

## 11. Điều kiện merge

- [ ] Hoàn thành phạm vi đã chọn.
- [x] Các test liên quan đều pass.
- [x] Output schema đã được kiểm tra.
- [ ] Không sử dụng test set cho calibration hoặc replay allocation.
- [ ] Memory và replay budget được audit.
- [x] Tài liệu được cập nhật.
- [x] Không commit dữ liệu hoặc artifact lớn ngoài chủ đích.

## 12. Tổng kết

- **Kết quả chính:** S01 hoàn thành; 20 full runs cung cấp class trajectory và class-wise forgetting hợp lệ.
- **Phần chưa hoàn thành:** S02–S04 và E01–E03.
- **Quyết định cho bước M1–M5:** Chưa đưa ra trước khi hoàn thành diagnostic experiments.
- **Follow-up branch/issue:** S02 — prediction và latent representation export.
