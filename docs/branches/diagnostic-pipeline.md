# Branch: feat/diagnostic-pipeline

## 1. Thông tin chung

| Thuộc tính | Giá trị |
| --- | --- |
| Branch | `feat/diagnostic-pipeline` |
| Base branch | `main` |
| Trạng thái | In progress |
| Ngày bắt đầu | 2026-08-02 |
| Người thực hiện | |
| Tài liệu liên quan | [`../diagnostics.md`](../diagnostics.md), [`../results/s01_results.md`](../results/s01_results.md), [`running_guildes/s02_guilde.md`](running_guildes/s02_guilde.md) |

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
- [x] **S02** Prediction và latent representation
- [x] **S03** Calibration
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
| S02 | Prediction và latent export | Done | | Schema v1; validation/test shards; A/B smoke pass |
| S03 | Calibration pipeline | Done | | 20 runs × 8 tasks; validation-only fit; 640 O05 rows |
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
| O03 | `s02/task_<t>/<split>/predictions.parquet` | Done | Explicit PyArrow schema, ZSTD |
| O04 | `s02/task_<t>/<split>/latent_features.npz` | Done | Compressed arrays, `allow_pickle=False` |
| O05 | `calibration_by_task.csv` | Done | 640 rows; raw/scaled × validation/test |
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
| `src/utils/logging.py` | Run ID, fail-fast cho output directory và provenance paths | S01 hardening |
| `tests/test_s01_hardening.py` | Regression tests cho class alignment, sampler và provenance | S01 hardening |
| `src/eval/cil_evaluation.py` | Metrics và optional raw prediction/latent collection | S02 |
| `src/eval/s02_artifacts.py` | Schema, invariants, atomic shard export và manifest | S02 |
| `src/models/mlp.py` | `encode`, `classify`, `forward_with_latent` tương thích checkpoint | S02 |
| `src/training/train_cil.py` | CLI và orchestration ngắn cho validation/test export | S02 |
| `tests/test_s02_prediction_export.py` | API, alignment, round-trip và negative cases | S02 |
| `src/eval/calibration_metrics.py` | ECE, Brier, NLL, confidence và safeguarded-Newton temperature scaling | S03 |
| `src/eval/run_s03_calibration.py` | Audit S02 provenance, validation-only fit và tổng hợp O05 | S03 |
| `tests/test_s03_calibration.py` | Raw-label mapping, metric, optimizer, leakage và output regression tests | S03 |

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

### D03 — Dynamic class map được giữ lại và khóa bằng test

- **Ngày:** 2026-08-03
- **Bối cảnh:** Raw class index thay đổi khi thêm class có ID nhỏ hơn old classes.
- **Quyết định:** Giữ dynamic dense map; remap từng classifier row bằng raw class ID và align student logits theo đúng teacher raw-class order.
- **Xác minh:** Regression test buộc old logits trước/sau head expansion giống nhau chính xác và từ chối metadata teacher không hợp lệ.

### D04 — Định danh chính xác replay protocol

- **Ngày:** 2026-08-03
- **Quyết định:** Giữ CLI aliases `replay`/`replay_distill` để tương thích, nhưng manifest ghi `replay[_distill]_balanced_per_class_cap50`.
- **Ảnh hưởng:** Kết quả legacy so sánh các packaged methods, không cô lập riêng đóng góp của replay memory; fixed-budget baseline vẫn thuộc S04.

### D05 — Một output directory chỉ chứa một run

- **Ngày:** 2026-08-03
- **Quyết định:** `train_cil.py` fail-fast nếu `--outdir` đã có nội dung; mỗi run có `run_id`, `run_config.json` và cùng `run_id` trên mọi event.
- **Ảnh hưởng:** Không còn append log/checkpoint của execution mới vào run cũ.

### D06 — Validation và training audit

- **Ngày:** 2026-08-03
- **Quyết định:** Ghi rõ policy `all_seen_classes_for_early_stopping`; lưu sampler, dataset size, memory trước/sau task, expected replay draws và optimizer steps trong `training_audit.csv`.
- **Ảnh hưởng:** Audit là mô tả protocol hiện tại; actual replay draw IDs và fixed replay exposure được hoàn thiện trong S04.

### D07 — S02 cô lập theo run/task/split

- **Ngày:** 2026-08-03
- **Quyết định:** Exporter sở hữu `<run>/s02`; mỗi task/split là một atomic shard gồm O03/O04, còn schema version và inventory nằm trong một manifest cấp run.
- **Lý do:** Hạn chế peak RAM, cô lập lỗi và tránh làm `train_cil.py` hoặc cấu trúc output phình thêm tầng không cần thiết.
- **Ảnh hưởng:** Không overwrite shard, không backfill legacy S01 directories; clean S04 runs sẽ bật S02 trong cùng execution.

### D08 — Evaluation không được làm thay đổi training RNG

- **Ngày:** 2026-08-03
- **Bối cảnh:** Smoke A/B đầu tiên cho thấy pass validation export bổ sung làm DataLoader tiến RNG, từ đó đổi weighted replay sampling ở task sau.
- **Quyết định:** Evaluator khôi phục PyTorch CPU/CUDA RNG state sau mọi evaluation pass.
- **Xác minh:** Regression test RNG và hai smoke 8-task cho O01/O02/task metrics giống byte-for-byte.

### D09 — Temperature chỉ fit trên validation theo từng run/task

- **Ngày:** 2026-08-07
- **Quyết định:** Fit một scalar temperature cho mỗi `(run_id, train_task)` bằng NLL trên validation; test chỉ nhận temperature đã khóa để báo cáo.
- **Xác minh:** Runner tách đường dữ liệu validation/test, manifest ghi `temperature_fit_split=validation`, unit test kiểm tra input của fitter và full audit xác nhận 640/640 hàng cùng policy.
- **Ảnh hưởng:** Không thay đổi prediction/accuracy; output có thể để trống high-confidence error rate khi scaled confidence không còn mẫu đạt ngưỡng 0.9.

## 8. Vấn đề và rủi ro

| ID | Vấn đề | Mức độ | Hướng xử lý | Trạng thái |
| --- | --- | --- | --- | --- |
| R01 | MPS không khả dụng trong agent runtime | Thấp | CPU smoke; full experiment chạy MPS trong user runtime | Resolved |
| R02 | Một số legacy run directories chứa log của nhiều execution | Trung bình | Giữ kết quả dưới nhãn legacy; run mới fail-fast và có run ID | Mitigated |
| R03 | Extra inference pass làm đổi replay sampling RNG | Cao | Evaluation RNG-neutral và A/B regression smoke | Resolved |

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
| 2026-08-03 | S01 hardening unit tests | 15/15 tests pass | Alignment, sampler, fail-fast và run ID |
| 2026-08-03 | 8-task replay-distill CPU smoke | Pass | `run_config.json`, 8-row `training_audit.csv`, đúng một run lifecycle |
| 2026-08-03 | S02 full unit suite | Pass: 24/24 | 15 tests cũ/S01 và 9 tests S02 |
| 2026-08-03 | S02 A/B 8-task CPU smoke | Pass: byte-identical O01/O02/task metrics | Export off so với validation+test export |
| 2026-08-03 | S02 artifact audit | Pass: 32 data files, 16 manifest entries | 2.3 MiB; peak shard 192 KiB; latent dim 256 |
| 2026-08-07 | S03 unit/integration suite | Pass: 29/29 tests | 24 tests S01–S02 và 5 tests S03 |
| 2026-08-07 | Full S03 calibration | Pass: 20 runs, 160 fits, 640 rows | O05 schema v1; validation-only fit; test reporting only |
| 2026-08-07 | S03 output audit | Pass: 640 unique keys, 160/160 converged | Accuracy invariant; validation NLL không tăng; không hit temperature bounds |

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

- Đã làm thêm: Khóa S01 về class alignment, provenance và protocol audit.
- Kết quả: Dynamic remap/distillation alignment pass; run mới không thể ghi vào directory cũ; sampler, memory và optimizer steps được audit.
- Giới hạn còn lại: Full S01 artifacts hiện tại vẫn là legacy balanced/per-class-cap runs; chưa chạy lại fixed-budget baseline.

- Đã làm thêm: Hoàn thành S02 schema v1, prediction/latent evaluator, atomic task/split exporter và CLI opt-in.
- Kết quả: 24/24 tests pass; smoke export tạo 16 O03 + 16 O04 và không làm đổi O01/O02/task metrics.
- Giới hạn còn lại: Không backfill 20 legacy S01 runs; test artifacts chỉ dành cho reporting, S03/S04 decisions phải dùng validation.

### 2026-08-07

- Đã làm: Hoàn thành S03 metrics, scalar temperature scaling và full aggregation cho 20 clean S02 runs.
- Kết quả: 160/160 fits hội tụ; final-task test ECE giảm rõ trên replay, replay-distill và sequential; xem [`s03_results.md`](../results/s03_results.md).
- Giới hạn: Temperature scaling tối ưu NLL nên ECE không giảm ở mọi seed; high-confidence error rate không xác định khi không còn mẫu đạt confidence 0.9.
- Bước tiếp theo: S04 fixed-total-memory/fixed-replay-exposure baseline hoặc E01 trên O01/O02.

## 11. Điều kiện merge

- [ ] Hoàn thành phạm vi đã chọn.
- [x] Các test liên quan đều pass.
- [x] Output schema đã được kiểm tra.
- [x] Không sử dụng test set cho calibration hoặc replay allocation.
- [ ] Memory và replay budget được audit.
- [x] Tài liệu được cập nhật.
- [x] Không commit dữ liệu hoặc artifact lớn ngoài chủ đích.

## 12. Tổng kết

- **Kết quả chính:** S01–S03 đã hoàn thành; prediction/latent artifacts và calibration-by-task đều có full 5-seed outputs cùng provenance audit.
- **Phần chưa hoàn thành:** S04 và E01–E03.
- **Quyết định cho bước M1–M5:** Chưa đưa ra trước khi hoàn thành diagnostic experiments.
- **Follow-up branch/issue:** Clean S04 fixed-budget runs bật `--export-s02`, hoặc E01 rare-class forgetting.
