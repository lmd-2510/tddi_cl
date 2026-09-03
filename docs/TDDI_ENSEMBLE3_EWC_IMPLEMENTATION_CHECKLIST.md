# Audit và implementation checklist — T-DDI Ensemble ×3 + EWC

Ngày audit: 2026-08-27  
Phạm vi: `EWC × tddi_ensemble3 × P4 × experiment_seed=0`  
Trạng thái: tài liệu lịch sử của nhánh EWC; study replay-distill P3 hiện tại dùng
`docs/TDDI_PAPER_REPLAY_DISTILL_P3_8TASK_GPU_RUNBOOK.md`.

## 1. CLI, config, model, seed, optimizer và output

### Hiện trạng chính xác

- Entrypoint training là `src/training/train_cil.py::main`, lấy tham số từ `parse_args()`.
- Model được chọn bằng `--variant`; T-DDI paper-size dùng riêng
  `tddi_paper_member`. Các MLP `small/base/large` chỉ là baseline chung.
- Method được chọn bằng `--method`; `ewc` đã là một choice hợp lệ.
- Seed duy nhất hiện tại là `--seed`. `main()` gọi `set_global_seed(args.seed)` đúng một lần trước khi load task/model. Seed này đang đồng thời chi phối Python, NumPy, Torch, model initialization, dropout và DataLoader shuffle.
- Model được tạo lại ở đầu mỗi task bởi `expand_model_for_seen_classes()`.
- Optimizer được tạo lại sau model expansion ở mỗi task bằng `torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)`.
- Classification loss là `FocalLoss(gamma=args.focal_gamma)`.
- Output được quyết định bởi `--outdir`; `ensure_run_paths(..., require_empty=True)` từ chối ghi vào directory không rỗng.
- P4 seed 0 dùng task file `outputs/tasks/constrained_mass_balanced_seed0_tasks.json`.

### Invariant phải giữ

- Mỗi study mới phải có config, runner và output root riêng.
- CLI cũ chỉ truyền `--seed` phải giữ behavior cũ.
- AdamW phải tiếp tục được tạo sau head expansion để optimizer nhìn thấy toàn bộ parameter mới.

## 2. Expandable head và class map

### Hiện trạng chính xác

1. Mỗi task lấy `current_raw_classes` từ task JSON.
2. `seen_raw_classes` là hợp các class của task `0..t`, sau đó được sort.
3. `build_seen_class_map()` gọi `build_global_class_map()` và tạo `raw_class_id -> dense index` theo raw ID tăng dần.
4. Labels train/validation/test được đổi sang dense index bằng `remap_labels()`.
5. `expand_model_for_seen_classes()` tạo model mới với `num_classes=len(current_seen_map)`, copy backbone cùng shape, rồi copy từng hàng head cũ theo raw class ID.
6. Sau task, class map được lưu tại `seen_class_map_task_<task_id>.json` bằng deterministic raw-key ordering.
7. `ordered_raw_classes()` kiểm tra dense indices `0..C-1` và trả đúng thứ tự cột logits.

### Invariant phải giữ

- `current_seen_map` là nguồn sự thật duy nhất cho head width và logit-column order.
- Cùng task/protocol phải tạo class map giống nhau ở cả ba member.
- Old head rows phải giữ nguyên ngay sau expansion; new rows chỉ được khởi tạo mới.
- Prediction artifacts phải mang `class_ids` theo đúng thứ tự cột logits.
- Ensemble phải fail rõ ràng nếu class order hoặc sample order khác nhau.

## 3. EWC, Fisher, theta_star và checkpoint

### Hiện trạng chính xác

- Với `method=ewc`, training loader chỉ chứa current-task samples, natural shuffle, không có replay.
- Task 0 train bằng Focal Loss mà chưa có EWC penalty.
- Sau khi early stopping chọn best validation Macro-F1, model load lại `best_state`.
- `compute_fisher()` chạy một vòng trên current-task `train_loader`, dùng empirical diagonal Fisher từ squared gradients của cross-entropy.
- `fisher_total` được cộng không decay: `F_total <- F_total + F_new`.
- `theta_star` được clone từ best model parameters sau task.
- Trước task tiếp theo, `grow_head_state()` mở rộng Fisher/theta theo raw class ID: new Fisher head rows bằng 0; new theta rows nhận giá trị khởi tạo tương ứng.
- Trong `train_one_epoch()`, loss là `classification_loss + ewc_lambda * ewc_penalty(...)` khi Fisher/theta tồn tại.
- Checkpoint hiện tại `checkpoints/task_<t>_model.pt` chỉ chứa CPU model `state_dict` tốt nhất.
- Fisher, `theta_star`, completed task, RNG state và optimizer state không được lưu. Không có CLI resume cho EWC; process bị ngắt sẽ mất toàn bộ EWC state trong RAM.

### Invariant phải giữ

- Mỗi member phải có Fisher và `theta_star` riêng.
- Baseline mới tiếp tục dùng CE empirical Fisher, Focal classification loss và accumulation không decay.
- New head rows không được nhận EWC penalty trước khi học task mới.
- Resume tại task boundary phải tái tạo `theta_star=clone(model_state)` và giữ nguyên Fisher/class map/next task.
- Checkpoint schema mới không được làm hỏng việc load model-only checkpoint cũ.

## 4. Evaluation và export

### Hiện trạng chính xác

- `evaluate_model()` dùng inverse seen map để chuyển local prediction/label về raw class IDs.
- Validation all-seen được dùng cho early stopping; test được đánh giá theo từng old task group và trên toàn bộ seen classes.
- Evaluation lưu/khôi phục CPU và CUDA RNG state, nên bản thân evaluation không được phép làm lệch training RNG trajectory.
- Khi `--export-s02` được bật, `PredictionOutputs` chứa logits, softmax probabilities, raw predictions, raw labels, latent features và ordered raw `class_ids`.
- S02 tạo `sample_id = drugid-drug_a|drugid-drug_b`, bắt buộc unique trong shard và kiểm tra row/shape/dtype/finiteness/probability sum/softmax/logit alignment.
- S02 manifest schema v1 lưu run, seed, method, task/split, class IDs và hash checkpoint/artifact; chưa có `experiment_seed`, `member_id`, `member_seed` hoặc ensemble provenance.
- S03 chỉ fit temperature trên validation và dùng test để reporting. Đây là invariant chống leakage cần giữ.
- S04 hiện khóa cho `replay_distill_fixed_budget_uniform`; không thể dùng trực tiếp để aggregate study EWC ensemble.

### Invariant phải giữ

- Evaluation metrics cũ và schema v1 phải tiếp tục đọc được.
- Ba member phải có cùng `(task, split, sample_ids, labels, class_ids)` trước khi average probabilities.
- Không average raw logits.
- Threshold/temperature chỉ được chọn hoặc fit trên validation rồi đóng băng trước test.
- Output study mới phải nằm ngoài output root P0-P8.

## 5. Implementation checklist theo giai đoạn

### A. Seed separation

- Dự kiến sửa: `src/utils/seed.py`, `src/training/train_cil.py::parse_args/main/write_run_config`.
- Thêm `experiment_seed`, `member_id`, deterministic `member_seed`; giữ `--seed` làm backward-compatible alias/path.
- Acceptance: protocol/task/class map giống nhau giữa member; initialization/shuffle khác nhau; manifest lưu đủ provenance; test cũ không đổi.

### B. EWC checkpoint/resume

- Dự kiến thêm: module checkpoint riêng dưới `src/training/`; tích hợp tối thiểu vào `train_cil.py`.
- Lưu schema version, model state, Fisher, class map, completed task, seed/config/RNG metadata; không lưu trùng `theta_star` ở task boundary.
- Acceptance: continuous và save/resume có cùng next-task head/Fisher/theta; model-only checkpoint cũ vẫn load được; output không bị overwrite.

### C. Paper-size member

- Dự kiến thêm: `src/models/tddi_paper_member.py`; đăng ký variant mới trong CLI/model factory.
- Không sửa preset `tddi=(1024,512)` hiện tại.
- Acceptance: input LayerNorm, shapes `3780->7560->7560->C_t`, interface `forward/encode/forward_with_latent`, parameter count và head expansion đúng.

### D. EWC instrumentation

- Dự kiến sửa: `src/methods/ewc.py`, `train_cil.py::train_one_epoch` và training audit rows.
- Acceptance: log riêng classification/raw penalty/scaled penalty/total; penalty bằng 0 tại theta; new head Fisher rows bằng 0; synthetic two-task test pass.

### E. Per-member export

- Dự kiến sửa có versioning: `src/eval/s02_artifacts.py`, `train_cil.py::export_s02_evaluation`; giữ reader schema v1.
- Acceptance: artifact lưu experiment/member seeds và member ID; sample/class alignment validation giữ nguyên hoặc chặt hơn; v1 regression tests pass.

### F. Offline ensemble và UE

- Dự kiến thêm: module/entrypoint riêng dưới `src/eval/`.
- Acceptance: chỉ nhận đúng ba member khác ID và cùng provenance; mean probabilities; normalized entropy/MI an toàn với `C_t<=1`; identical members cho MI xấp xỉ 0; misalignment phải fail.

### G. Threshold/calibration

- Dự kiến thêm runner riêng hoặc mở rộng S03 theo schema mới mà không đổi behavior cũ.
- Acceptance: grid đến từ config; selection chỉ dùng validation; test reporting-only; artifact frozen có rule, coverage và provenance.

### H. Sequential orchestrator

- Đã thêm `configs/tddi_ensemble3_ewc_p4_seed0.json`,
  `src/training/tddi_ensemble3_study.py` và output root riêng; locked runner không đổi.
- Dry-run mặc định thể hiện member 0 -> 1 -> 2; `--execute --member-id N` chạy/resume
  riêng từng full-data trajectory. Offline ensemble chỉ được gọi sau khi cả ba member
  hoàn tất và pass kiểm tra alignment.
- Acceptance đã kiểm bằng synthetic integration: không có member process đồng thời,
  skip run hoàn tất, từ chối directory dở dang không có checkpoint và manifest liên
  kết đúng ba prediction artifact cho từng task/split.

### I. Remote GPU smoke config/runbook

- Dự kiến sửa tài liệu: `docs/RUNBOOK.md` và thêm smoke config có thể chuyển sang máy GPU riêng.
- Không chạy training trên máy hiện tại. Acceptance: config member 0/task 0 và lệnh task 1 có điều kiện, batch nhỏ/ít epoch, command setup/chạy/thu log rõ ràng; máy GPU riêng ghi peak VRAM, runtime, checkpoint size, head/Fisher stats và loss components; không tự chạy full experiment.

## 6. Test cần bổ sung

- Seed: derivation deterministic, backward compatibility, shared experiment invariants và distinct member RNG.
- Backbone: architecture/parameter shapes, forward/latent, old-row preservation khi expand.
- EWC: Fisher/theta expansion, zero penalty at reference, positive penalty after perturbation, per-member isolation.
- Checkpoint: schema validation, model-only compatibility, task-boundary round-trip và resume equivalence.
- Export: schema v1 regression, new provenance fields, sample/class alignment và duplicate rejection.
- Ensemble UE: hand-computed probability aggregation, entropy/MI/variance, identical-member MI, alignment failures.
- Calibration: validation-only selection, frozen threshold, test leakage guard.
- Orchestrator: dry-run ordering, output namespace, skip-complete/refuse-overwrite và synthetic end-to-end.

## 7. Baseline test trước implementation

Command:

```text
python -m pytest -q
```

Môi trường thực tế:

```text
Python executable: C:\Users\Dell\AppData\Local\Programs\Python\Python312\python.exe
Torch:             2.13.0+cpu
Pytest:            8.3.4
Workspace venv:    không có .venv/Scripts/python.exe
```

Kết quả:

```text
62 passed, 3 failed in 102.66s
```

Ba failure có sẵn đều là dependency/environment failures trong `tests/test_backbone_adapters.py`:

1. `test_tabm_member_training_and_probability_aggregation`: thiếu package `tabm`.
2. `test_tabm_expansion_preserves_old_member_heads`: thiếu package `tabm`.
3. `test_ddi_gcn_forward_and_expansion_preserve_old_heads`: thiếu package `rdkit`.

`requirements.txt` đã khai báo cả `tabm` và `rdkit`, nhưng chúng không được cài trong Python hệ thống dùng để chạy baseline. Không cài dependency và không sửa/skip test trong task audit này.

## 8. Điểm bắt đầu cho task implementation tiếp theo

Bắt đầu bằng seed separation. Trước khi sửa code, tạo môi trường đúng theo `requirements.txt` hoặc ghi rõ rằng ba optional-backbone baseline failures vẫn được chấp nhận tạm thời. Không bắt đầu paper-size model trước khi test seed và backward compatibility pass.
