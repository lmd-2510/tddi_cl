# Hướng dẫn các file của study T-DDI Ensemble3 × EWC

Tài liệu này giải thích ngắn gọn các file đã được thêm hoặc chỉnh sửa cho study:

```text
EWC × tddi_ensemble3 × P4 × experiment seed 0
```

Mục tiêu là giúp người mới nhìn vào repo biết file nào phụ trách việc gì mà không cần
đọc toàn bộ code.

## 1. Luồng hoạt động tổng quát

```text
Config study
  -> chạy lần lượt member 0, 1, 2
  -> mỗi member train CIL bằng EWC và lưu checkpoint
  -> mỗi task export prediction của từng member
  -> ghép ba prediction bằng mean probabilities
  -> tính uncertainty/confidence
  -> chọn threshold trên validation
  -> dùng threshold đã đóng băng để báo cáo test
```

Ba member dùng cùng dữ liệu, P4, experiment seed và class order. Điểm khác nhau giữa
chúng chỉ là member seed, vì vậy model initialization, dropout và thứ tự shuffle khác
nhau.

## 2. Các file code chính

### `src/utils/seed.py`

**Làm gì:** Tách `experiment_seed` khỏi `member_seed` và tạo member seed một cách
deterministic từ `(experiment_seed, member_id)`.

**Tác dụng:** Protocol, task order và dữ liệu vẫn giống nhau giữa ba member, nhưng ba
model có initialization/dropout/DataLoader shuffle khác nhau. Command cũ chỉ có
`--seed` vẫn giữ cách hoạt động cũ.

### `src/models/tddi_paper_member.py`

**Làm gì:** Định nghĩa một member numerical-only có kiến trúc:

```text
LayerNorm(3780) -> Linear(7560) -> activation/dropout
                -> Linear(7560) -> activation/dropout
                -> expandable head C_t
```

File cũng cung cấp `forward`, `encode`, `forward_with_latent` và metadata kiến trúc.

**Tác dụng:** Tạo backbone mới gần kích thước T-DDI paper mà không thay đổi variant
`tddi` cũ `3780 -> 1024 -> 512`.

### `src/training/ewc_checkpoint.py`

**Làm gì:** Lưu và load checkpoint EWC tại ranh giới task. Checkpoint chứa model,
Fisher, class map, task đã hoàn tất, seed/config và RNG state.

**Tác dụng:** Có thể tiếp tục member từ task kế tiếp sau khi process bị dừng. File không
lưu `theta_star` lần hai; khi resume, `theta_star` được clone lại từ model state để tiết
kiệm dung lượng.

### `src/training/train_cil.py`

**Làm gì:** Đây vẫn là entrypoint training chính. File được nối thêm các chức năng:

- CLI `--member-id` và `--resume-ewc-checkpoint`;
- variant `tddi_paper_member`;
- experiment/member seed metadata;
- mở rộng head và copy lớp cũ theo raw class ID;
- mở rộng Fisher/theta khi có lớp mới;
- log `classification_loss`, `raw_ewc_penalty`, `scaled_ewc_penalty`, `total_loss`;
- checkpoint EWC sau mỗi task;
- export validation/test predictions cho từng member.

**Tác dụng:** Kết nối model, EWC, seed, checkpoint và export thành một CIL trajectory
hoàn chỉnh. Các option mới chỉ có hiệu lực khi được bật; command study cũ vẫn dùng
đường legacy.

### `src/eval/classwise_metrics.py`

**Làm gì:** Thêm khả năng phục hồi class-wise trajectory từ checkpoint đã validate.

**Tác dụng:** Khi resume EWC, báo cáo class-wise/forgetting tiếp tục từ các task cũ thay
vì mất lịch sử hoặc tính lại sai.

### `src/eval/s02_artifacts.py`

**Làm gì:** Public hóa helper tạo stable sample ID từ cặp drug ID, vẫn giữ validation
và thứ tự dòng của S02.

**Tác dụng:** Prediction artifact của ba member dùng cùng quy tắc sample ID, giúp phát
hiện thiếu dòng, trùng dòng hoặc permutation trước ensemble.

### `src/eval/member_predictions.py`

**Làm gì:** Định nghĩa schema `.npz` cho prediction của một member và hàm export/load
có validation.

Mỗi artifact lưu run/method/protocol, task/split, member ID, hai seed, sample IDs,
labels, raw class column order, logits và probabilities.

**Tác dụng:** Tạo đầu vào độc lập, có provenance rõ ràng cho offline ensemble. Artifact
sai shape, duplicate sample ID hoặc thiếu class order sẽ bị từ chối.

### `src/eval/offline_ensemble.py`

**Làm gì:** Load đúng ba member artifact, kiểm tra chúng hoàn toàn thẳng hàng rồi lấy
trung bình probabilities.

File tính thêm predictive entropy, expected member entropy, mutual information,
normalized entropy/MI, probability variance, confidence và disagreement.

**Tác dụng:** Ensemble được thực hiện offline, không cần giữ ba model cùng lúc trên GPU.
File cố ý không average raw logits.

### `src/eval/confidence_threshold.py`

**Làm gì:** Chọn confidence threshold từ candidate grid trên validation, freeze thành
JSON, rồi load artifact đó để đánh giá test.

File báo cáo full-set accuracy/Macro-F1/ECE/NLL/Brier và high-confidence
accuracy/Macro-F1/coverage.

**Tác dụng:** Ngăn test leakage. Test split không được phép dùng để chọn threshold và
không được đánh giá bằng threshold chưa load lại từ frozen artifact.

### `src/training/tddi_ensemble3_study.py`

**Làm gì:** Orchestrator của study. Nó dựng command cho member `0 -> 1 -> 2`, phát hiện
trạng thái `fresh/resume/complete`, và chỉ gọi offline ensemble sau khi đủ ba member.

**Tác dụng:** Bảo đảm các member chạy tuần tự, mỗi member có output namespace riêng,
không ghi đè run hoàn tất, có resume theo checkpoint, kiểm tra alignment và tạo
`study_manifest.json`. Mặc định chỉ dry-run; phải thêm `--execute` mới thực thi.

## 3. Các config

### `configs/tddi_ensemble3_ewc_p4_seed0.json`

Config full study: EWC, paper-size member, P4, seed 0, ba member và tám task. Nó khai
báo đường dẫn dữ liệu, hyperparameter, output root và prediction splits.

Đây không phải file tự chạy; người dùng phải chủ động gọi orchestrator với `--execute`.

### `configs/tddi_ensemble_confidence_threshold.json`

Chứa candidate threshold grid, minimum coverage, selection rule, tie-breakers và số
bin calibration.

Tác dụng của việc để grid trong config là threshold được đặt trước, không sửa tùy ý sau
khi xem test.

### `configs/tddi_ensemble3_ewc_p4_seed0_smoke_task0.json`

Smoke Phase A dành cho máy GPU riêng: chỉ member 0, task 0, ba epoch, microbatch 64,
effective batch 1024 và accumulation 16. Output nằm trong namespace smoke riêng.

### `configs/tddi_ensemble3_ewc_p4_seed0_smoke_tasks01.json`

Smoke Phase B: member 0 chạy task 0 rồi task 1. Chỉ dùng sau khi Phase A được đánh giá
`GO`. Đây là run mới vì checkpoint khóa task-file hash và task count.

### `configs/smoke/p4_seed0_task0.json`

Bản rút gọn chính xác task 0 từ P4 seed 0, gồm 38 lớp. Nó giúp Phase A dừng sau task 0
mà không cần sửa training code.

### `configs/smoke/p4_seed0_tasks01.json`

Bản rút gọn chính xác task 0 và task 1 từ P4 seed 0. Sau task 1 có tổng cộng 58 lớp đã
thấy, dùng để kiểm tra head expansion và EWC penalty.

## 4. Các file tài liệu

### `docs/TDDI_ENSEMBLE3_EWC_PLAN.md`

Tài liệu thiết kế tổng thể: mục tiêu, kiến trúc, khó khăn, giải pháp, invariant và các
task implementation theo thứ tự.

### `docs/TDDI_ENSEMBLE3_EWC_IMPLEMENTATION_CHECKLIST.md`

Kết quả audit ban đầu và checklist file/function/acceptance criteria. File này giúp
kiểm tra mỗi giai đoạn đã làm đủ hay chưa.

### `docs/TDDI_ENSEMBLE3_RUNBOOK.md`

Hướng dẫn ngắn để dry-run, chạy/resume từng member và hiểu khi nào offline ensemble
được gọi.

### `docs/TDDI_ENSEMBLE3_REMOTE_GPU_TRAINING_GUIDE.md`

Runbook đầy đủ để chuyển repo/data sang máy khác, tính trước VRAM/RAM/disk, chạy
preflight và unit tests, chạy smoke/full bằng `nohup`, xử lý OOM, resume và thu artifact
GO/NO-GO.

### `docs/RUNBOOK.md`

Runbook chung được bổ sung phần GPU smoke chi tiết: setup môi trường, kiểm tra CUDA,
chạy Phase A/Phase B, đo peak allocated/reserved VRAM, runtime, checkpoint size, head,
Fisher, loss components, validation metrics, xử lý OOM và danh sách artifact cần gửi
lại để quyết định GO/NO-GO.

### `docs/TDDI_ENSEMBLE3_FILES_GUIDE.md`

Chính là tài liệu hiện tại: bản đồ ngắn gọn của toàn bộ file đã thay đổi.

## 5. Các file test

### `tests/test_ensemble_seed.py`

Kiểm tra seed derivation ổn định, member khác nhau có RNG khác nhau, shared protocol và
class metadata không đổi, đồng thời CLI `--seed` cũ vẫn hoạt động.

### `tests/test_tddi_paper_member.py`

Kiểm tra kiến trúc, parameter shape/count, vị trí LayerNorm, forward/latent, head
expansion và copy hàng lớp cũ.

### `tests/test_ewc_checkpoint.py`

Kiểm tra checkpoint round-trip, schema/metadata guard và continuous run so với
save/resume có cùng model/Fisher/theta/next task.

### `tests/test_tddi_paper_member_ewc.py`

Kiểm tra Focal Loss, CE Fisher, penalty bằng 0 tại `theta_star`, penalty dương khi dịch
trọng số quan trọng, new head rows chưa bị phạt và smoke CIL hai task bằng model nhỏ.

### `tests/test_member_prediction_export.py`

Kiểm tra export/load prediction, namespace, stable sample order, provenance, duplicate
ID và các lỗi row/class width.

### `tests/test_offline_ensemble.py`

Kiểm tra mean probabilities và uncertainty bằng số tính tay; kiểm tra class/sample
misalignment phải fail và ba member giống nhau cho mutual information gần 0.

### `tests/test_confidence_threshold.py`

Kiểm tra chọn threshold từ config/validation, freeze/load, chặn test leakage, config
hash và các metric report.

### `tests/test_tddi_ensemble3_study.py`

Kiểm tra dry-run không tạo output, thứ tự chạy tuần tự, resume/skip/refuse overwrite,
class alignment và offline ensemble invocation bằng synthetic artifacts nhỏ.

## 6. Những phần không bị thay đổi

- Variant `tddi` cũ vẫn là `3780 -> 1024 -> 512`.
- Config và runner study P0-P8 không được đưa ensemble mới vào.
- Baseline EWC vẫn dùng Focal Loss để train và empirical diagonal Fisher dựa trên
  cross-entropy, không thêm Fisher decay hoặc balanced Fisher.
- Không có full tám-task training hoặc GPU smoke nào được chạy tự động khi tạo các file
  này.

Nói ngắn gọn: model mới tạo một member, training/checkpoint chạy một trajectory, export
tạo dữ liệu dự đoán, offline ensemble ghép ba member, threshold xử lý confidence, và
orchestrator điều phối toàn bộ theo thứ tự an toàn.
