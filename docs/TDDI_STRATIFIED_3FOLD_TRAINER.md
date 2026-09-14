# Prompt 9 — Nối trainer cho pilot 3-fold A/B

Luồng mới **opt-in**, không thay các lệnh seeded/stratified cũ hay EWC.
Đọc [decision record](TDDI_STRATIFIED_3FOLD_DECISION_RECORD.md) trước khi chạy.
Đây là phần tích hợp trainer, **chưa phải pipeline full study hoàn chỉnh**.
Không có dataset thật hoặc job GPU nào được chạy trong lần triển khai này.

## File và vai trò

| File | Vai trò |
| --- | --- |
| `src/training/train_cil.py` | Nhận CLI mới và chuyển sang trainer opt-in; dùng lại kernel loss, head expansion và evaluation. Bổ sung số liệu distillation/optimizer vào kết quả epoch, không đổi công thức cũ. |
| `src/training/fold_pilot_training.py` | Điều phối một member: validate nguồn → load preprocessing frozen → current/validation → replay → train → chọn best validation → cập nhật buffer. Tách khỏi `main()` cũ để tránh áp policy mới lên EWC/replay cũ. |
| `tests/test_fold_pilot_training.py` | Hai task synthetic A/B, class/ID/order alignment, head copy theo raw ID, loss, tail accumulation, CLI và guard. Model test nhỏ, không cấp phát MLP production. |

Các API dùng lại, không triển khai bản thứ hai:

- `src/data/ddi_dataset.py`: validate assignment/source byte hashes và ID/label;
  đọc raw descriptors từ hai nguồn theo assignment đã lưu.
- `src/data/fold_preprocessing.py`: load artifact A/B đã chuẩn bị, không fit trong trainer.
- `src/data/fold_replay_buffer.py`: quota nền 10 + sqrt tần suất đã quan sát,
  chọn gần trung bình class trong không gian chuẩn hóa từng mẫu; chỉ giữ raw exemplar.
- `src/data/fold_replay_sampler.py`: current mỗi mẫu một lần, replay mục tiêu
  `floor(N_current/7)`, tối đa `3 × số exemplar`, luân phiên class/exemplar.
- `src/eval/evaluation.py`: đánh giá bằng head chứa **mọi class đã thấy**.
- `src/utils/seed.py`: experiment seed 0, member RNG deterministic.

## CLI và giới hạn

Bật `--fold-replay-policy stratified_fraction_v1`, kèm:

- `--member-id 0/1/2`; held-out fold tự xác định bằng member ID.
- `--fold-assignments PATH` và `--fold-manifest PATH`.
- `--fold-preprocessing PATH`: JSON frozen được tạo ở Prompt 6.
- `--preprocessing-policy raw_identity` (A) hoặc `task0_standard_frozen` (B).
- Giữ `--train`, `--validation`, `--test`, `--feature-cols`, `--task-file`, `--outdir`.
- `--validation-only`: không đọc descriptors/labels test để đánh giá, không export test.
  File test vẫn cần tồn tại để xác minh hash và kiểm tra giao drug-pair IDs đầu run.
- `--stop-after-task 1`: chỉ chạy task 0–1 nhưng vẫn dùng **file P3 tám task gốc**,
  giữ nguyên SHA256/layout `[38,20,20,20,20,20,20,20]` và 178 raw class IDs.

Không kết hợp policy mới với `--ensemble-mode stratified_3fold` cũ. Manifest mới
ghi `ensemble_mode=frozen_stratified_3fold`, không dùng thuật toán chia fold runtime.

Không truyền `--scaler`, `--total-memory-budget`, `--replay-draws-per-epoch` cũ.
Budget tự tính bằng 4% development slots chia ba, không quay lại 6.800.
Với 694.455 dòng: tổng 27.778, member 0/1/2 lần lượt 9.260/9.259/9.259.

Mặc định riêng mode mới: microbatch 64, effective target 1.024, accumulation 16.
Hiện contract pilot yêu cầu giữ bộ ba này. Không bỏ batch cuối; nhóm accumulation
cuối được chuẩn hóa theo **số mẫu thực có**, không chia cố định 1.024 khi thiếu mẫu.
Các lệnh cũ vẫn giữ default microbatch 1.024, accumulation 1.

Run mới cần namespace output **chưa tồn tại**, kể cả thư mục rỗng.
Từ Prompt 10, dùng `--resume-fold-checkpoint PATH` để tiếp tục trong cùng output
từ task boundary hợp lệ. Không xóa/ghi đè task cũ. Nếu task 0 chưa tạo checkpoint,
giữ run dở và dùng namespace mới khi muốn chạy lại từ đầu.
Xem [checkpoint/resume mới](TDDI_STRATIFIED_3FOLD_CHECKPOINT.md).

## Trình tự một task

1. Current = toàn bộ class mới trong hai training folds, theo thứ tự nguồn train
   rồi validation và chỉ số dòng nguồn. Held-out = all-seen classes trong fold còn lại.
2. Raw current + raw exemplar đang giữ được transform bằng **cùng artifact của member**.
   A giữ raw; B dùng scaler đã fit trên training task 0 và đóng băng.
3. Buffer ranking dùng raw features riêng, không dùng input đã qua scaler hoặc latent
   model. Vì vậy A/B có thể kiểm tra cùng exemplar IDs và order ở cùng task/epoch.
4. Head tăng theo raw class map. Teacher là best model task trước, frozen/eval.
   Không truy xuất lại descriptor của training task cũ đã bị loại khỏi buffer.
5. Tạo AdamW mới mỗi task; không scheduler. Early stopping theo held-out all-seen
   Macro-F1; hòa điểm thì giữ checkpoint trước. Không dùng test chọn best epoch.
6. Load best weights; báo validation all-seen/old/current với cùng head all-seen.
   Macro-F1 nhóm tính trên các class của nhóm đó, không che logits class nhóm khác.
   Nếu không bật validation-only, test chỉ được report sau khi chọn best validation.
7. Cập nhật buffer từ current + retained cũ, audit quota/IDs rồi hoàn thành task.

Loss giữ như baseline trên **mọi lượt current + replay**:

```text
total = Focal(gamma)
      + alpha × T² × KL(teacher old-class softmax || student old-class softmax)
      + feature_weight × MSE(student latent, teacher latent)
```

Task 0 chưa có teacher nên hai distillation loss bằng 0. KL chỉ lấy các cột old
classes đúng thứ tự raw ID của teacher, không lấy nhầm các cột đầu head mới.
Log ghi cả loss thô, loss đã nhân trọng số, các trọng số và temperature.

## Artifact đọc để kiểm tra

- `run_config.json`: seeds, task/assignment/source/preprocessing hashes, model/hyper,
  ranking/budget policy, implementation hashes, config hash và giới hạn thực thi.
- `stdout.log`, `train.log`, `events.csv`: tiến độ và loss mỗi epoch.
- `task_t/input_audit.json`: current/held-out/retained IDs trước task, class map,
  head size, sampler seed derivation và policy.
- `task_t/epoch_e_audit.json`: replay target/actual, coverage/repeat cap, order hashes,
  loss components, số mẫu thực học, optimizer steps, tail effective batch, runtime epoch.
- `task_t/buffer_audit.json`: quota/counts, IDs/raw labels còn giữ sau task và ranking.
- `task_t/metrics.csv`: validation (và test nếu được bật) cho all-seen/old/current.
  Nhóm rỗng được ghi `status=empty`, không gán accuracy giả bằng 0.
- `task_t/training_audit.csv`: loss/steps/runtime/validation theo epoch.
- `task_t/best_model.pt`: best weights + raw class map; **không phải resume checkpoint**.
- `task_t/completed_task.json`: best epoch, head, số slot, runtime task, checkpoint size,
  peak allocated/reserved VRAM (CPU ghi null). Chỉ xác nhận hoàn tất khi có cả
  `checkpoints/task_t.pt` hợp lệ và các artifact khớp hash.
- `checkpoints/task_t.pt`: checkpoint task-boundary mới, chứa buffer/preprocessing/
  RNG/scheduling/progress; không dùng cờ resume legacy để đọc.
- `metrics.csv`, `training_audit.csv`, `run_summary.json`, `run_summary.md`: tổng hợp
  khi hoàn thành phạm vi yêu cầu. Task 0–1 hoàn tất không được ghi là xong full tám task.

## Cập nhật Prompt 10 và các bước còn lại

- Checkpoint/resume mới **đã triển khai ở Prompt 10**, chỉ tại task boundary.
  Hai cờ resume legacy vẫn bị từ chối trong mode mới. Không hỗ trợ mid-epoch resume.
- Sau resume, báo cáo tổng hợp mới nằm ở `reports/through_task_N_<token>/`; báo cáo
  phạm vi cũ ở root được giữ nguyên. Không dùng root summary cũ để suy ra tiến độ mới.
- Prediction export có fold/provenance mới và các bước OOF/ensemble/threshold:
  theo các prompt tiếp theo. Hiện từ chối `--export-member-predictions` trong mode mới.
- Không chọn preprocessing thắng, không sửa hyper hoặc config study cũ, không chạy full.

## Kiểm thử trên máy code

```bash
python -m pytest tests/test_fold_pilot_training.py tests/test_fold_replay_sampler.py tests/test_fold_replay_buffer.py tests/test_fold_preprocessing.py tests/test_development_fold_loader.py tests/test_s04_fixed_budget.py tests/test_replay_checkpoint.py tests/test_ensemble_seed.py tests/test_stratified_ensemble_mode.py -q
```

Chỉ synthetic/unit tests. Chạy A/B thật trên server sau khi hoàn thiện các prompt
và runbook tương ứng; không sao chép cấu hình pilot cũ thiếu assignment/provenance.

Kết quả kiểm chứng ngày 2026-09-14:

- `python -m pytest tests -q`: **391 passed**, 275,65 giây; không có failure.
- `python src/training/train_cil.py --help`: thành công, có đủ cờ opt-in mới.
- `git diff --check`: thành công.
- Test mới còn kiểm tra early stopping dùng best weights cho teacher (không lấy
  epoch cuối), file nguồn bị sửa phải fail trước khi tạo output, report test chỉ
  đọc sau khi chọn best validation và EWC loss/update giữ nguyên.
- Chưa kiểm chứng runtime/VRAM hoặc độ chính xác trên dataset thật. Kết quả unit
  tests không thay cho pilot trên server GPU.
