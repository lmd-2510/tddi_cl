# Hướng dẫn các file — T-DDI Ensemble3 Replay-Distill P3

## Study hiện tại

```text
method          = replay_distill_fixed_budget_uniform
backbone        = tddi_paper_member
architecture    = LayerNorm(3780) -> 7560 -> 7560 -> expandable head C_t
protocol        = P3 / tail_to_head
experiment seed = 0
member IDs      = 0, 1, 2
tasks           = 8 / 178 lớp
```

Tên `tddi` cũ đã bị loại khỏi CLI và model factory. Các variant `small`, `base` và
`large` là MLP baseline chung, không phải T-DDI. Study này chỉ chấp nhận
`tddi_paper_member`.

## Config và tài liệu để chạy

### `configs/tddi_ensemble3_replay_distill_p3_seed0.json`

Đây là nguồn cấu hình chính cho cả ba member. File khóa protocol P3, task-file, data,
backbone, optimizer, batch, replay budget, distillation, seed, checkpoint và output.
Nhờ đó ba member chỉ khác RNG riêng, còn task/class map/replay exemplar vẫn giống nhau.

### `configs/tddi_ensemble_confidence_threshold_full_p3_primary.json`

Đặt trước lưới threshold trên `entropy_confidence`. Threshold chỉ được chọn bằng
validation, yêu cầu coverage tối thiểu 0.5, sau đó được freeze để đánh giá test.

### `configs/tddi_ensemble_confidence_threshold_full_p3_low_coverage_sensitivity.json`

Phân tích phụ với coverage tối thiểu thấp hơn. Không dùng file này làm kết quả chính.

### `configs/tddi_ensemble_temperature_calibration_full_p3.json`

Cấu hình temperature scaling tùy chọn. Một scalar temperature được fit bằng validation
NLL rồi freeze; test chỉ load artifact đã freeze.

### `docs/TDDI_PAPER_REPLAY_DISTILL_P3_8TASK_GPU_RUNBOOK.md`

Runbook thực thi trên máy GPU: preflight, unit test, dry-run, nohup từng member,
resume, acceptance sau mỗi member, offline ensemble, UE audit và threshold.

### `README.md` và `docs/RUNBOOK.md`

Trang vào ngắn gọn của repo. Cả hai trỏ tới study P3 hiện tại và không còn giới thiệu
MLP 1024/512 là T-DDI.

## Model, seed và dữ liệu replay

### `src/models/tddi_paper_member.py`

Định nghĩa member numerical-only: chuẩn hóa 3780 feature bằng LayerNorm, hai hidden
layer 7560, activation/dropout và head mở rộng theo số lớp đã thấy. Khi mở head, các
row của lớp cũ được copy đúng theo raw class ID.

### `src/models/mlp.py`

Chỉ còn các baseline `small/base/large`. Alias `tddi` đã bị xóa để không thể vô tình
chạy MLP baseline dưới tên T-DDI.

### `src/utils/seed.py`

Tách hai vai trò seed. `experiment_seed=0` giữ protocol, class map, preprocessing và
replay exemplar chung; `member_seed` điều khiển model initialization, dropout và thứ
tự sampler/DataLoader riêng của member 0/1/2.

### `src/data/fixed_budget_replay.py`

Quản lý replay memory tổng cộng 6800 mẫu. Việc chọn exemplar dựa trên experiment seed,
nên ba member giữ cùng exemplar IDs/counts dù sampler order khác nhau.

### `src/training/replay_checkpoint.py`

Lưu checkpoint tại ranh giới task gồm best model, class map, replay buffer, progress,
seed/RNG và config hash. Khi resume, teacher được dựng lại từ model state; không lưu
thêm một bản teacher trùng lặp.

## Training và điều phối

### `src/training/train_cil.py`

Entrypoint huấn luyện một trajectory. File này:

- tạo `tddi_paper_member` và expandable head;
- train bằng Focal Loss;
- dùng fixed-budget replay và output/feature distillation từ task 1;
- dùng member seed cho sampler order nhưng experiment seed cho memory identity;
- lưu audit, metric, prediction artifact và replay checkpoint sau mỗi task;
- resume đúng task kế tiếp bằng `--resume-replay-checkpoint`.

CLI study chỉ nhận tên paper-size rõ ràng là `--variant tddi_paper_member`.

### `src/training/tddi_ensemble3_study.py`

Orchestrator đọc config và điều phối tuần tự member 0 → 1 → 2. Nó không tạo model ở
dry-run, không giữ hai member trên GPU cùng lúc, skip run hoàn tất, resume run dở có
checkpoint và từ chối ghi đè run dở không có checkpoint. Chỉ khi đủ ba member mới gọi
offline ensemble và ghi manifest tổng hợp.

### `src/training/train_static.py`

Entrypoint baseline tĩnh. Nó không còn có lựa chọn `tddi`; file này không tham gia
study Ensemble3 P3.

## Prediction, ensemble và uncertainty

### `src/eval/member_predictions.py`

Export/load prediction của từng member với sample ID ổn định, label, task/member ID,
seed, xác suất và raw class-column order. Loader chặn duplicate ID và shape/class-order
không hợp lệ.

### `src/eval/offline_ensemble.py`

Ghép đúng ba artifact bằng trung bình xác suất, không trung bình raw logits. File kiểm
tra alignment và xuất predictive entropy, expected entropy, MI, normalized scores,
probability variance, disagreement, entropy confidence và max probability.

### `src/eval/offline_ue_audit.py`

Đánh giá UE offline: thống kê/quantile, khả năng phát hiện dự đoán sai (AUROC/AUPRC),
risk-coverage/AURC và breakdown old/current cùng rarity dựa trên train counts. P3 được
xác thực bằng task-file `tail_to_head`; test chỉ được dùng để report.

### `src/eval/confidence_threshold.py`

Chọn threshold trên validation rồi lưu frozen artifact. Nó phân biệt rõ
`entropy_confidence` với `max_probability`, và test không được tự chọn lại threshold.

### `src/eval/offline_temperature_calibration.py`

Calibration tùy chọn trên mean probability. Fit scalar temperature ở validation,
không đổi raw artifact và bảo đảm argmax/accuracy không thay đổi khi áp dụng lên test.

## Tests chính

- `tests/test_tddi_paper_member.py`: shape, LayerNorm, latent và expandable head.
- `tests/test_ensemble_seed.py`: derivation seed và bất biến shared metadata.
- `tests/test_replay_checkpoint.py`: checkpoint/resume replay-distill.
- `tests/test_tddi_ensemble3_replay_study.py`: P3 config, dry-run, tuần tự,
  skip/resume và ensemble gate.
- `tests/test_member_prediction_export.py`: member artifact round-trip/alignment.
- `tests/test_offline_ensemble.py`: công thức ensemble/UE.
- `tests/test_offline_ue_audit.py`: error detection, selective prediction và P3 guard.
- `tests/test_confidence_threshold.py`: validation-only threshold và legacy schema.
- `tests/test_offline_temperature_calibration.py`: calibration freeze/load và bất biến
  accuracy.

Các test trên dùng tensor/data synthetic nhỏ; chúng không chạy full dataset.

## File lịch sử còn được giữ

Một số config/tài liệu EWC P4 dùng `tddi_paper_member` vẫn còn để bảo toàn khả năng đọc
checkpoint và tái lập run EWC trước đây. Chúng không dùng backbone cũ và không được
orchestrator P3 chọn khi chạy config study hiện tại.
