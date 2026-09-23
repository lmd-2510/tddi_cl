# Baseline chuẩn — P3 Hybrid không distillation

Tài liệu này là **gốc bất biến** để so sánh các thí nghiệm tiếp theo. Mọi
thay đổi về replay exposure, exemplar ranking, loss, protocol hoặc preprocessing
phải tạo config/output namespace mới; không sửa ngược baseline này.

## 1. Config chuẩn

- Config: `configs/p3_hybrid_full8_e30_mem4.json`
- Output namespace: `outputs/p3_hybrid_full8_e30_mem4_seed0`
- Experiment seed: `0`
- Fold seed: `42`
- Config SHA256 trong kết quả chuẩn: `7c0faaf27a875b0a77be8a63e2794276f2ad48afd9fe63d75791f6eb13ad3531`
- Task-file SHA256: `0d64c465b0c4bd34f66e6c76088b6b73fd60839ade3e56017b5fd36c21a26e79`

Đây là control tốt nhất hiện có trong các full run đã hoàn tất. “Tốt nhất” ở
đây nghĩa là baseline có kết quả task-7 Macro-F1/Balanced Accuracy tốt nhất
để làm mốc, không có nghĩa mọi dataset hoặc mọi seed đều tối ưu.

## 2. Dữ liệu và protocol

| Thành phần | Giá trị chuẩn |
|---|---|
| Protocol | P3 `tail_to_head` |
| Task layout | `[38, 20, 20, 20, 20, 20, 20, 20]` |
| Tổng class | `178` |
| Development data | `train + validation`, `694,455` rows |
| Test | giữ riêng, không dùng fit scaler/buffer |
| Fold assignment | stratified 3-fold, seed 42 |
| Member 0/1/2 | mỗi member giữ một validation fold riêng |
| Preprocessing | `task0_standard_frozen` |
| Scaler | fit trên task 0 của member rồi đóng băng |

Mỗi member train trên hai fold development và dùng fold còn lại làm
validation. Test không tham gia train, exemplar selection hay threshold fitting.

## 3. Model và training

| Nhóm | Giá trị chuẩn |
|---|---|
| Method wrapper | `replay_distill_fixed_budget_uniform` |
| Loss variant | `hybrid` |
| Loss thực tế | focal classification + replay, **không logit distillation, không feature distillation** |
| Backbone | `tddi_paper_member` |
| Input/hidden | `3780 → [7560, 7560]` |
| Activation/norm | GELU + LayerNorm |
| Dropout | `0.2` |
| Optimizer | AdamW mới ở mỗi task |
| Learning rate | `0.001` |
| Weight decay | `0.0001` |
| Focal gamma | `1.0` |
| Epoch tối đa | `30/task` |
| Early stopping | patience `5`, chọn held-out all-seen validation Macro-F1 |
| Micro/effective batch | `64 / 1024` |
| Gradient accumulation | `16` |
| Weight alignment | `none` |

Các trường `distill_alpha`, `temperature` và `feature_distill_weight` vẫn có
trong schema dùng chung, nhưng không được áp dụng khi `loss_variant=hybrid`.

## 4. Replay và exemplar

| Thành phần | Giá trị chuẩn |
|---|---|
| Budget policy | `per_member_4_percent` |
| Budget/member | `27,778` exemplar |
| Global physical slots | `83,334` cho ba member |
| Tỷ lệ budget | `4%` của development data (`train + validation`) mỗi member |
| Buffer policy | `fold_min_quota_sqrt_capacity_v1` |
| Base quota | `10/class` nếu class có đủ mẫu |
| Exemplar ranking | `raw_sample_normalized_class_mean_control_v1` |
| Replay fraction | `0.125` |
| Repeat cap | `3` lần/exemplar/epoch tối đa |
| Replay bắt đầu | từ task `1` |
| Sampler | `stratified_fraction_v1` |

Ở mỗi task, buffer được cập nhật sau khi task hoàn tất. Class cũ có thể bị
giảm quota; class mới được chọn exemplar gần class mean; tổng buffer không vượt
`27,778/member`.

Baseline này **không được hiểu là mỗi epoch dùng toàn bộ 27,778 exemplar**.
Toàn bộ buffer được đưa vào training dataset, nhưng sampler chỉ chọn số replay
theo fraction và repeat cap.

## 5. Evaluation chuẩn

- Checkpoint: validation Macro-F1 trên toàn bộ class đã thấy.
- Báo cáo chính: task-7 `seen_all`.
- Metric ưu tiên: Macro-F1 và Balanced Accuracy.
- Báo cáo phụ: Accuracy, Weighted-F1, old/current split, forgetting trajectory.
- Ensemble: trung bình xác suất raw của ba member.
- UE/threshold: chọn trên OOF validation; không dùng test để chọn threshold.
- Prediction artifact: export cả `validation.npz` và `test.npz` cho mỗi task.

## 6. Kết quả mốc hiện tại

Kết quả lấy từ `docs/P3_HYBRID_E30_MEM4_RESULTS.md`.

### Member mean tại task 7

| Metric | Giá trị |
|---|---:|
| Accuracy | `0.861914` |
| Macro-F1 | `0.711944` |
| Weighted-F1 | `0.855887` |
| Balanced Accuracy | `0.672547` |

### Offline ensemble tại task 7

| Metric | Giá trị |
|---|---:|
| Accuracy | `0.886438` |
| Macro-F1 | `0.750904` |
| Weighted-F1 | `0.880469` |
| Balanced Accuracy | `0.712910` |

### Old/current diagnostic

| Nhóm | Macro-F1 | Balanced Accuracy |
|---|---:|---:|
| Old classes, task 0–6 | `0.718349` | `0.643095` |
| Current task 7 | `0.907194` | `0.905216` |

Task 6 → task 7 giảm khoảng `0.101829` Macro-F1 và `0.130625` Balanced
Accuracy ở member mean. Buffer đã đầy từ task 6, nên đây là mốc để kiểm tra
effective replay exposure và task-7 dominance, không phải mốc để kết luận buffer
bị rỗng.

## 7. Quy tắc tạo thí nghiệm mới

Không thay đổi nhiều nhóm cùng lúc. Thứ tự khuyến nghị:

1. Giữ nguyên baseline này làm control.
2. Chỉ thay replay exposure: pilot `fraction=0.25`, `repeat_cap=4` trên task 6→7.
3. Nếu old-class Macro-F1 tăng mà current-class không giảm mạnh, mới chạy full 8 task.
4. Sau replay pilot mới thử class-balanced current sampler hoặc exemplar policy khác.
5. Distillation chỉ được thử như một ablation riêng; không trộn vào replay pilot.

Các pilot mới phải có:

- config riêng;
- output root riêng;
- log rõ actual replay fraction, `actual_replay_draws`, `max_repeat`,
  `exemplar_coverage` và `per_class.draws`;
- so sánh validation trước khi đọc test.

Lưu ý: code hiện tại còn kiểm soát `fraction=0.125` và `repeat_cap=3` trong
sampler/contract. Vì vậy không được chỉ sửa hai con số trong JSON rồi cho rằng
replay đã thành 25%; cần truyền chúng thật sự vào `FoldReplayFractionSampler`.

## 8. Lệnh tham khảo khi chạy lại baseline

Trên server, đặt đúng fold/preprocessing root rồi chạy dry-run trước:

```bash
export P3_CONFIG="$PWD/configs/p3_hybrid_full8_e30_mem4.json"
export P3_OUT="$PWD/outputs/p3_hybrid_full8_e30_mem4_seed0"
export P3_PREP_ROOT="$PWD/study_assets/preprocessing_p3_seed0_fold42"
export P3_FOLD_ROOT="$PWD/outputs/fold_preparation_seed42_<timestamp>/folds"

bash scripts/run_p3_hyb.sh check
bash scripts/run_p3_hyb.sh dry-run
```

Chỉ chạy member khi dry-run hợp lệ và output namespace chưa chứa một run khác.
Không resume baseline bằng config replay 25% hoặc config distillation; đó là
hai thí nghiệm khác nhau.
