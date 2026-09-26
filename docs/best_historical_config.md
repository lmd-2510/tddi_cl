# Best historical anchor và cấu hình P3 được chốt để phát triển tiếp

Tài liệu này giữ lại kết quả lịch sử tốt nhất làm mốc đối chiếu, đồng thời chốt
cấu hình tiếp theo dựa trên mốc đó. Hai phần cần được phân biệt:

- **Historical anchor (bất biến):** run Focal-all với buffer theo quota lịch sử;
  không sửa lại cấu hình hay số liệu đã chạy.
- **Cấu hình được chốt để phát triển tiếp:** giữ các thiết lập historical, chỉ
  đổi sang macro/equal-class buffer và Hybrid loss (Focal cho dữ liệu task hiện
  tại, CE cho replay). Cấu hình này đã có file riêng và full-run runbook.

Mọi thay đổi khác về replay exposure, exemplar ranking, protocol, preprocessing
hoặc training phải tạo config/output namespace mới.

## Cấu hình được chốt hiện tại

- Config: `configs/p3_hybrid_equal_buffer_full8_e30_mem4.json`
- Runbook/lệnh chạy: `docs/P3_HYBRID_EQUAL_BUFFER_FULL_RUNBOOK.md`
- Output gốc trong config: `outputs/p3_hybrid_equal_buffer_full8_e30_mem4_seed0`
- Hai khác biệt so với historical anchor:
  1. Buffer: `fold_min_quota_sqrt_capacity_v1` →
     `fold_equal_class_capacity_v1` (macro/equal-class, giới hạn bởi số mẫu còn
     giữ được của từng class).
  2. Loss: Focal-all → Hybrid: Focal cho current-task samples + CE cho replay
     samples; không bật logit/feature distillation.
- Các thiết lập khác giữ theo historical setup, gồm P3 tail-to-head, 3 members,
  30 epoch tối đa/task, patience 5, replay fraction 12.5%, repeat cap 3 và
  4% budget cho mỗi member.

Đây là **cấu hình thí nghiệm được chọn**, không thay đổi sự thật rằng historical
anchor bên dưới đã chạy bằng Focal-all. Kết quả của cấu hình mới phải được báo
cáo riêng, không ghi đè số liệu historical.

## 1. Historical anchor: config và provenance

- Config: `configs/p3_hybrid_full8_e30_mem4.json`
- Output namespace: `outputs/p3_hybrid_full8_e30_mem4_seed0`
- Experiment seed: `0`
- Fold seed: `42`
- Config SHA256 trong kết quả chuẩn: `7c0faaf27a875b0a77be8a63e2794276f2ad48afd9fe63d75791f6eb13ad3531`
- Task-file SHA256: `0d64c465b0c4bd34f66e6c76088b6b73fd60839ade3e56017b5fd36c21a26e79`

Đây là kết quả tốt nhất hiện có trong các full run đã hoàn tất tại thời điểm
tài liệu được tổng hợp. “Tốt nhất” nói về **kết quả lịch sử đã quan sát**, không
phải cấu hình tiếp theo được chọn ở đầu tài liệu. Đây không phải control Hybrid
Focal-current/CE-replay hợp lệ.

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
| Config label | `loss_variant=hybrid` |
| Loss thực tế trong historical anchor | **Focal(current) + Focal(replay); không logit/feature distillation** |
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

**Lưu ý quan trọng về thực thi:** bundle kết quả ghi commit
`1f9cc1f1769fa4238460b04a0deed1b5206de390`. Ở source commit này, code chỉ tạo
`student_old_indices` khi có teacher; run `hybrid` không có teacher. Vì vậy
nhánh phân biệt current/replay không chạy và classification loss rơi về Focal
trên cả batch. Báo cáo cũ ghi `focal_current_cross_entropy_replay` dựa trên
metadata theo config, không phải bằng chứng loss đó đã chạy.

Từ commit `b3dc8ee`, code đã sửa điều kiện này. **Chạy lại cùng JSON config bằng
source hiện tại sẽ là Focal-current + CE-replay và không tái tạo chính xác kết
quả lịch sử Focal-all.** Không dùng lại output root cũ cho run mới.

## 4. Replay và exemplar của historical anchor

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

Kết quả lấy từ `docs/P3_FOCAL_ALL_MEM4.md`.

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

1. Giữ kết quả này làm historical anchor Focal-all; không diễn giải lại nó thành Hybrid.
2. Cấu hình được chốt để phát triển tiếp là Hybrid + macro/equal-class buffer ở đầu tài liệu và trong runbook liên kết.
3. Khi đánh giá tác động riêng của loss hoặc buffer, so sánh với run có cùng các yếu tố còn lại; tổ hợp hai thay đổi chỉ cho biết hiệu quả của cấu hình kết hợp, không tách được đóng góp riêng của từng thay đổi.
4. Mỗi thí nghiệm sau đó chỉ đổi một yếu tố; không trộn distillation, sampler, exemplar policy và replay exposure.

Các pilot mới phải có:

- config riêng;
- output root riêng;
- log rõ actual replay fraction, `actual_replay_draws`, `max_repeat`,
  `exemplar_coverage` và `per_class.draws`;
- so sánh validation trước khi đọc test.

Lưu ý: code hiện tại còn kiểm soát `fraction=0.125` và `repeat_cap=3` trong
sampler/contract. Vì vậy không được chỉ sửa hai con số trong JSON rồi cho rằng
replay đã thành 25%; cần truyền chúng thật sự vào `FoldReplayFractionSampler`.

## 8. Tham chiếu lệnh chạy cấu hình được chốt

Runbook dưới đây chạy cấu hình Hybrid + equal-class buffer đã chốt; đây không
phải lệnh tái lập historical anchor Focal-all. Trên server, bảo đảm fold và
preprocessing artifacts vẫn tồn tại rồi chạy:

```bash
bash scripts/run_p3_hybrid_equal_buffer_full.sh check
bash scripts/run_p3_hybrid_equal_buffer_full.sh
```

Script mặc định chạy `start`: kiểm tra đầu vào/dry-run, chạy tuần tự ba member,
offline ensemble, tổng hợp metric/visualization và gói review. Xem runbook để
theo dõi tiến trình và tìm ZIP kết quả. Không dùng lại output namespace của run
cũ; config replay 25% và các config có distillation là thí nghiệm khác.
