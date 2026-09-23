# P3 Hybrid + Distillation — Full 8-task, 3-member Results

## 1. Phạm vi run

- **Method:** `replay_distill_fixed_budget_uniform`
- **Loss variant:** `hybrid_distill`
- **Backbone:** `tddi_paper_member`
- **Protocol:** P3 `tail_to_head`
- **Task layout:** 8 task, tổng 178 class
- **Experiment seed:** `0`
- **Fold seed:** `42`
- **Member:** `0, 1, 2` (ba seed thành viên khác nhau, không phải ba experiment seed độc lập)
- **Epoch tối đa:** 30/task; patience 5
- **Replay budget:** 27,778 exemplar/member, tương đương khoảng 4% development data/member; tổng ba member là 83,334 slot
- **Preprocessing:** `task0_standard_frozen`
- **Exemplar ranking:** `raw_sample_normalized_class_mean_control_v1`
- **Weight alignment:** không dùng (`none`)

Run được mô tả bởi `full_manifest.json`, với `completed_tasks=[0,1,2,3,4,5,6,7]` và thứ tự member `0 -> 1 -> 2`.

## 2. Integrity và phạm vi dữ liệu đã kiểm tra

Bundle `p3_hybrid_distill_full8_e30_mem4_summary_clean.zip` đã khớp checksum đi kèm:

```text
SHA256 = 34418efd34424c1ac55bd7abed5580ded84befc38922b46824523c7534582c0f
```

Bundle có đủ artifact của cả ba member, đủ task `0–7` và đủ các report `oof_threshold_report.json`, `test_threshold_report.json`, `frozen_threshold.json`. Raw `.npz` và checkpoint `.pt` được cố ý loại khỏi archive để giảm kích thước. Vì vậy các số liệu UE/threshold dưới đây là số liệu đã được pipeline xuất ra và có thể đánh giá, nhưng không tái tính lại raw probabilities từ archive này.

## 3. Best validation Macro-F1 theo task

| Task | Member 0 (epoch) | Member 1 (epoch) | Member 2 (epoch) |
|---:|---:|---:|---:|
| 0 | 0.908564 (10) | 0.906137 (12) | 0.879004 (11) |
| 1 | 0.860100 (13) | 0.871786 (16) | 0.852974 (13) |
| 2 | 0.876947 (25) | 0.874952 (18) | 0.863179 (14) |
| 3 | 0.872334 (30) | 0.885686 (26) | 0.880137 (15) |
| 4 | 0.874574 (15) | 0.873464 (15) | 0.866000 (11) |
| 5 | 0.869192 (25) | 0.863773 (18) | 0.862547 (19) |
| 6 | 0.837083 (8) | 0.846610 (12) | 0.846991 (19) |
| 7 | 0.659967 (13) | 0.671298 (17) | 0.678249 (15) |

Validation giảm mạnh ở task 7. Đây không phải lỗi dừng sớm riêng của một member: cả ba member đều có cùng xu hướng, cho thấy task cuối/P3 tail-to-head là nút thắt chính.

## 4. Final test — seen-all tại task 7

Đây là kết quả của model sau khi học xong đủ 8 task, đánh giá trên toàn bộ class đã thấy (`seen_all`).

| Member | Accuracy | Macro-F1 | Weighted-F1 | Balanced Accuracy | Loss |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.843382 | 0.641908 | 0.836532 | 0.584110 | 0.472828 |
| 1 | 0.846418 | 0.661179 | 0.840073 | 0.606203 | 0.474164 |
| 2 | 0.843647 | 0.654593 | 0.837372 | 0.598921 | 0.477734 |
| **Mean ± sample std** | **0.844483 ± 0.001681** | **0.652560 ± 0.009795** | **0.837993 ± 0.001850** | **0.596411 ± 0.011258** | **0.474908 ± 0.002536** |

### Cách đọc

- Accuracy và weighted-F1 khá cao vì task 7 có số lượng mẫu lớn và mô hình dự đoán tốt các class nhiều mẫu.
- Macro-F1 và balanced accuracy thấp hơn nhiều, phản ánh các class ít mẫu/old class vẫn bị bỏ quên.
- Độ lệch chuẩn giữa member nhỏ, tức run khá ổn định giữa ba initialization; vấn đề chính là **bias theo class/task**, không phải variance giữa member.

## 5. Test theo từng task (trung bình ba member)

Các dòng dưới đây là `seen_all` tại từng task boundary, không phải trung bình qua các task.

| Eval task | Accuracy | Macro-F1 | Weighted-F1 | Balanced Accuracy |
|---:|---:|---:|---:|---:|
| 0 | 0.904762 | 0.888975 | 0.897005 | 0.897222 |
| 1 | 0.863426 | 0.834755 | 0.854245 | 0.831999 |
| 2 | 0.893275 | 0.845818 | 0.883403 | 0.837938 |
| 3 | 0.918749 | 0.859957 | 0.912706 | 0.846136 |
| 4 | 0.922298 | 0.866192 | 0.918686 | 0.849219 |
| 5 | 0.936762 | 0.858073 | 0.934040 | 0.841309 |
| 6 | 0.928986 | 0.837572 | 0.927253 | 0.820233 |
| 7 | 0.844483 | 0.652560 | 0.837993 | 0.596411 |

Metric tốt đến task 6 nhưng giảm rõ ở task 7. Vì vậy không nên báo cáo một “mean qua 8 task” thay cho kết quả cuối; kết quả chính của CIL vẫn là dòng task 7 `seen_all`, kèm trajectory/forgetting.

## 6. Old classes so với current task ở task 7

Trung bình ba member:

| Nhóm class | Accuracy | Macro-F1 | Weighted-F1 | Balanced Accuracy | Số mẫu xấp xỉ |
|---|---:|---:|---:|---:|---:|
| Old classes (task 0–6) | 0.571098 | 0.649401 | 0.700357 | 0.559791 | 23,756/member |
| Current task (task 7) | 0.887820 | 0.888818 | 0.890453 | 0.885714 | 149,858/member |

Khoảng cách Macro-F1 current–old khoảng **0.239** và balanced accuracy khoảng **0.326**. Đây là bằng chứng rõ rằng model đang ưu tiên dữ liệu task 7; có thể gọi là old-class degradation/forgetting proxy. Không gọi đây là official forgetting score vì bundle không có đầy đủ `class_trajectory.csv`/`class_forgetting.csv` để tái dựng định nghĩa chính thức.

## 7. Kiểm tra hai loss distillation

- Task 0 không có teacher nên `logit_distillation_loss` và `feature_distillation_loss` bằng 0 là đúng.
- Từ task 1 trở đi, hai loss có giá trị khác 0, tức distillation thực sự được bật.
- Ví dụ member 0, task 7, epoch 18: classification loss `0.091930`, raw logit loss `0.174468`, raw feature loss `0.103818`, feature loss sau weight `0.051909`, total loss `0.318307`.
- Ở task 1, feature loss có lúc khoảng `3.8–6.4` (sau weight khoảng `1.9–3.2`), lớn hơn classification loss. Đây là dấu hiệu cần kiểm tra scale của feature MSE; không thể kết luận distillation có lợi chỉ từ việc loss khác 0.

Kết luận kỹ thuật: loss distillation đang chạy đúng, nhưng cấu hình hiện tại có khả năng để feature term chi phối total loss ở một số task. Muốn kết luận hiệu quả phải so sánh với run cùng seed/config nhưng tắt logit + feature distillation.

## 8. Offline ensemble và UE/threshold (các task có trong bundle)

`confidence_score=entropy_confidence`, probability source là `raw`, threshold được chọn trên OOF validation theo balanced accuracy với minimum coverage 0.5.

| Task | Test Accuracy | Test Macro-F1 | Test Balanced Acc. | Threshold | Selective coverage | Selective Macro-F1 | Selective Bal. Acc. |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.936508 | 0.940216 | 0.941667 | 0.88 | 0.833333 | 0.890977 | 0.894737 |
| 1 | 0.920139 | 0.913063 | 0.908477 | 0.94 | 0.513889 | 0.778397 | 0.775862 |
| 2 | 0.926901 | 0.896727 | 0.888629 | 0.87 | 0.557018 | 0.806624 | 0.807692 |
| 3 | 0.943327 | 0.910561 | 0.895237 | 0.91 | 0.440585 | 0.785619 | 0.785714 |
| 4 | 0.939484 | 0.899018 | 0.882164 | 0.88 | 0.515061 | 0.848506 | 0.846328 |
| 5 | 0.951174 | 0.893792 | 0.875890 | 0.86 | 0.653622 | 0.887238 | 0.882934 |
| 6 | 0.948055 | 0.875711 | 0.858273 | 0.94 | 0.470365 | 0.739837 | 0.733465 |
| 7 | 0.871952 | 0.676011 | 0.614565 | 0.87 | 0.703474 | 0.610153 | 0.576090 |

Threshold giúp tăng accuracy trên subset tự tin, nhưng không đảm bảo Macro-F1/balanced accuracy tăng. Ở task 7, selective Macro-F1 và balanced accuracy còn thấp hơn full test; model đang giữ lại chủ yếu các dự đoán của class/head dễ, loại bỏ mẫu khó và class tail. Vì vậy threshold không phải cách sửa forgetting.

### Ensemble gain tại task 7

So với trung bình ba member đơn lẻ:

| Metric | Member mean | Ensemble | Gain |
|---|---:|---:|---:|
| Accuracy | 0.844483 | 0.871952 | +0.027469 |
| Macro-F1 | 0.652560 | 0.676011 | +0.023451 |
| Weighted-F1 | 0.837993 | 0.865381 | +0.027389 |
| Balanced Accuracy | 0.596411 | 0.614565 | +0.018154 |

Ensemble có lợi ích đo được, nhưng gain vừa phải và chưa giải quyết được old-class forgetting.

## 9. Kết luận

1. Full run đã hoàn tất đủ 8 task cho cả ba member theo manifest.
2. Run ổn định giữa member (std nhỏ), nhưng chất lượng cuối bị giới hạn bởi task 7 và mất cân bằng old/current.
3. Distillation không bị “tắt”; hai loss hoạt động từ task 1. Tuy nhiên feature term có lúc lớn, cần ablation trước khi khẳng định là có ích.
4. Ensemble cải thiện task-7 Macro-F1 khoảng 2.35 điểm phần trăm và balanced accuracy khoảng 1.82 điểm phần trăm so với member mean, nhưng chưa đủ để xử lý forgetting.
5. UE/threshold hiện phù hợp để phân tích selective prediction, không nên dùng để che thay đổi phân bố class hoặc thay thế replay/forgetting mitigation.

## 10. Việc nên làm tiếp theo

- Bundle summary hiện tại đã đủ để phân tích training, forgetting và các report UE đã xuất. Chỉ cần gửi thêm `.npz` nếu muốn tái tính UE độc lập.
- Chạy ablation cùng preprocessing, protocol, budget và seed: (a) hybrid + distillation; (b) hybrid không distillation. So sánh task-7 Macro-F1, balanced accuracy, old/current metrics và forgetting.
- Nếu feature loss tiếp tục chi phối total loss, thử chuẩn hóa feature MSE theo dimension/activation hoặc giảm `feature_distill_weight`; chỉ chọn sau khi xem validation, không chọn trên test.
- Báo cáo chính nên dùng task-7 `seen_all` + old/current + forgetting trajectory; ghi rõ đây là một experiment seed với ba ensemble members, không phải mean ± std qua nhiều experiment seed độc lập.

## 11. Cấu hình và method đầy đủ để phát triển tiếp

### 11.1 Pipeline thực tế

Mỗi member dùng một fold validation khác nhau nhưng cùng assignment/manifest đóng băng. Dữ liệu được đưa qua preprocessing `task0_standard_frozen`, sau đó học tuần tự theo layout `[38, 20, 20, 20, 20, 20, 20, 20]` (tổng 178 class, protocol `tail_to_head`). Sau mỗi task, model mở rộng classifier head, đánh giá trên validation/test của toàn bộ class đã thấy, lưu checkpoint và prediction artifact. Cuối cùng ba member được gộp offline bằng xác suất raw để tạo ensemble và chọn threshold từ OOF validation.

### 11.2 Các tham số của run hiện tại

| Nhóm | Giá trị hiện tại | Ý nghĩa khi điều chỉnh |
|---|---|---|
| Method/loss | `replay_distill_fixed_budget_uniform`, `hybrid_distill` | Replay mẫu cũ, focal classification và distillation trên cột class cũ/latent |
| Backbone | `tddi_paper_member` | Paper-size member; không phải toàn bộ training recipe của paper T-DDI |
| Epoch/patience | 30/task, patience 5 | Tối đa số vòng học mỗi task; dừng sớm theo validation Macro-F1 |
| Optimizer | AdamW mới ở mỗi task, lr `1e-3`, weight decay `1e-4`, scheduler null | Tăng lr có thể học task mới nhanh hơn nhưng dễ quên; weight decay lớn hơn làm regularization mạnh hơn |
| Batch | microbatch 64, effective batch 1024, gradient accumulation 16 | Batch hiệu dụng dùng cho cập nhật; không đồng nghĩa mỗi epoch thấy cùng số mẫu giữa các task |
| Backbone details | input 3780 → hidden `[7560, 7560]` → expandable linear head; GELU, LayerNorm, dropout 0.2 | Không đổi nếu muốn so sánh công bằng với run này |
| Classification | focal current cross-entropy, `focal_gamma=1`, weight 1.0 | Gamma cao hơn tập trung mẫu khó nhưng có thể làm tail/noisy sample chi phối |
| Logit distillation | old-column KL, `distill_alpha=1`, temperature `T=2` | Giữ phân bố logit cũ; giảm alpha nếu teacher không ổn định |
| Feature distillation | latent MSE, weight `0.5` | Giữ biểu diễn cũ; nên chuẩn hóa theo dimension/scale trước khi tăng weight |
| Memory | `fold_member_budget=27,778` cho mỗi member; global 83,334; replay fraction 0.125; repeat cap 3 | Budget đã là 4% development data/member trong run này; thay đổi phải ghi namespace mới |
| Buffer policy | `fold_min_quota_sqrt_capacity_v1`, ranking `raw_sample_normalized_class_mean_control_v1` | Quota tối thiểu + phân bổ theo căn bậc hai tần suất; không phải quota bằng nhau tuyệt đối |
| Sampler/RNG | `fold_fraction_capped_rotating_v1`, PCG64 SeedSequence theo member/task/epoch | Quyết định mẫu replay và tính tái lập |
| Preprocessing | `task0_standard_frozen` | Fit scaler ở task 0 theo fold của member rồi đóng băng; không fit lại trên task tương lai |
| Evaluation | early stopping `held_out_all_seen_macro_f1`; threshold OOF tối ưu balanced accuracy, minimum coverage 0.5 | Không dùng test để chọn checkpoint/threshold |

Về mặt khái niệm, loss từ task 1 trở đi có dạng:

```text
L = L_focal(current + replay)
    + 1.0 * KL_T=2(student_old_logits, frozen_teacher_old_logits)
    + 0.5 * MSE(student_latent, frozen_teacher_latent)
```

Teacher là checkpoint task trước và không được cập nhật trong task hiện tại. Task 0 không có teacher nên hai thành phần distillation bằng 0 là đúng. Tên `loss_scope` trong checkpoint contract là `focal_current_cross_entropy_replay_plus_old_column_KL_T2_and_latent_MSE`.

### 11.3 Thứ tự điều chỉnh nên dùng

Để biết nguyên nhân, mỗi lần chỉ thay một nhóm:

1. Chạy control **hybrid không distillation**, giữ nguyên fold, seed, budget, sampler và epoch.
2. Nếu control tốt hơn, bật lại **chỉ logit distillation** với alpha `0.25–0.5`; giữ feature bằng 0.
3. Chỉ sau đó thử feature với weight `0.05–0.1`, đồng thời chuẩn hóa MSE theo số chiều/độ lệch chuẩn latent.
4. Cuối cùng mới thay replay fraction/quota. Luôn xem thêm old/current Macro-F1 và Balanced Accuracy, không chỉ Accuracy.

Không nên đồng thời đổi protocol, scaler, buffer, loss và epoch vì khi đó không thể biết cải thiện đến từ đâu.

## 12. Task 6 → task 7: điều gì đang xảy ra?

Mean qua ba member trên test `seen_all`:

| Boundary | Macro-F1 | Balanced Accuracy |
|---|---:|---:|
| Task 6 | 0.837572 | 0.820233 |
| Task 7 | 0.652560 | 0.596411 |
| Giảm | **−0.185012** | **−0.223822** |

Đây là giảm khoảng 18.5 điểm Macro-F1 và 22.4 điểm balanced accuracy, không phải dao động nhỏ. Ở task 7, old classes chỉ đạt Macro-F1 `0.649401`/balanced accuracy `0.559791`, trong khi current task đạt `0.888818`/`0.885714`. Khoảng cách này cho thấy model ưu tiên task 7 và old-class degradation rất mạnh.

Không nên kết luận “buffer bị thiếu” theo nghĩa buffer rỗng. Audit cho thấy số slot đã bão hòa từ task 6:

| Sau task | Stored slots |
|---:|---:|
| 4 | 9,833 |
| 5 | 21,472 |
| 6 | 27,778 (đầy) |
| 7 | 27,778 (đầy) |

Tuy nhiên **effective replay exposure** vẫn có thể thiếu: task 7 có dòng dữ liệu hiện tại rất lớn, audit ghi nhận khoảng 447 optimizer steps/456,709 examples so với khoảng 47 steps/47,859 examples ở task 6; replay chỉ chiếm 12.5% microbatch và repeat cap là 3. Vì vậy cùng một buffer đầy vẫn không bảo đảm old class được nhìn đủ thường xuyên. Quota căn bậc hai cũng có thể dành nhiều slot hơn cho class có tần suất cao, trong khi mục tiêu Macro-F1 cần bảo vệ class tail.

Diễn giải hợp lý nhất là **task-7 dominance + replay exposure theo class chưa đủ**, có thể cộng thêm xung đột distillation và thay đổi phân bố. Đây là bằng chứng cần đo replay exposure, không phải bằng chứng rằng chỉ cần tăng số slot là chắc chắn giải quyết được.

## 13. Vì sao bỏ distillation có thể cao hơn bật distillation?

Bundle hiện tại chỉ chứa run có distillation; vì vậy chưa thể khẳng định no-distill đã thắng. Nếu control no-distill cao hơn, các nguyên nhân có thể kiểm chứng là:

- **Gradient interference:** ở task 1, feature MSE sau weight có thể lớn hơn classification loss; gradient giữ latent cũ cạnh tranh với gradient học class mới.
- **Teacher chưa đủ tốt:** teacher task 0 chỉ học 38 class và chịu imbalance; KL buộc student bắt chước cả các logit/độ tự tin sai của teacher.
- **Lỗi tích lũy:** teacher của task sau là checkpoint của task trước, nên bias và calibration không tốt truyền qua nhiều task.
- **Scale chưa cân bằng:** logit KL và latent MSE chưa được chuẩn hóa theo độ lớn logit/feature hoặc số chiều latent; trọng số `1.0/0.5` không có nghĩa là hai gradient có cùng ảnh hưởng.
- **Task 7 quá lớn:** khi dòng current áp đảo, distillation bảo vệ một teacher cũ không hoàn hảo nhưng vẫn không tạo đủ old-class exposure; kết quả là vừa hạn chế học task mới, vừa không cứu được old class.

Do đó no-distill cao hơn (nếu xảy ra) không chứng minh distillation vô dụng nói chung; nó chỉ cho thấy cấu hình teacher/weight/normalization hiện tại không phù hợp với dữ liệu này. Ablation phải giữ nguyên mọi thứ khác và so sánh: task-7 Macro-F1, balanced accuracy, old/current metrics, per-task trajectory, loss components và replay exposure.

## 14. Kết luận và hướng phát triển

Run hiện tại là một baseline có kiểm soát tốt: ba member, cùng protocol P3 tail-to-head, mỗi member budget 4%, đủ 8 task và có UE/ensemble. Điểm nghẽn chính không phải thiếu checkpoint mà là suy giảm old class khi task 7 có lượng dữ liệu rất lớn. Bước tiếp theo hợp lý là chạy control hybrid không distillation; sau đó giảm riêng logit alpha, rồi feature weight/normalization nếu cần. Chỉ khi một ablation cải thiện ổn định trên validation và task-7 old/current metrics mới đưa vào full ensemble run. Mọi kết quả mới phải dùng output namespace riêng và ghi rõ config SHA để tránh trộn với run hiện tại.
