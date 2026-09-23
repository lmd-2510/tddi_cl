# P3 Hybrid không distillation — full 8-task, 30 epochs, 4%/member

## 1. Phạm vi và tính toàn vẹn

Báo cáo này phân tích bundle p3_hybrid_full8_e30_mem4_summary_clean.zip và file checksum đi kèm.

- SHA256 bundle: 61220f8099aa90bf7339b50e8e764251c13c64efb96c959cfcec39257ee1310c.
- Git commit ghi trong bundle: 1f9cc1f1769fa4238460b04a0deed1b5206de390.
- Config SHA256: 7c0faaf27a875b0a77be8a63e2794276f2ad48afd9fe63d75791f6eb13ad3531.
- Task-file SHA256: 0d64c465b0c4bd34f66e6c76088b6b73fd60839ade3e56017b5fd36c21a26e79.
- Bundle có đủ member_0, member_1, member_2, offline_evaluation và manifest cho 8 task.
- Checkpoint .pt, prediction .npz và Parquet đã được loại khỏi bundle. Vì vậy report có thể được kiểm tra từ JSON/CSV, nhưng không thể tái tính prediction/UE từ đầu.

## 2. Cấu hình và method

| Nhóm | Giá trị |
|---|---|
| Protocol | P3 tail_to_head |
| Task layout | [38,20,20,20,20,20,20,20], tổng 178 class |
| Experiment/fold seed | 0 / 42 |
| Member | 3 member, chạy tuần tự |
| Backbone | tddi_paper_member, input 3780 → hidden [7560,7560] → expandable head |
| Activation/normalization | GELU, LayerNorm, dropout 0.2 |
| Method wrapper | replay_distill_fixed_budget_uniform |
| Loss variant | hybrid |
| Loss scope thực tế | focal_current_cross_entropy_replay_no_distillation |
| Epoch/patience | tối đa 30 epoch/task, patience 5 |
| Optimizer | AdamW mới ở mỗi task, lr 0.001, weight decay 0.0001 |
| Batch | microbatch 64, effective batch 1024, gradient accumulation 16 |
| Focal | focal_gamma=1.0 |
| Replay budget | 27,778 slot/member, global 83,334 slot |
| Replay sampling | fraction 0.125, repeat cap 3, rotating deterministic sampler |
| Buffer policy | fold_min_quota_sqrt_capacity_v1 |
| Exemplar ranking | raw_sample_normalized_class_mean_control_v1 |
| Preprocessing | task0_standard_frozen |
| Checkpoint selection | held-out all-seen validation Macro-F1 |
| UE | entropy confidence, raw probabilities, threshold chọn trên OOF theo balanced accuracy |

### Ý nghĩa của “không distillation”

Tên wrapper vẫn là replay_distill_fixed_budget_uniform vì đây là pipeline replay/frozen-fold dùng chung. Tuy nhiên checkpoint contract xác nhận loss_scope=focal_current_cross_entropy_replay_no_distillation, và training_audit.csv cho thấy ở mọi task:

~~~text
logit_distillation_loss   = 0
feature_distillation_loss = 0
total_loss                = classification_loss
~~~

Các trường metadata distill_alpha=1.0, feature_distill_weight=0.5 vẫn còn trong config chung, nhưng không được áp dụng khi loss_variant=hybrid.

## 3. Trạng thái hoàn thành và validation

Cả ba member đều hoàn thành task 0–7, không phải validation-only và không phải legacy checkpoint. Best validation Macro-F1 ở task 7 là:

| Member | Best epoch | Validation Macro-F1 |
|---:|---:|---:|
| 0 | 24 | 0.732339 |
| 1 | 27 | 0.728580 |
| 2 | 28 | 0.726390 |

Các giá trị gần nhau cho thấy run tương đối ổn định giữa ba member, dù đây vẫn là ba member ensemble trong cùng experiment seed, không phải ba experiment seed độc lập.

## 4. Test seen_all theo từng task

Các giá trị dưới đây là mean ± sample standard deviation qua ba member.

| Task | Accuracy | Macro-F1 | Weighted-F1 | Balanced Accuracy |
|---:|---:|---:|---:|---:|
| 0 | 0.904762 ± 0.007937 | 0.888975 ± 0.020426 | 0.897005 ± 0.003295 | 0.897222 ± 0.021145 |
| 1 | 0.880787 ± 0.026970 | 0.838977 ± 0.037225 | 0.873850 ± 0.028239 | 0.847295 ± 0.037002 |
| 2 | 0.909357 ± 0.011038 | 0.865524 ± 0.014525 | 0.905050 ± 0.011678 | 0.869395 ± 0.019291 |
| 3 | 0.919358 ± 0.010383 | 0.876056 ± 0.009925 | 0.915568 ± 0.012000 | 0.873227 ± 0.013689 |
| 4 | 0.915242 ± 0.011437 | 0.856101 ± 0.008208 | 0.912561 ± 0.011220 | 0.849726 ± 0.004802 |
| 5 | 0.907815 ± 0.001344 | 0.826636 ± 0.009637 | 0.903500 ± 0.001329 | 0.819457 ± 0.016639 |
| 6 | 0.914464 ± 0.006955 | 0.813773 ± 0.011108 | 0.912668 ± 0.007545 | 0.803171 ± 0.016719 |
| 7 | 0.861914 ± 0.002066 | 0.711944 ± 0.005959 | 0.855887 ± 0.002345 | 0.672547 ± 0.005200 |

Run giữ được Accuracy/Weighted-F1 khá cao nhờ phân bố dữ liệu lệch, nhưng Macro-F1 và Balanced Accuracy giảm dần từ task 3–6 và giảm rõ ở task 7. Vì vậy Macro-F1/Balanced Accuracy phải được ưu tiên khi đánh giá chất lượng CIL.

## 5. Kết quả cuối task 7: member và ensemble

### Member mean

| Metric | Mean ± sample std |
|---|---:|
| Accuracy | 0.861914 ± 0.002066 |
| Macro-F1 | 0.711944 ± 0.005959 |
| Weighted-F1 | 0.855887 ± 0.002345 |
| Balanced Accuracy | 0.672547 ± 0.005200 |

### Offline ensemble

| Metric | Ensemble | Gain so với member mean |
|---|---:|---:|
| Accuracy | 0.886438 | +0.024524 |
| Macro-F1 | 0.750904 | +0.038959 |
| Weighted-F1 | 0.880469 | +0.024582 |
| Balanced Accuracy | 0.712910 | +0.040363 |

Ensemble có lợi ích rõ ràng hơn ở Macro-F1 và Balanced Accuracy so với từng member trung bình. Tuy nhiên giá trị cuối vẫn thấp hơn nhiều so với task 0–4, nên ensemble làm giảm variance và cải thiện dự đoán, nhưng không tự giải quyết được forgetting.

## 6. Old class và current task ở task 7

Mean qua ba member trên test:

| Nhóm | Số mẫu/member | Accuracy | Macro-F1 | Weighted-F1 | Balanced Accuracy |
|---|---:|---:|---:|---:|---:|
| Old classes, task 0–6 | 23,756 | 0.592075 | 0.718349 | 0.714067 | 0.643095 |
| Current task, task 7 | 149,858 | 0.904690 | 0.907194 | 0.907397 | 0.905216 |

Khoảng cách current–old là khoảng 0.189 Macro-F1 và 0.262 Balanced Accuracy. Đây là proxy mạnh của old-class degradation. Không gọi đây là official forgetting score vì bundle không có class-level trajectory đầy đủ để tái dựng định nghĩa forgetting chính thức.

## 7. Task 6 → task 7 và vai trò của buffer

| Boundary | Macro-F1 | Balanced Accuracy |
|---|---:|---:|
| Task 6 | 0.813773 | 0.803171 |
| Task 7 | 0.711944 | 0.672547 |
| Giảm | **−0.101829** | **−0.130625** |

Task 7 làm giảm khoảng 10.2 điểm Macro-F1 và 13.1 điểm Balanced Accuracy so với task 6. Đây là dấu hiệu task-7 dominance và forgetting, nhưng không phải bằng chứng buffer rỗng.

Buffer audit cho thấy cả ba member đều đã bão hòa ở task 6:

| Sau task | Stored slots/member |
|---:|---:|
| 4 | khoảng 9,833–9,834 |
| 5 | khoảng 21,472–21,474 |
| 6 | 27,778 |
| 7 | 27,778 |

Training audit ghi nhận task 7 có 447 optimizer steps và khoảng 13.2–13.7 triệu examples cộng dồn theo epoch, trong khi task 6 có 47 steps và khoảng 0.62–1.01 triệu examples. Vì replay chỉ chiếm 12.5% microbatch và repeat cap là 3, buffer đầy vẫn chưa bảo đảm old class có đủ replay exposure tương đối so với dòng current khổng lồ.

Kết luận phù hợp là effective replay exposure theo class còn thấp, không phải “không có buffer”. Nếu muốn kiểm chứng, cần thêm audit số replay draw theo từng class, old/current ratio ở từng task và tỷ lệ mẫu tail thực sự xuất hiện trong batch.

## 8. UE và threshold

Tóm tắt offline ensemble trên toàn bộ test theo task boundary:

| Task | Accuracy | Macro-F1 | Balanced Accuracy | Threshold | Coverage |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.936508 | 0.940216 | 0.941667 | 0.88 | 0.833333 |
| 1 | 0.927083 | 0.897665 | 0.903284 | 0.96 | 0.621528 |
| 2 | 0.938596 | 0.914690 | 0.913598 | 0.96 | 0.707602 |
| 3 | 0.946374 | 0.913704 | 0.907303 | 0.99 | 0.528946 |
| 4 | 0.945183 | 0.901032 | 0.891544 | 0.99 | 0.531072 |
| 5 | 0.938626 | 0.883811 | 0.875761 | 0.99 | 0.506274 |
| 6 | 0.944267 | 0.870275 | 0.856048 | 0.99 | 0.470871 |
| 7 | 0.886438 | 0.750904 | 0.712910 | 0.99 | 0.473965 |

Ở task 7, ensemble trên toàn bộ test đạt:

| Metric | Full test |
|---|---:|
| Accuracy | 0.886438 |
| Macro-F1 | 0.750904 |
| Balanced Accuracy | 0.712910 |
| ECE | 0.016722 |
| AURC | 0.026288 |
| Brier score | 0.173543 |
| Negative log-likelihood | 0.604027 |

Threshold OOF được chọn là 0.99 theo entropy confidence và balanced accuracy. Trên test:

- Coverage: 0.473965 tức giữ lại 47.4% mẫu.
- Accuracy trên subset: 0.987130.
- Macro-F1 trên subset: 0.790923.
- Balanced Accuracy trên subset: 0.781838.
- target_met=false trong report threshold.

Threshold cải thiện chất lượng của nhóm mẫu rất tự tin nhưng bỏ hơn một nửa dữ liệu. Đây là selective prediction, không phải cải thiện mô hình trên toàn bộ test và không phải phương pháp sửa forgetting.

## 9. So sánh với run hybrid + distillation trước đó

So sánh này dùng report P3_HYBRID_DISTILL_E30_MEM4_RESULTS.md trong repo. Hai run cùng P3, 30 epoch, 4% buffer/member và seed; khác chính ở loss distillation.

### Member mean tại task 7

| Metric | Có distillation | Không distillation | Chênh lệch |
|---|---:|---:|---:|
| Accuracy | 0.844483 | 0.861914 | +0.017431 |
| Macro-F1 | 0.652560 | 0.711944 | **+0.059384** |
| Weighted-F1 | 0.837993 | 0.855887 | +0.017894 |
| Balanced Accuracy | 0.596411 | 0.672547 | **+0.076136** |

### Ensemble tại task 7

| Metric | Có distillation | Không distillation | Chênh lệch |
|---|---:|---:|---:|
| Accuracy | 0.871952 | 0.886438 | +0.014486 |
| Macro-F1 | 0.676011 | 0.750904 | **+0.074893** |
| Weighted-F1 | 0.865381 | 0.880469 | +0.015088 |
| Balanced Accuracy | 0.614565 | 0.712910 | **+0.098345** |

Đây là chênh lệch lớn và nhất quán, đặc biệt ở các metric coi trọng class tail. Tuy nhiên đây mới là một ablation ở cùng experiment seed, chưa đủ để khẳng định distillation luôn có hại trên mọi seed.

Các lý do hợp lý cho kết quả này:

1. Feature MSE và logit KL có thể tạo gradient cạnh tranh với việc học task mới.
2. Teacher cũ có thể đã bias theo class head; distillation truyền bias đó sang student.
3. Sai số teacher tích lũy qua 8 task.
4. Distillation giữ biểu diễn cũ nhưng replay exposure vẫn thấp so với task 7, nên không bảo vệ được old class một cách hiệu quả.
5. Khi tắt distillation, model tự do điều chỉnh representation và classifier theo toàn bộ replay/current data, giúp Macro-F1 và Balanced Accuracy tăng.

Không nên kết luận từ run này rằng distillation nói chung vô dụng. Kết luận đúng hơn là cấu hình teacher, trọng số và cách chuẩn hóa distillation hiện tại không phù hợp với distribution của study này.

## 10. Kết luận và hướng phát triển

Run hybrid không distillation là kết quả tốt hơn rõ ràng so với run có distillation ở task 7, nhất là Macro-F1 và Balanced Accuracy. Nó vẫn còn forgetting đáng kể, vì old-class metrics thấp hơn current-class metrics khoảng 19–26 điểm phần trăm.

Thứ tự thí nghiệm tiếp theo nên là:

1. Giữ run không distillation làm control chính.
2. Kiểm tra replay exposure theo class, đặc biệt ở task 7.
3. Nếu muốn đánh giá distillation công bằng, thử chỉ bật logit với distill_alpha=0.25–0.5, giữ feature bằng 0.
4. Chỉ thử feature distillation sau khi chuẩn hóa latent MSE theo dimension/scale và dùng weight nhỏ 0.05–0.1.
5. Chạy thêm experiment seed trước khi đưa kết luận cuối vào báo cáo chính.

Các run mới cần dùng output namespace riêng và giữ nguyên protocol, preprocessing, budget, epoch để ablation không bị trộn biến.
