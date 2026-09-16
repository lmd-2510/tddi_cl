# Ensemble3 Replay-Distill P3 Final Report

## Experiment

- Protocol: P3 `tail_to_head`
- Method: `replay_distill_fixed_budget_uniform`
- Backbone: `tddi_paper_member`
- Experiment seed: `0`
- Ensemble members: `[0, 1, 2]`
- Member seeds: `[409845317, 215626784, 3041879697]`
- Tasks: `8`

## Average metrics

Mỗi task có trọng số bằng nhau trong bảng trung bình này.

| Metric | Mean across tasks |
| --- | --- |
| Full Accuracy | 0.912850 |
| Full Balanced Accuracy | 0.817696 |
| Full Macro-F1 | 0.842629 |
| Full Weighted F1 | 0.905433 |
| Threshold Accuracy | 0.968135 |
| Threshold Balanced Accuracy | 0.754427 |
| Threshold Macro-F1 | 0.768483 |
| Threshold Weighted F1 | 0.962828 |
| Average Coverage | 0.682720 |

## Per-task results

| task_id | full_accuracy | full_balanced_accuracy | full_macro_f1 | full_weighted_f1 | threshold_accuracy | threshold_balanced_accuracy | threshold_macro_f1 | threshold_weighted_f1 | coverage | threshold_value | mean_pairwise_disagreement | mean_member_normalized_mi |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.936508 | 0.941667 | 0.940216 | 0.932494 | 0.990476 | 0.894737 | 0.890977 | 0.986395 | 0.833333 | 0.880000 | 0.097884 | 0.079229 |
| 1 | 0.920139 | 0.908477 | 0.913513 | 0.916288 | 0.985714 | 0.758621 | 0.761155 | 0.983999 | 0.486111 | 0.950000 | 0.168981 | 0.108651 |
| 2 | 0.922515 | 0.877927 | 0.891529 | 0.914645 | 0.991848 | 0.782051 | 0.780590 | 0.988031 | 0.538012 | 0.870000 | 0.108674 | 0.078075 |
| 3 | 0.932968 | 0.877666 | 0.895975 | 0.926423 | 0.988073 | 0.858427 | 0.860434 | 0.984755 | 0.664229 | 0.790000 | 0.087345 | 0.059240 |
| 4 | 0.938670 | 0.866449 | 0.888314 | 0.935456 | 0.991600 | 0.834863 | 0.839831 | 0.990724 | 0.646133 | 0.810000 | 0.083401 | 0.056497 |
| 5 | 0.944092 | 0.864037 | 0.887213 | 0.941838 | 0.987272 | 0.793308 | 0.797765 | 0.985633 | 0.673500 | 0.820000 | 0.066841 | 0.048460 |
| 6 | 0.873969 | 0.757813 | 0.793998 | 0.866052 | 0.974492 | 0.658907 | 0.681624 | 0.970921 | 0.623800 | 0.870000 | 0.099245 | 0.054979 |
| 7 | 0.833936 | 0.447533 | 0.530275 | 0.810269 | 0.835602 | 0.454504 | 0.535488 | 0.812164 | 0.996642 | 0.590000 | 0.144666 | 0.082441 |

## Forgetting summary

- Ensemble mean forgetting trên các old task: `0.467328`.
- Member task forgetting, mean ± population std: `0.457361 ± 0.011245`.
- Member class forgetting ở task cuối, mean ± population std: `0.407519 ± 0.009904`.

### Member forgetting

| member_id | task_forgetting_mean_old_tasks | class_forgetting_mean_final | class_forgetting_median_final | zero_f1_classes_final | evaluated_classes_final |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.449479 | 0.402554 | 0.434072 | 15 | 178 |
| 1 | 0.473263 | 0.421344 | 0.465908 | 17 | 178 |
| 2 | 0.449341 | 0.398660 | 0.416637 | 15 | 178 |

### Ensemble task-group forgetting

| task_id | best_macro_f1 | final_macro_f1 | forgetting |
| --- | --- | --- | --- |
| 0 | 0.940216 | 0.491446 | 0.448769 |
| 1 | 0.965189 | 0.558879 | 0.406310 |
| 2 | 0.972697 | 0.669765 | 0.302932 |
| 3 | 0.980973 | 0.431539 | 0.549434 |
| 4 | 0.982556 | 0.540268 | 0.442288 |
| 5 | 0.988784 | 0.434903 | 0.553881 |
| 6 | 0.980236 | 0.412553 | 0.567682 |
| 7 | 0.922868 | 0.922868 | 0.000000 |
| mean_old_tasks | n.a. | n.a. | 0.467328 |

Forgetting của task là best Macro-F1 từng đạt trừ Macro-F1 tại task cuối. `mean_old_tasks` chỉ lấy các task cũ, không tính task cuối vừa học.

## Ensemble diversity

| task_id | sample_count | member_count | mean_pairwise_disagreement | mean_mutual_information | mean_member_normalized_mi | mean_total_probability_variance |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 126 | 3 | 0.097884 | 0.087042 | 0.079229 | 0.048619 |
| 1 | 288 | 3 | 0.168981 | 0.119365 | 0.108651 | 0.049863 |
| 2 | 684 | 3 | 0.108674 | 0.085775 | 0.078075 | 0.026040 |
| 3 | 1641 | 3 | 0.087345 | 0.065082 | 0.059240 | 0.018754 |
| 4 | 3685 | 3 | 0.083401 | 0.062069 | 0.056497 | 0.018343 |
| 5 | 8049 | 3 | 0.066841 | 0.053239 | 0.048460 | 0.015841 |
| 6 | 23756 | 3 | 0.099245 | 0.060401 | 0.054979 | 0.021571 |
| 7 | 173614 | 3 | 0.144666 | 0.090570 | 0.082441 | 0.043417 |

Pooled mean pairwise disagreement, có trọng số theo số mẫu: `0.134995`.

Pairwise disagreement đo tỷ lệ các cặp member dự đoán khác nhau. Mutual information và probability variance đo mức khác biệt giữa phân phối xác suất của ba member. Diversity cao hơn không tự động đồng nghĩa accuracy tốt hơn; nó phải được đọc cùng classification metrics và UE error detection.

## Metric definitions

- Balanced Accuracy là macro recall, nên mỗi class có trọng số bằng nhau.
- Weighted F1 lấy F1 từng class và đặt trọng số theo số mẫu thật của class.
- Metric `full_*` dùng toàn bộ test samples của các class đã thấy.
- Metric `threshold_*` chỉ dùng samples vượt frozen validation/OOF threshold và phải được báo cáo cùng coverage.
- Diversity được tổng hợp từ offline ensemble artifact, không chạy lại model.
