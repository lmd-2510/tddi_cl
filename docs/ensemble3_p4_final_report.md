# Ensemble3 Replay-Distill P4 Final Report

## Experiment

- Protocol: P4 `constrained_mass_balanced`
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
| Full Accuracy | 0.433563 |
| Full Balanced Accuracy | 0.628608 |
| Full Macro-F1 | 0.605322 |
| Full Weighted F1 | 0.383374 |
| Threshold Accuracy | 0.532050 |
| Threshold Balanced Accuracy | 0.577227 |
| Threshold Macro-F1 | 0.581706 |
| Threshold Weighted F1 | 0.446864 |
| Average Coverage | 0.562811 |

## Per-task results

| task_id | full_accuracy | full_balanced_accuracy | full_macro_f1 | full_weighted_f1 | threshold_accuracy | threshold_balanced_accuracy | threshold_macro_f1 | threshold_weighted_f1 | coverage | threshold_value | mean_pairwise_disagreement | mean_member_normalized_mi |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.956213 | 0.935983 | 0.946139 | 0.956243 | 0.996626 | 0.865248 | 0.863801 | 0.996631 | 0.628402 | 0.930000 | 0.085092 | 0.056562 |
| 1 | 0.720240 | 0.810032 | 0.801602 | 0.692005 | 0.929883 | 0.737774 | 0.747497 | 0.915195 | 0.451257 | 0.930000 | 0.175590 | 0.087200 |
| 2 | 0.576715 | 0.643738 | 0.618906 | 0.529525 | 0.671947 | 0.585767 | 0.592869 | 0.595943 | 0.683558 | 0.860000 | 0.127723 | 0.061711 |
| 3 | 0.350276 | 0.567820 | 0.555788 | 0.278278 | 0.493368 | 0.493348 | 0.499329 | 0.353036 | 0.478646 | 0.920000 | 0.121503 | 0.054921 |
| 4 | 0.283893 | 0.535154 | 0.521963 | 0.199424 | 0.310937 | 0.491106 | 0.506748 | 0.187005 | 0.747228 | 0.850000 | 0.083061 | 0.042133 |
| 5 | 0.219334 | 0.548792 | 0.493836 | 0.160347 | 0.316635 | 0.526239 | 0.523976 | 0.208153 | 0.535793 | 0.830000 | 0.182760 | 0.074651 |
| 6 | 0.185197 | 0.488377 | 0.471157 | 0.129184 | 0.242872 | 0.444046 | 0.456548 | 0.137577 | 0.518735 | 0.860000 | 0.122632 | 0.067004 |
| 7 | 0.176639 | 0.498967 | 0.433182 | 0.121988 | 0.294128 | 0.474290 | 0.462880 | 0.181370 | 0.458869 | 0.790000 | 0.261772 | 0.087218 |

## Forgetting summary

- Ensemble mean forgetting trên các old task: `0.342789`.
- Member task forgetting, mean ± population std: `0.342357 ± 0.009612`.
- Member class forgetting ở task cuối, mean ± population std: `0.223801 ± 0.004652`.

### Member forgetting

| member_id | task_forgetting_mean_old_tasks | class_forgetting_mean_final | class_forgetting_median_final | zero_f1_classes_final | evaluated_classes_final |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.329529 | 0.219417 | 0.171209 | 6 | 178 |
| 1 | 0.344874 | 0.230241 | 0.166667 | 7 | 178 |
| 2 | 0.352667 | 0.221746 | 0.166667 | 7 | 178 |

### Ensemble task-group forgetting

| task_id | best_macro_f1 | final_macro_f1 | forgetting |
| --- | --- | --- | --- |
| 0 | 0.946139 | 0.581144 | 0.364995 |
| 1 | 0.909719 | 0.496231 | 0.413489 |
| 2 | 0.781522 | 0.524587 | 0.256935 |
| 3 | 0.908192 | 0.508041 | 0.400151 |
| 4 | 0.889103 | 0.650729 | 0.238375 |
| 5 | 0.867569 | 0.471138 | 0.396431 |
| 6 | 0.906420 | 0.577272 | 0.329149 |
| 7 | 0.887873 | 0.887873 | 0.000000 |
| mean_old_tasks | n.a. | n.a. | 0.342789 |

Forgetting của task là best Macro-F1 từng đạt trừ Macro-F1 tại task cuối. `mean_old_tasks` chỉ lấy các task cũ, không tính task cuối vừa học.

## Ensemble diversity

| task_id | sample_count | member_count | mean_pairwise_disagreement | mean_mutual_information | mean_member_normalized_mi | mean_total_probability_variance |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 20280 | 3 | 0.085092 | 0.062140 | 0.056562 | 0.030358 |
| 1 | 40549 | 3 | 0.175590 | 0.095799 | 0.087200 | 0.042977 |
| 2 | 66584 | 3 | 0.127723 | 0.067797 | 0.061711 | 0.030743 |
| 3 | 88673 | 3 | 0.121503 | 0.060337 | 0.054921 | 0.027924 |
| 4 | 112817 | 3 | 0.083061 | 0.046288 | 0.042133 | 0.019799 |
| 5 | 133085 | 3 | 0.182760 | 0.082013 | 0.074651 | 0.034780 |
| 6 | 153350 | 3 | 0.122632 | 0.073612 | 0.067004 | 0.027333 |
| 7 | 173614 | 3 | 0.261772 | 0.095818 | 0.087218 | 0.036827 |

Pooled mean pairwise disagreement, có trọng số theo số mẫu: `0.159795`.

Pairwise disagreement đo tỷ lệ các cặp member dự đoán khác nhau. Mutual information và probability variance đo mức khác biệt giữa phân phối xác suất của ba member. Diversity cao hơn không tự động đồng nghĩa accuracy tốt hơn; nó phải được đọc cùng classification metrics và UE error detection.

## Metric definitions

- Balanced Accuracy là macro recall, nên mỗi class có trọng số bằng nhau.
- Weighted F1 lấy F1 từng class và đặt trọng số theo số mẫu thật của class.
- Metric `full_*` dùng toàn bộ test samples của các class đã thấy.
- Metric `threshold_*` chỉ dùng samples vượt frozen validation/OOF threshold và phải được báo cáo cùng coverage.
- Diversity được tổng hợp từ offline ensemble artifact, không chạy lại model.
