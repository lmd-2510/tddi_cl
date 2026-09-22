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
| Full Accuracy | 0.911333 |
| Full Balanced Accuracy | 0.817983 |
| Full Macro-F1 | 0.841864 |
| Full Weighted F1 | 0.903920 |
| Threshold Accuracy | 0.972063 |
| Threshold Balanced Accuracy | 0.787795 |
| Threshold Macro-F1 | 0.797032 |
| Threshold Weighted F1 | 0.968654 |
| Average Coverage | 0.776185 |

## Per-task results

| task_id | full_accuracy | full_balanced_accuracy | full_macro_f1 | full_weighted_f1 | threshold_accuracy | threshold_balanced_accuracy | threshold_macro_f1 | threshold_weighted_f1 | coverage | threshold_value | mean_pairwise_disagreement | mean_member_normalized_mi |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.936508 | 0.941667 | 0.940216 | 0.932494 | 0.990476 | 0.894737 | 0.890977 | 0.986395 | 0.833333 | 0.880000 | 0.097884 | 0.079229 |
| 1 | 0.920139 | 0.908477 | 0.913513 | 0.916288 | 0.985149 | 0.876437 | 0.879788 | 0.983925 | 0.701389 | 0.830000 | 0.168981 | 0.108651 |
| 2 | 0.919591 | 0.875193 | 0.887907 | 0.911822 | 0.965278 | 0.881884 | 0.893749 | 0.961231 | 0.842105 | 0.640000 | 0.100877 | 0.075823 |
| 3 | 0.929921 | 0.873604 | 0.890776 | 0.923979 | 0.961082 | 0.927702 | 0.933307 | 0.957187 | 0.923827 | 0.540000 | 0.086736 | 0.059005 |
| 4 | 0.937856 | 0.874095 | 0.893555 | 0.934672 | 0.963093 | 0.919271 | 0.925675 | 0.961207 | 0.926459 | 0.550000 | 0.079331 | 0.055413 |
| 5 | 0.941483 | 0.849937 | 0.873739 | 0.938902 | 0.962670 | 0.882119 | 0.893043 | 0.961206 | 0.925208 | 0.590000 | 0.066634 | 0.048096 |
| 6 | 0.872832 | 0.767034 | 0.795656 | 0.864638 | 0.971872 | 0.744609 | 0.770873 | 0.968440 | 0.645016 | 0.860000 | 0.102094 | 0.057387 |
| 7 | 0.832335 | 0.453853 | 0.539550 | 0.808565 | 0.976885 | 0.175604 | 0.188841 | 0.969644 | 0.412144 | 0.970000 | 0.146760 | 0.082692 |

## Forgetting summary

- Ensemble mean forgetting trên các old task: `0.460143`.
- Member task forgetting, mean ± population std: `0.459440 ± 0.006301`.
- Member class forgetting ở task cuối, mean ± population std: `0.406360 ± 0.006309`.

### Member forgetting

| member_id | task_forgetting_mean_old_tasks | class_forgetting_mean_final | class_forgetting_median_final | zero_f1_classes_final | evaluated_classes_final |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.450576 | 0.398421 | 0.437794 | 13 | 178 |
| 1 | 0.464668 | 0.413857 | 0.443129 | 15 | 178 |
| 2 | 0.463075 | 0.406803 | 0.437409 | 18 | 178 |

### Ensemble task-group forgetting

| task_id | best_macro_f1 | final_macro_f1 | forgetting |
| --- | --- | --- | --- |
| 0 | 0.940216 | 0.524905 | 0.415311 |
| 1 | 0.965189 | 0.584916 | 0.380274 |
| 2 | 0.970556 | 0.667451 | 0.303106 |
| 3 | 0.979394 | 0.432116 | 0.547279 |
| 4 | 0.982349 | 0.529280 | 0.453069 |
| 5 | 0.990189 | 0.439604 | 0.550585 |
| 6 | 0.979217 | 0.407842 | 0.571375 |
| 7 | 0.921011 | 0.921011 | 0.000000 |
| mean_old_tasks | n.a. | n.a. | 0.460143 |

Forgetting của task là best Macro-F1 từng đạt trừ Macro-F1 tại task cuối. `mean_old_tasks` chỉ lấy các task cũ, không tính task cuối vừa học.

## Ensemble diversity

| task_id | sample_count | member_count | mean_pairwise_disagreement | mean_mutual_information | mean_member_normalized_mi | mean_total_probability_variance |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 126 | 3 | 0.097884 | 0.087042 | 0.079229 | 0.048619 |
| 1 | 288 | 3 | 0.168981 | 0.119365 | 0.108651 | 0.049863 |
| 2 | 684 | 3 | 0.100877 | 0.083301 | 0.075823 | 0.024310 |
| 3 | 1641 | 3 | 0.086736 | 0.064823 | 0.059005 | 0.018170 |
| 4 | 3685 | 3 | 0.079331 | 0.060878 | 0.055413 | 0.017758 |
| 5 | 8049 | 3 | 0.066634 | 0.052839 | 0.048096 | 0.015336 |
| 6 | 23756 | 3 | 0.102094 | 0.063046 | 0.057387 | 0.023047 |
| 7 | 173614 | 3 | 0.146760 | 0.090847 | 0.082692 | 0.043404 |

Pooled mean pairwise disagreement, có trọng số theo số mẫu: `0.136922`.

Pairwise disagreement đo tỷ lệ các cặp member dự đoán khác nhau. Mutual information và probability variance đo mức khác biệt giữa phân phối xác suất của ba member. Diversity cao hơn không tự động đồng nghĩa accuracy tốt hơn; nó phải được đọc cùng classification metrics và UE error detection.

## Metric definitions

- Balanced Accuracy là macro recall, nên mỗi class có trọng số bằng nhau.
- Weighted F1 lấy F1 từng class và đặt trọng số theo số mẫu thật của class.
- Metric `full_*` dùng toàn bộ test samples của các class đã thấy.
- Metric `threshold_*` chỉ dùng samples vượt frozen validation/OOF threshold và phải được báo cáo cùng coverage.
- Diversity được tổng hợp từ offline ensemble artifact, không chạy lại model.
