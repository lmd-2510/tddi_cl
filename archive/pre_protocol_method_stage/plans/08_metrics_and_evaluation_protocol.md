# Metrics and Evaluation Protocol

## Objective

Chuẩn hóa cách evaluate sau mỗi task để so sánh continual methods công bằng, đọc được forgetting thật, và không che khuất thất bại ở rare classes.

## Evaluation Rule

Sau khi train task `t`:

1. Evaluate trên **test samples thuộc tất cả seen classes đến task t**
2. Evaluate riêng cho từng group class của các task cũ và task mới
3. Không dùng unseen/future classes trong main metric

Validation được dùng cho:

- early stopping
- chọn epoch tốt nhất
- không dùng test cho tuning

## Required Classification Metrics

- Accuracy
- Macro-F1
- Weighted-F1
- Balanced Accuracy
- Per-class F1
- Confusion matrix top errors

## Required Continual Learning Metrics

- Average incremental accuracy
- Average incremental Macro-F1
- Final average accuracy
- Forgetting per task
- Mean forgetting
- Backward transfer
- Performance drop from best to final

## Required Calibration Metrics

- ECE
- Brier score
- Negative log-likelihood
- confidence vs accuracy
- reliability diagram
- selective risk/coverage nếu khả thi

## Required Long-Tail Metrics

- Head-class Macro-F1
- Medium-class Macro-F1
- Tail-class Macro-F1
- Rare-class forgetting
- Rare-class ECE

## Result Matrix

Phải tạo ma trận:

```text
R[t_train, t_test]
```

Trong đó:

- `R[i, j]` = performance trên task `j` sau khi train xong task `i`
- chỉ hợp lệ khi `j <= i`

## Forgetting Formula

```text
forgetting_j = max_i R[i, j] - R[T, j], for i in [j, ..., T]
mean_forgetting = average(forgetting_j over old tasks)
```

## Planned Outputs

```text
outputs/results/task_performance_matrix.csv
outputs/results/incremental_metrics.csv
outputs/results/forgetting_metrics.csv
outputs/results/per_class_metrics.csv
outputs/results/calibration_metrics.csv
outputs/figures/performance_over_tasks.png
outputs/figures/forgetting_by_task.png
outputs/figures/rare_class_performance.png
outputs/figures/reliability_diagram_task_final.png
```

## Evaluation Granularity

### After Each Task

- overall seen-class metrics
- per-task-group metrics
- per-class metrics
- calibration metrics

### Final

- final mean across seeds
- mean ± std theo method
- rare-class summary
- calibration drift summary

## Rare-Class Reporting

Ít nhất phải có:

- metric riêng cho các lớp `<= 5`
- metric riêng cho các lớp `<= 10`
- metric riêng cho các lớp `<= 20`

Ngưỡng chính thức lấy từ output của `03_class_distribution_and_long_tail_analysis.md`.

## Pseudocode

```python
for task_train in tasks:
    model = train_task(...)
    seen = union_classes_up_to(task_train)

    for task_test in seen_task_groups:
        scores = evaluate(model, test_subset_for(task_test, seen_only=True))
        R[task_train, task_test] = scores["macro_f1"]

forgetting = compute_forgetting(R)
save_all_metrics(R, forgetting, calibration, per_class)
```

## Acceptance Criteria

- [ ] Macro metrics được ưu tiên, không chỉ accuracy tổng
- [ ] Forgetting được tính theo công thức rõ ràng
- [ ] Calibration và rare-class metrics có output artifact riêng
- [ ] Rule “seen classes only” cho main metric được ghi rõ

## Definition of Done

- [ ] Có spec đầy đủ cho result matrix và forgetting
- [ ] Có danh sách artifact và figure phải xuất
- [ ] Không có ambiguity về việc evaluate trên split nào và class subset nào
