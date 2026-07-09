# Class Distribution Summary

## Key Findings

- Dataset exhibits strong long-tail behavior across all three splits.
- Train split remains the reference split for task design and rare-class policy.
- Main benchmark should keep all 178 classes; rare classes are reported separately, not removed.

## Split-Level Summary

| split | num_classes | min_count | max_count | median_count | mean_count | num_classes_<=5 | num_classes_<=10 | num_classes_<=20 | num_classes_<=50 | imbalance_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 178 | 3 | 73634 | 147.0 | 2926.0730337078653 | 9 | 20 | 45 | 64 | 24544.666666666668 |
| validation | 178 | 1 | 24545 | 49.0 | 975.3595505617977 | 39 | 54 | 68 | 90 | 24545.0 |
| test | 178 | 1 | 24545 | 48.5 | 975.3595505617977 | 35 | 54 | 68 | 90 | 24545.0 |

## Top Classes

### Train Top-10

| class_id | count | split |
| --- | --- | --- |
| 137 | 73634 | train |
| 6 | 65314 | train |
| 15 | 58826 | train |
| 25 | 40864 | train |
| 13 | 33630 | train |
| 10 | 23051 | train |
| 30 | 19609 | train |
| 28 | 18958 | train |
| 7 | 18245 | train |
| 32 | 17330 | train |

### Validation Top-10

| class_id | count | split |
| --- | --- | --- |
| 137 | 24545 | validation |
| 6 | 21771 | validation |
| 15 | 19608 | validation |
| 25 | 13621 | validation |
| 13 | 11210 | validation |
| 10 | 7683 | validation |
| 30 | 6537 | validation |
| 28 | 6319 | validation |
| 7 | 6082 | validation |
| 32 | 5777 | validation |

### Test Top-10

| class_id | count | split |
| --- | --- | --- |
| 137 | 24545 | test |
| 6 | 21771 | test |
| 15 | 19609 | test |
| 25 | 13621 | test |
| 13 | 11210 | test |
| 10 | 7684 | test |
| 30 | 6536 | test |
| 28 | 6320 | test |
| 7 | 6082 | test |
| 32 | 5777 | test |

## Rare-Class Policy

- Keep the full 178-class benchmark as the main setting.
- Report rare-class metrics separately for thresholds `<=5`, `<=10`, and `<=20`.
- Do not overclaim per-class F1 for classes with only a few samples.
- Optional `min-count filtered benchmark` remains an ablation, not the main protocol.

## Rare-Class Preview

| class_id | train_count | validation_count | test_count | train_le_5 | train_le_10 | train_le_20 | train_le_50 | rare_bucket_train |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 111 | 3 | 1 | 1 | True | True | True | True | <=5 |
| 104 | 4 | 1 | 1 | True | True | True | True | <=5 |
| 182 | 4 | 1 | 2 | True | True | True | True | <=5 |
| 108 | 5 | 1 | 2 | True | True | True | True | <=5 |
| 116 | 5 | 2 | 2 | True | True | True | True | <=5 |
| 145 | 5 | 2 | 2 | True | True | True | True | <=5 |
| 160 | 5 | 2 | 2 | True | True | True | True | <=5 |
| 191 | 5 | 1 | 2 | True | True | True | True | <=5 |
| 199 | 5 | 2 | 2 | True | True | True | True | <=5 |
| 86 | 6 | 2 | 2 | False | True | True | True | <=10 |
| 176 | 6 | 1 | 2 | False | True | True | True | <=10 |
| 119 | 7 | 2 | 2 | False | True | True | True | <=10 |
| 59 | 8 | 3 | 2 | False | True | True | True | <=10 |
| 120 | 8 | 3 | 3 | False | True | True | True | <=10 |
| 122 | 8 | 3 | 3 | False | True | True | True | <=10 |

## Interpretation Notes

- Large head classes can dominate optimization and hide tail failures if only overall accuracy is reported.
- Macro-F1 and Balanced Accuracy are critical for this dataset.
- Replay memory should be class-balanced rather than globally random.
