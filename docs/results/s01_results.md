# S01 Full Experiment Results

## 1. Phạm vi

Full S01 experiment hoàn thành ngày 2026-08-03 với cấu hình:

```text
methods = sequential, replay, replay_distill, joint_seen
seeds   = 0, 1, 2, 3, 4
tasks   = 8
device  = mps
```

Tổng cộng có 20 runs trong `outputs/runs_s01_full/`. Các giá trị dưới đây là mean ± sample standard deviation trên 5 seeds.

## 2. Kiểm tra tính toàn vẹn

Tất cả 20 runs đều đạt các điều kiện:

- có event `run_completed`;
- `class_trajectory.csv`: 864 dòng/run;
- `class_forgetting.csv`: 864 dòng/run;
- `metrics.csv`: 44 dòng/run;
- `task_matrix.csv`: 8 dòng/run;
- `forgetting.csv`: 9 dòng/run;
- task cuối có đủ 178 classes;
- không trùng khóa `(seed, method, train_task, class_id)`;
- mọi class có `test_count > 0`;
- forgetting không âm và bằng 0 tại lần đầu class xuất hiện.

Kết quả gộp gồm 17.280 class-trajectory rows và 17.280 class-forgetting rows.

## 3. Final seen-class performance

| Method | Accuracy | Macro-F1 | Weighted-F1 | Balanced Accuracy |
| --- | ---: | ---: | ---: | ---: |
| `joint_seen` | 0.9266 ± 0.0046 | 0.8344 ± 0.0072 | 0.9262 ± 0.0047 | 0.8264 ± 0.0078 |
| `replay` | 0.1905 ± 0.0587 | 0.2682 ± 0.0267 | 0.1452 ± 0.0288 | 0.6383 ± 0.0179 |
| `replay_distill` | 0.2235 ± 0.0431 | 0.2611 ± 0.0217 | 0.1636 ± 0.0270 | 0.6110 ± 0.0166 |
| `sequential` | 0.1368 ± 0.1000 | 0.0341 ± 0.0130 | 0.0604 ± 0.0507 | 0.0785 ± 0.0308 |

Final Macro-F1 theo seed:

| Seed | `joint_seen` | `replay` | `replay_distill` | `sequential` |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.8361 | 0.2787 | 0.2899 | 0.0385 |
| 1 | 0.8301 | 0.2388 | 0.2389 | 0.0179 |
| 2 | 0.8272 | 0.2635 | 0.2530 | 0.0243 |
| 3 | 0.8329 | 0.3081 | 0.2776 | 0.0503 |
| 4 | 0.8459 | 0.2521 | 0.2460 | 0.0396 |

## 4. Forgetting và class coverage

| Method | Task-level forgetting | Final class-wise forgetting | Số class có F1 = 0 ở task cuối |
| --- | ---: | ---: | ---: |
| `joint_seen` | 0.0229 ± 0.0050 | 0.0574 ± 0.0081 | 4.6 ± 1.5 |
| `replay` | 0.2381 ± 0.0246 | 0.2161 ± 0.0283 | 1.4 ± 0.5 |
| `replay_distill` | 0.2008 ± 0.0285 | 0.1979 ± 0.0331 | 2.0 ± 0.0 |
| `sequential` | 0.8269 ± 0.0796 | 0.4531 ± 0.0478 | 157.2 ± 4.0 |

Định nghĩa:

- task-level forgetting lấy từ dòng `mean_old_tasks` trong `forgetting.csv`;
- final class-wise forgetting là trung bình `forgetting` của 178 classes tại task 7;
- số class có F1 bằng 0 được đếm tại task 7.

## 5. Nhận xét ban đầu

1. `joint_seen` đạt Macro-F1 0.8344, gần static baseline đã báo cáo trong repo, cho thấy evaluation pipeline và full-data oracle hoạt động hợp lý.
2. `sequential` bị catastrophic forgetting rất mạnh: task-level forgetting 0.8269 và trung bình 157/178 classes có F1 bằng 0 ở task cuối.
3. Cả hai replay methods cải thiện rõ rệt so với `sequential` về final Macro-F1, balanced accuracy, forgetting và class coverage.
4. `replay` có final Macro-F1 cao hơn nhẹ, trong khi `replay_distill` có forgetting thấp hơn. Chưa kết luận phương pháp nào tốt hơn nếu chưa có paired statistical analysis.
5. Replay có balanced accuracy cao nhưng Macro-F1 và weighted-F1 thấp hơn nhiều. Điều này gợi ý Recall giữa các class tương đối rộng nhưng Precision còn yếu; cần phân tích confusion và class frequency trong E01.
6. Confidence và entropy hiện chưa calibrated, vì vậy không dùng các giá trị này để kết luận độ tin cậy trước S03.

S01 chỉ xác nhận pipeline class-wise và tạo dữ liệu đầu vào. Các câu hỏi về rare-class forgetting, multi-prototype và uncertainty vẫn cần E01–E03.

## 6. Kết luận S01

S01 hoàn thành với đủ F1 trajectory và class-wise forgetting cho 20 main runs. Hai artifact O01/O02 đã sẵn sàng để dùng cho E01 và làm nguồn temporal performance cho E03.
