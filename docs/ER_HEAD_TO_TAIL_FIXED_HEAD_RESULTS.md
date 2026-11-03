# ER — Head-to-Tail — Fixed Head Results

## Nguồn kết quả

- Run: `er_head_to_tail_fixed_head_full_seed0_20261103T032635Z`
- Gói kết quả: `er_head_to_tail_fixed_head_full_seed0_20261103T032635Z_metrics_review.zip`
- SHA256 ZIP: `BE6D8895E83274000AAAA8662F1016ADEF544FF450B33B203B89C7F63E9A75EA`
- Trạng thái: hoàn thành 8 task, 3 member và offline ensemble.

## Cấu hình chính

| Thành phần | Thiết lập |
| --- | --- |
| Method | Experience Replay, uniform reservoir |
| Protocol | `ORDERED / head_to_tail` |
| Backbone | `tddi_paper_member` |
| Head | Fixed 178-class head, mở mask theo task |
| Task layout | `38, 20, 20, 20, 20, 20, 20, 20` |
| Ensemble | 3 frozen stratified-fold members |
| Experiment seed / fold seed | `0 / 42` |
| Preprocessing | `task0_standard_frozen` |
| Loss | Cross-entropy |
| Buffer | `reservoir_uniform_v1`, `27,778/member`, `83,334` tổng |
| Replay | Current/replay 1:1, uniform sampling with replacement |
| Optimizer | AdamW, learning rate `0.001`, weight decay `0.0001` |
| Batch | Microbatch `64`, effective batch `1024` |
| Training | Tối đa 30 epoch/task, patience `5` |

## Kết quả chính — offline ensemble

Các metric chính là full-coverage test metric tại task 7.

| Phạm vi | Accuracy | Balanced Accuracy | Macro-F1 | Weighted-F1 |
| --- | ---: | ---: | ---: | ---: |
| Task 7 / kết quả cuối | 0.889145 | 0.549176 | **0.500563** | 0.902237 |
| Trung bình 8 task | 0.910235 | 0.809054 | **0.745165** | 0.916117 |

Threshold Macro-F1 ở task 7 là `0.477571` với coverage `0.749214`; đây không phải metric chính để so sánh giữa các run vì coverage không đạt 100%.

## Diễn biến Macro-F1 theo task

| Task | Nhóm | Test mẫu mới | Tổng test đã thấy | Full Macro-F1 | Balanced Accuracy |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0 | Head | 164,910 | 164,910 | 0.948592 | 0.947533 |
| 1 | Mid | 4,696 | 169,606 | 0.880018 | 0.945163 |
| 2 | Mid | 2,220 | 171,826 | 0.849036 | 0.934199 |
| 3 | Mid | 1,049 | 172,875 | 0.814598 | 0.902703 |
| 4 | Mid | 427 | 173,302 | 0.725751 | 0.829305 |
| 5 | Mid | 174 | 173,476 | 0.666455 | 0.748895 |
| 6 | Mid | 93 | 173,569 | 0.576307 | 0.615458 |
| 7 | Tail | 45 | 173,614 | 0.500563 | 0.549176 |

Macro-F1 giảm `0.075744` từ task 6 sang task 7. Như vậy ER cải thiện đáng kể kết quả cuối nhưng chưa loại bỏ hoàn toàn forgetting ở các task trung gian/tail.

## Continual-learning metrics

| Metric | Giá trị |
| --- | ---: |
| Mean BWT trên task cũ | **-0.237162** |
| Mean forgetting trên task cũ | **0.237162** |
| Member task-forgetting, mean ± std | 0.293965 ± 0.027485 |
| Member class-forgetting cuối, mean ± std | 0.086684 ± 0.011438 |
| Zero-F1 classes cuối/member | 52 / 48 / 51 |

Forgetting lớn nhất theo task-group nằm ở task 4 (`0.688792`), tiếp theo là task 5 (`0.332338`) và task 2 (`0.297561`).

## Kết quả từng member ở task 7

| Member | Accuracy | Balanced Accuracy | Macro-F1 | Weighted-F1 |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.821967 | 0.504756 | 0.470285 | 0.861653 |
| 1 | 0.810839 | 0.506341 | 0.453004 | 0.854145 |
| 2 | 0.818333 | 0.524083 | 0.459325 | 0.858534 |
| Trung bình member | 0.817046 | 0.511727 | 0.460871 | 0.858111 |
| Offline ensemble | **0.889145** | **0.549176** | **0.500563** | **0.902237** |

Ensemble tăng Macro-F1 khoảng `+0.039692` so với trung bình member.

