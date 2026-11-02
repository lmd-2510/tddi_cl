# A-GEM — Head-to-Tail — Fixed Head Results

## Nguồn kết quả

- Run: `agem_head_to_tail_fixed_head_full_seed0_20261102T125708Z`
- Gói kết quả: `agem_head_to_tail_fixed_head_full_seed0_20261102T125708Z_metrics_review.zip`
- SHA256 ZIP: `4c565a012fb7c8a3cef12ddbd802cf42f92cd343cff56a5019fd6ad04b74ec27`
- Trạng thái: hoàn thành 8 task, 3 member và offline ensemble.

## Cấu hình chính

| Thành phần | Thiết lập |
| --- | --- |
| Method | Original A-GEM, task-uniform episodic memory |
| Protocol | `ORDERED / head_to_tail` |
| Backbone | `tddi_paper_member` |
| Head | Fixed head 178 class, mở mask theo task |
| Task layout | `38, 20, 20, 20, 20, 20, 20, 20` |
| Ensemble | 3 frozen stratified-fold members |
| Experiment seed / fold seed | `0 / 42` |
| Preprocessing | `task0_standard_frozen` |
| Loss | Cross-entropy |
| A-GEM reference batch | `256` |
| Memory | `27,778/member`, `83,334` tổng; quota đều theo task đã hoàn thành |
| Optimizer | AdamW, learning rate `0.001`, weight decay `0.0001` |
| Batch | Microbatch `64`, effective batch `1024` |
| Training | Tối đa 30 epoch/task, patience `5` |

## Kết quả chính — offline ensemble

Các metric dưới đây là full-coverage test metric trên toàn bộ mẫu của các class đã thấy.

| Phạm vi | Accuracy | Balanced Accuracy | Macro-F1 | Weighted-F1 |
| --- | ---: | ---: | ---: | ---: |
| Task 7 / kết quả cuối | 0.665885 | 0.507593 | **0.388085** | 0.755299 |
| Trung bình 8 task | 0.769422 | 0.699066 | **0.593823** | 0.819187 |

Selective/threshold Macro-F1 trung bình là `0.589783` với coverage trung bình `0.896204`; đây không phải metric chính để so sánh CL vì coverage không luôn đạt 100%.

## Diễn biến theo task

| Task | Nhóm | Class mới | Test mẫu mới | Tổng test đã thấy | Accuracy | Balanced Accuracy | Macro-F1 | Weighted-F1 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | Head | 38 | 164,910 | 164,910 | 0.941107 | 0.948290 | 0.948217 | 0.940924 |
| 1 | Mid | 20 | 4,696 | 169,606 | 0.877510 | 0.836372 | 0.780486 | 0.881723 |
| 2 | Mid | 20 | 2,220 | 171,826 | 0.839122 | 0.784289 | 0.661212 | 0.853849 |
| 3 | Mid | 20 | 1,049 | 172,875 | 0.792075 | 0.719706 | 0.566108 | 0.826049 |
| 4 | Mid | 20 | 427 | 173,302 | 0.722150 | 0.641320 | 0.509323 | 0.790258 |
| 5 | Mid | 20 | 174 | 173,476 | 0.681460 | 0.598081 | 0.465344 | 0.766749 |
| 6 | Mid | 20 | 93 | 173,569 | 0.636064 | 0.556875 | 0.431812 | 0.738646 |
| 7 | Tail | 20 | 45 | 173,614 | 0.665885 | 0.507593 | 0.388085 | 0.755299 |

Macro-F1 giảm từ `0.948217` ở task 0 xuống `0.388085` ở task 7. Riêng task 6 → 7 giảm `0.043727` (`0.431812 → 0.388085`).

## Continual-learning metrics

| Metric | Giá trị |
| --- | ---: |
| Mean BWT trên task cũ | **-0.325515** |
| Mean forgetting trên task cũ | **0.333314** |
| Member task-forgetting, mean ± std | 0.333532 ± 0.013551 |
| Member class-forgetting cuối, mean ± std | 0.125470 ± 0.004828 |

Task-group forgetting lớn nhất:

- Task 5: `0.686796`
- Task 6: `0.567420`
- Task 4: `0.584058`
- Task 0: `0.207222`

BWT âm mạnh nhất cũng nằm ở task 5 (`-0.686796`), task 6 (`-0.567420`) và task 4 (`-0.584058`). Điều này cho thấy A-GEM vẫn quên đáng kể các task trung gian/tail trước khi đến task 7.

## Kết quả từng member ở task 7

| Member | Accuracy | Balanced Accuracy | Macro-F1 | Weighted-F1 |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.546217 | 0.400880 | 0.284424 | 0.658086 |
| 1 | 0.516306 | 0.438203 | 0.316239 | 0.638385 |
| 2 | 0.548533 | 0.459653 | 0.330009 | 0.664263 |
| Trung bình member | 0.537019 | 0.432912 | 0.310224 | 0.653578 |
| Offline ensemble | **0.665885** | **0.507593** | **0.388085** | **0.755299** |

Ensemble tăng Macro-F1 khoảng `+0.077861` so với trung bình member. Tuy nhiên, phần tăng này không loại bỏ được forgetting nội tại của từng member.

## Nhận xét ngắn

1. A-GEM có kết quả cuối Macro-F1 `0.388085`, thấp hơn đáng kể so với chất lượng task 0; BWT `-0.325515` xác nhận gradient projection chưa đủ để bảo toàn các task cũ trên dữ liệu long-tail.
2. Task 5–6 là vùng suy giảm nghiêm trọng nhất. Task 7 có chỉ 45 mẫu test mới, vì vậy Macro-F1 cuối phản ánh chủ yếu chất lượng trên các class cũ đã bị quên, không chỉ khả năng học task 7.
3. Offline ensemble giúp cải thiện rõ rệt so với từng member, nhưng diversity tăng dần qua các task (`0.087` ở task 0 lên `0.583` ở task 6), cho thấy các member ngày càng bất đồng khi forgetting tăng.

