# IDER — Head-to-Tail — Fixed Head Results

## Nguồn kết quả

- Run: `ider_head_to_tail_fixed_head_full_seed0_20261103T104614Z`
- Gói kết quả: `ider_head_to_tail_fixed_head_full_seed0_20261103T104614Z_metrics_review.zip`
- SHA256 ZIP: `E3B9A47F7D5052078D8654ACADF1DF85632FE9BCADF75A9C6105FC40CFAAF236`
- Trạng thái: hoàn thành 8 task, 3 member và offline ensemble.

## Cấu hình chính

| Thành phần | Thiết lập |
| --- | --- |
| Method | IDER |
| Protocol | `ORDERED / head_to_tail` |
| Backbone | `tddi_paper_member` |
| Head | Fixed 178-class head, mở mask theo task |
| Task layout | `38, 20, 20, 20, 20, 20, 20, 20` |
| Ensemble | 3 frozen stratified-fold members |
| Experiment seed / fold seed | `0 / 42` |
| Preprocessing | `task0_standard_frozen` |
| Buffer | Equal-class fixed budget, `27,778/member`, `83,334` tổng |
| Replay | `12.5%`, repeat cap `3`, IDER replay batch `64` |
| IDER coefficients | `a=0.5`, `b=0.4`, `c=0.5`, uniform probability `1.0` |
| Optimizer | AdamW, learning rate `0.001`, weight decay `0.0001` |
| Batch | Microbatch `64`, effective batch `1024` |
| Training | Tối đa 30 epoch/task, patience `5` |

## Kết quả chính — offline ensemble

Các metric chính là full-coverage test metric tại task 7.

| Phạm vi | Accuracy | Balanced Accuracy | Macro-F1 | Weighted-F1 |
| --- | ---: | ---: | ---: | ---: |
| Task 7 / kết quả cuối | 0.727418 | 0.604937 | **0.376521** | 0.784882 |
| Trung bình 8 task | 0.806714 | 0.776714 | **0.589979** | 0.843064 |

Threshold Macro-F1 ở task 7 cũng là `0.376521` với coverage `1.0`; do đó trong run này threshold không lọc thêm mẫu ở task cuối.

## Diễn biến Macro-F1 theo task

| Task | Nhóm | Test mẫu mới | Tổng test đã thấy | Full Macro-F1 | Balanced Accuracy |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0 | Head | 164,910 | 164,910 | 0.950811 | 0.950645 |
| 1 | Mid | 4,696 | 169,606 | 0.790746 | 0.779505 |
| 2 | Mid | 2,220 | 171,826 | 0.685138 | 0.857013 |
| 3 | Mid | 1,049 | 172,875 | 0.569265 | 0.809789 |
| 4 | Mid | 427 | 173,302 | 0.495068 | 0.795325 |
| 5 | Mid | 174 | 173,476 | 0.444596 | 0.746328 |
| 6 | Mid | 93 | 173,569 | 0.407684 | 0.670169 |
| 7 | Tail | 45 | 173,614 | 0.376521 | 0.604937 |

Macro-F1 giảm `0.031163` từ task 6 sang task 7. Mức giảm cuối nhỏ hơn các bước trước, nhưng chất lượng tổng thể vẫn giảm dần theo thứ tự head-to-tail.

## Continual-learning metrics

| Metric | Giá trị |
| --- | ---: |
| Mean BWT trên task cũ | **-0.028054** |
| Mean forgetting trên task cũ | **0.105573** |
| Member task-forgetting, mean ± std | 0.092112 ± 0.026413 |
| Member class-forgetting cuối, mean ± std | 0.059790 ± 0.001685 |
| Zero-F1 classes cuối/member | 51 / 43 / 21 |

IDER có BWT và forgetting tốt hơn rõ rệt so với các run EWC, A-GEM và ER trước đó, nghĩa là giữ các task cũ tốt hơn. Tuy nhiên, điểm cuối task 7 thấp hơn ER; khả năng giữ kiến thức không đồng nghĩa với khả năng học tốt các lớp tail mới.

## Forgetting theo task-group

| Task | Best Macro-F1 | Final Macro-F1 | Forgetting |
| ---: | ---: | ---: | ---: |
| 0 | 0.950811 | 0.877104 | 0.073707 |
| 1 | 0.925519 | 0.899784 | 0.025735 |
| 2 | 0.921966 | 0.909597 | 0.012370 |
| 3 | 0.832218 | 0.788534 | 0.043684 |
| 4 | 0.756312 | 0.593593 | 0.162719 |
| 5 | 0.527617 | 0.307652 | 0.219965 |
| 6 | 0.320833 | 0.120000 | 0.200833 |

Forgetting lớn nhất ở task 5 và task 6; task 4 cũng đáng chú ý. Đây là các nhóm có ít test mẫu lớp mới, nên Macro-F1 nhạy với sai số trên từng lớp.

## Kết quả từng member ở task 7

| Member | Accuracy | Balanced Accuracy | Macro-F1 | Weighted-F1 |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.646820 | 0.494186 | 0.296797 | 0.719541 |
| 1 | 0.630329 | 0.545398 | 0.311009 | 0.712015 |
| 2 | 0.616678 | 0.601741 | 0.329486 | 0.702620 |
| Trung bình member | 0.631276 | 0.547108 | 0.312431 | 0.711392 |
| Offline ensemble | **0.727418** | **0.604937** | **0.376521** | **0.784882** |

Offline ensemble tăng Macro-F1 khoảng `+0.064090` so với trung bình member.

## Nhận xét ngắn

1. IDER là run có khả năng giữ kiến thức cũ tốt nhất trong nhóm hiện đã có: BWT chỉ `-0.028054` và forgetting `0.105573`.
2. Điểm cuối Macro-F1 `0.376521` vẫn thấp, vì IDER ưu tiên ổn định biểu diễn và replay; nó không tự giải quyết intra-task long-tail của các lớp mới.
3. Task 6→7 chỉ giảm khoảng `0.031`, nhỏ hơn nhiều so với các bước trước. Vì vậy vấn đề chính không chỉ là catastrophic forgetting ở task cuối, mà là chất lượng học các nhóm mid/tail vốn có rất ít mẫu.
4. Ensemble giúp đáng kể so với từng member, nhưng không bù được hoàn toàn sự suy giảm Macro-F1 của các lớp ít mẫu.

