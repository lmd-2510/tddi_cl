# TIES — Head-to-Tail — Fixed Head Results

## Nguồn kết quả

- Run: `ties_head_to_tail_fixed_head_full_seed0_20261104T235759Z`
- Gói kết quả: `ties_head_to_tail_fixed_head_full_seed0_20261104T235759Z_metrics_review.zip`
- SHA256 ZIP: `0EBF182D8C97001E2C11D176636670A12772B6670B5126685AE64110B0FFED01`
- Trạng thái: hoàn thành 8 boundary, 3 member và offline ensemble.

## Cấu hình chính

| Thành phần | Thiết lập |
| --- | --- |
| Method | TIES |
| Protocol | `ORDERED / head_to_tail` |
| Backbone | `tddi_paper_member` |
| Head | Fixed 178-class head, ownership theo task |
| Task layout | `38, 20, 20, 20, 20, 20, 20, 20` |
| Ensemble | 3 frozen stratified-fold members |
| Experiment seed / fold seed | `0 / 42` |
| Preprocessing | `task0_standard_frozen` |
| Merge scope | Learned backbone parameters only; head rows được stitch theo ownership |
| Training recipe | Task 0 là base; mỗi expert task 1–7 reset từ task-0 base |
| TIES density | `0.20` |
| TIES scale | `1.0` |
| Sign election | `mass` |
| Reduction | `disjoint_mean` |
| Optimizer | AdamW, learning rate `0.001`, weight decay `0.0001` |
| Batch | Microbatch `64`, effective batch `1024` |
| Training | Tối đa 30 epoch/task, patience `5` |

## Kết quả chính — offline ensemble

Các metric là full-coverage test metric tại boundary cuối (task 7).

| Phạm vi | Accuracy | Balanced Accuracy | Macro-F1 | Weighted-F1 |
| --- | ---: | ---: | ---: | ---: |
| Task 7 / kết quả cuối | 0.854148 | 0.208110 | **0.231388** | 0.837363 |
| Trung bình Macro-F1 của 8 boundary | — | — | **0.460646** | — |

## Diễn biến theo boundary

| Boundary | Số class đã thấy | Test samples | Accuracy | Balanced Accuracy | Macro-F1 | Weighted-F1 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 38 | 164,910 | 0.940143 | 0.947025 | 0.947408 | 0.939943 |
| 1 | 58 | 169,606 | 0.892238 | 0.649656 | 0.681373 | 0.884651 |
| 2 | 78 | 171,826 | 0.855237 | 0.472604 | 0.523469 | 0.844332 |
| 3 | 98 | 172,875 | 0.835719 | 0.360739 | 0.407611 | 0.822301 |
| 4 | 118 | 173,302 | 0.832789 | 0.300302 | 0.341302 | 0.818141 |
| 5 | 138 | 173,476 | 0.840208 | 0.261256 | 0.294502 | 0.824604 |
| 6 | 158 | 173,569 | 0.846845 | 0.230489 | 0.258115 | 0.830598 |
| 7 | 178 | 173,614 | 0.854148 | 0.208110 | 0.231388 | 0.837363 |

## So sánh trực tiếp với Task Arithmetic

| Method | Accuracy | Balanced Accuracy | Macro-F1 | Weighted-F1 |
| --- | ---: | ---: | ---: | ---: |
| Task Arithmetic | 0.819018 | 0.184912 | 0.208722 | 0.802104 |
| TIES | **0.854148** | **0.208110** | **0.231388** | **0.837363** |

TIES cải thiện so với Task Arithmetic khoảng `+0.022666` Macro-F1 ở boundary cuối. Tuy nhiên, mức Macro-F1 tuyệt đối vẫn thấp.

## Nhận xét

1. TIES tốt hơn Task Arithmetic trên cả bốn metric cuối. Việc trim 20%, chọn sign theo mass và dùng `disjoint_mean` giúp giảm một phần xung đột giữa các task vector.
2. Macro-F1 vẫn giảm từ `0.947408` ở boundary 0 xuống `0.231388` ở boundary 7. Mức giảm từ boundary 6 sang 7 là `0.026727`; vấn đề chính là suy giảm tích lũy khi số class tăng.
3. Accuracy `0.854148` và Weighted-F1 `0.837363` cao hơn nhiều so với Macro-F1. Đây là bằng chứng model vẫn thiên về các class head có nhiều mẫu.
4. Balanced Accuracy cuối chỉ `0.208110`, cho thấy recall trung bình trên 178 class rất thấp dù accuracy tổng thể cao.
5. TIES hiện chưa có BWT/forgetting trong package vì đây là independent-expert merging, không phải trajectory CL tuần tự. Không nên so sánh BWT của TIES trực tiếp với EWC/ER/UPCL.

## Kết luận

TIES là phép merge tốt hơn Task Arithmetic trong cùng pipeline, nhưng chưa giải quyết được long-tail của TDDI-CIL. Cải thiện hiện tại chủ yếu đến từ việc giảm xung đột task vector; nó chưa đủ để nâng chất lượng các class mid/tail lên mức Macro-F1 cao.

