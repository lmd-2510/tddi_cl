# DDI-CIL Experiment Metrics Summary

File này là bảng tổng hợp chung cho các run hoàn chỉnh. Mỗi run dùng kết quả test full-coverage của offline ensemble; không dùng selective metric nếu coverage nhỏ hơn 100%.

## Bảng kết quả

| Run | Method | Protocol | Backbone | Final Accuracy | Final Balanced Accuracy | Final Macro-F1 | Final Weighted-F1 | Mean Task Macro-F1 | BWT | Forgetting | Chi tiết |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `ewc_head_to_tail_fixed_head_full_seed0_20261102T071912Z` | EWC | Head-to-Tail | TDDI | 0.569257 | 0.548442 | **0.377209** | 0.668461 | 0.575171 | -0.210327 | 0.217414 | [EWC results](EWC_HEAD_TO_TAIL_FIXED_HEAD_RESULTS.md) |
| `agem_head_to_tail_fixed_head_full_seed0_20261102T125708Z` | A-GEM | Head-to-Tail | TDDI | 0.665885 | 0.507593 | **0.388085** | 0.755299 | 0.593823 | -0.325515 | 0.333314 | [A-GEM results](AGEM_HEAD_TO_TAIL_FIXED_HEAD_RESULTS.md) |
| `er_head_to_tail_fixed_head_full_seed0_20261103T032635Z` | ER | Head-to-Tail | TDDI | 0.889145 | 0.549176 | **0.500563** | 0.902237 | 0.745165 | -0.237162 | 0.237162 | [ER results](ER_HEAD_TO_TAIL_FIXED_HEAD_RESULTS.md) |
| `ider_head_to_tail_fixed_head_full_seed0_20261103T104614Z` | IDER | Head-to-Tail | TDDI | 0.727418 | 0.604937 | **0.376521** | 0.784882 | 0.589979 | -0.028054 | 0.105573 | [IDER results](IDER_HEAD_TO_TAIL_FIXED_HEAD_RESULTS.md) |
| `upcl_head_to_tail_fixed_head_full_seed0_20261103T165325Z` | UPCL | Head-to-Tail | TDDI | 0.322388 | 0.653767 | **0.219061** | 0.381671 | 0.389315 | 0.147812 | 0.029976 | [UPCL results](UPCL_HEAD_TO_TAIL_FIXED_HEAD_RESULTS.md) |
| `task_arithmetic_head_to_tail_fixed_head_full_seed0_20261104T120427Z` | Task Arithmetic | Head-to-Tail | TDDI | 0.819018 | 0.184912 | **0.208722** | 0.802104 | 0.446440 | n.a. | n.a. | [Task Arithmetic results](TASK_ARITHMETIC_HEAD_TO_TAIL_FIXED_HEAD_RESULTS.md) |
| `ties_head_to_tail_fixed_head_full_seed0_20261104T235759Z` | TIES | Head-to-Tail | TDDI | 0.854148 | 0.208110 | **0.231388** | 0.837363 | 0.460646 | n.a. | n.a. | [TIES results](TIES_HEAD_TO_TAIL_FIXED_HEAD_RESULTS.md) |

## Quy ước

- `Final`: kết quả ở task boundary cuối, trên toàn bộ test samples của các class đã thấy.
- `Mean Task Macro-F1`: trung bình Macro-F1 của tám task boundary, mỗi boundary có trọng số bằng nhau.
- `BWT`: trung bình backward transfer trên các task cũ; càng gần 0 hoặc dương càng tốt.
- `Forgetting`: best Macro-F1 trước đây trừ Macro-F1 cuối; càng thấp càng tốt.
- Một run gồm ba ensemble member nhưng vẫn chỉ là một experiment seed.

