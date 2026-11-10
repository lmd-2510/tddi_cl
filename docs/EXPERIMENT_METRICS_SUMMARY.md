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
| `opcm_head_to_tail_fixed_head_full_seed0_20261108T141808Z` | OPCM | Head-to-Tail | TDDI | 0.437949 | 0.070953 | **0.082172** | 0.425959 | 0.278576 | n.a. | n.a. | [OPCM results](OPCM_HEAD_TO_TAIL_FIXED_HEAD_RESULTS.md) |
| `magmax_head_to_tail_fixed_head_full_seed0_20261108T182538Z` | MagMax | Head-to-Tail | TDDI | 0.585074 | 0.115060 | **0.132647** | 0.591144 | 0.272200 | n.a. | n.a. | [MagMax results](MAGMAX_HEAD_TO_TAIL_FIXED_HEAD_RESULTS.md) |
| `ewc_ddinet_head_to_tail_fixed_head_full_seed0_20261109T083845Z` | EWC | Head-to-Tail | DDINet | 0.870310 | 0.251004 | **0.223290** | 0.873725 | 0.474474 | -0.017201 | 0.017499 | [EWC + DDINet results](EWC_DDINET_HEAD_TO_TAIL_FIXED_HEAD_RESULTS.md) |
| `agem_ddinet_head_to_tail_fixed_head_full_seed0_20261109T113318Z` | A-GEM | Head-to-Tail | DDINet | 0.867741 | 0.304916 | **0.256433** | 0.880240 | 0.511796 | -0.039699 | 0.069837 | [A-GEM + DDINet results](AGEM_DDINET_HEAD_TO_TAIL_FIXED_HEAD_RESULTS.md) |
| `er_ddinet_head_to_tail_fixed_head_full_seed0_20261109T140200Z` | ER | Head-to-Tail | DDINet | 0.934694 | 0.554412 | **0.521134** | 0.936575 | 0.764030 | -0.097410 | 0.097410 | [ER + DDINet results](ER_DDINET_HEAD_TO_TAIL_FIXED_HEAD_RESULTS.md) |
| `ider_ddinet_head_to_tail_fixed_head_full_seed0_20261109T173342Z` | IDER | Head-to-Tail | DDINet | 0.820953 | 0.417082 | **0.301342** | 0.854357 | 0.548116 | -0.015086 | 0.030465 | [IDER + DDINet results](IDER_DDINET_HEAD_TO_TAIL_FIXED_HEAD_RESULTS.md) |
| `upcl_ddinet_head_to_tail_fixed_head_full_seed0_20261110T081853Z` | UPCL | Head-to-Tail | DDINet (256-d UPCL) | 0.476638 | 0.560851 | **0.284112** | 0.559114 | 0.518095 | 0.157033 | 0.028480 | [UPCL + DDINet results](UPCL_DDINET_HEAD_TO_TAIL_FIXED_HEAD_RESULTS.md) |
| `task_arithmetic_ddinet_head_to_tail_fixed_head_full_seed0_20261110T151047Z` | Task Arithmetic | Head-to-Tail | DDINet | 0.891760 | 0.206883 | **0.202022** | 0.871610 | 0.432453 | n.a. | n.a. | [Task Arithmetic + DDINet results](TASK_ARITHMETIC_DDINET_HEAD_TO_TAIL_FIXED_HEAD_RESULTS.md) |
| `ties_ddinet_head_to_tail_fixed_head_full_seed0_20261110T174458Z` | TIES | Head-to-Tail | DDINet | 0.889473 | 0.206924 | **0.203080** | 0.869786 | 0.481333 | n.a. | n.a. | [TIES + DDINet](#ties--ddinet--20261110t174458z) |

## Quy ước

- `Final`: kết quả ở task boundary cuối, trên toàn bộ test samples của các class đã thấy.
- `Mean Task Macro-F1`: trung bình Macro-F1 của tám task boundary, mỗi boundary có trọng số bằng nhau.
- `BWT`: trung bình backward transfer trên các task cũ; càng gần 0 hoặc dương càng tốt.
- `Forgetting`: best Macro-F1 trước đây trừ Macro-F1 cuối; càng thấp càng tốt.
- Một run gồm ba ensemble member nhưng vẫn chỉ là một experiment seed.
- Với EWC + DDINet, task-group Macro-F1 của task 3–7 bằng 0 ngay tại boundary vừa học. Đọc BWT/forgetting cùng khả năng học lớp mới trong báo cáo chi tiết.
- UPCL + DDINet dùng hidden cuối 256 chiều để đáp ứng prototype 178 lớp. Các run DDINet còn lại trong bảng dùng hidden cuối 128 chiều.
- A-GEM + DDINet dùng thực tế khoảng 15,2 nghìn exemplar/member, thấp hơn budget tối đa 27.778 do quota theo task và số mẫu có sẵn. ER, IDER và UPCL + DDINet dùng đủ 27.778 exemplar/member ở boundary cuối.
- F1 trong ma trận task-group được tính sau khi lọc mẫu theo nhãn thật của từng nhóm. Trung bình các F1 này có thể khác Macro-F1 toàn cục vì tập false positive được tính khác nhau.

## TIES + DDINet — 20261110T174458Z

### Cấu hình

- Protocol `ORDERED / head_to_tail`, task layout `38, 20, 20, 20, 20, 20, 20, 20`, fixed 178-class head và ownership theo task.
- Ba stratified-fold member, experiment seed `0`, fold seed `42`.
- DDINet dùng Morgan 1024-bit/drug, ordered concatenation thành 2048 chiều, hidden `1024 → 512 → 256 → 128`, ReLU, BatchNorm, dropout `0.3`, preprocessing `raw_identity`.
- Task 0 là base. Expert task 1–7 fine-tune độc lập từ checkpoint task 0. Mỗi boundary chỉ merge các expert của task đã thấy.
- TIES giữ density `0.2`, scale `1.0`, mass sign election và disjoint mean. Merge learned backbone parameters, ghép head rows theo ownership, gộp BatchNorm moments theo số mẫu train của các expert.
- AdamW, learning rate `0.001`, weight decay `0.0001`, microbatch `64`, effective batch `1024`, evaluation batch `512`, tối đa 30 epoch/expert, patience `5`. Batch một mẫu dùng `running_stats_for_singleton_v1`.

### Kết quả

Các số sau dùng toàn bộ test của các lớp đã thấy. Boundary cuối gồm 178 lớp và 173.614 mẫu, coverage 100%.

| Boundary | Accuracy | Balanced Accuracy | Macro-F1 | Weighted-F1 |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.967958 | 0.974745 | 0.971231 | 0.967909 |
| 1 | 0.928528 | 0.872946 | 0.838795 | 0.929496 |
| 2 | 0.903199 | 0.576255 | 0.583168 | 0.895633 |
| 3 | 0.883893 | 0.412903 | 0.421034 | 0.871069 |
| 4 | 0.881225 | 0.326480 | 0.330108 | 0.865456 |
| 5 | 0.883874 | 0.272536 | 0.271362 | 0.866215 |
| 6 | 0.887186 | 0.234842 | 0.231885 | 0.868307 |
| 7 | 0.889473 | 0.206924 | 0.203080 | 0.869786 |

| Phạm vi | Final Macro-F1 | Mean boundary Macro-F1 |
| --- | ---: | ---: |
| Member 0 | 0.206523 | 0.457943 |
| Member 1 | 0.204703 | 0.461289 |
| Member 2 | 0.203508 | 0.464443 |
| Trung bình member | 0.204911 | 0.461225 |
| Offline ensemble | **0.203080** | **0.481333** |

- Ensemble thấp hơn Macro-F1 trung bình member `0.001831`. Accuracy cao đi cùng Macro-F1 và Balanced Accuracy thấp, cho thấy chất lượng giữa các lớp còn chênh lệch lớn.
- Macro-F1 giảm `0.028805`, tương đương 2,88 điểm phần trăm từ task 6 sang 7.
- So với Task Arithmetic–DDINet, Final Macro-F1 tăng `0.001058` và Mean boundary Macro-F1 tăng `0.048880`. Lợi ích ở các boundary đầu chưa duy trì được tới boundary cuối.
- Gói này không xuất BWT, forgetting và task-group matrix, nên bảng so sánh ghi `n.a.` cho hai CL metric đó. Chưa xác định được lớp quên nhiều nhất từ các bảng có sẵn.

### Nguồn và kiểm tra

- ZIP `ties_ddinet_head_to_tail_fixed_head_full_seed0_20261110T174458Z_metrics_review.zip`.
- SHA256 ZIP `ab7170ccd6d3c3fae75d89122fc05a96e61ef35eff3cd8c3ed7586e70fbbec9f`. Đã kiểm tra CRC toàn bộ ZIP và đối chiếu summary với đủ 8 dòng `ensemble_metrics.csv`.
- Các nguồn chính gồm `ensemble_summary.json`, `ensemble_metrics.csv`, `member_*/run_summary.json` và `member_0/run_config.json`.
- DDINet giữ các lựa chọn adaptation/provisional về hidden widths và fingerprint recipe. Config M0 ghi commit `914d9c20fdd0e00624f3f59c169f7937faa20504` với `dirty=true`.

