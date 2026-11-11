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
| `magmax_ddinet_head_to_tail_fixed_head_full_seed0_20261110T190322Z` | MagMax | Head-to-Tail | DDINet | 0.792816 | 0.206171 | **0.217054** | 0.784123 | 0.422485 | n.a. | n.a. | [MagMax + DDINet](#magmax--ddinet--20261110t190322z) |
| `opcm_ddinet_head_to_tail_fixed_head_full_seed0_20261111T092045Z` | OPCM | Head-to-Tail | DDINet | 0.167607 | 0.008698 | **0.003996** | 0.066847 | 0.249151 | n.a. | n.a. | [OPCM + DDINet](#opcm--ddinet--20261111t092045z) |

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

## MagMax + DDINet — 20261110T190322Z

### Cấu hình

- Protocol `ORDERED / head_to_tail`, task layout `38, 20, 20, 20, 20, 20, 20, 20`, ba frozen stratified-fold member, experiment seed `0`, fold seed `42`.
- Backbone `ddinet_multiclass_cil` dùng Morgan 1024-bit/drug, ordered concatenation thành 2048 chiều, hidden `1024 → 512 → 256 → 128`, ReLU, BatchNorm và dropout `0.3`. Preprocessing `raw_identity`, policy batch một mẫu `running_stats_for_singleton_v1`.
- Shared fixed head 178 lớp, row ownership theo task. Tại boundary t, đánh giá trên toàn bộ lớp đã thấy tới task t. Merge learned backbone parameters và gộp BatchNorm moments theo số mẫu train từng task.
- Reference là khởi tạo model trước task 0 theo `pretrained_reference=model_initialization_before_task0`.
- MagMax fine-tune tuần tự. Task tiếp theo bắt đầu từ checkpoint expert vừa train. Tại mỗi boundary, chọn từng phần tử task vector có trị tuyệt đối lớn nhất trong các task đã thấy rồi áp dụng scale `0.5` để tạo model đánh giá.
- Loss expert cross-entropy, optimizer AdamW, learning rate `0.001`, weight decay `0.0001`, microbatch `64`, effective batch `1024`, evaluation batch `512`, tối đa 30 epoch/expert, patience `5`, chọn checkpoint theo validation Macro-F1 của task expert.

### Kết quả

Metric chính là mean-probability offline ensemble trên toàn bộ test của các lớp đã thấy, coverage 100%. Boundary cuối có 178 lớp và 173.614 mẫu.

| Boundary | Test samples | Accuracy | Balanced Accuracy | Macro-F1 | Weighted-F1 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 164,910 | 0.919647 | 0.880956 | 0.907050 | 0.918970 |
| 1 | 169,606 | 0.874497 | 0.590598 | 0.602007 | 0.864967 |
| 2 | 171,826 | 0.839401 | 0.434070 | 0.452587 | 0.828178 |
| 3 | 172,875 | 0.819494 | 0.355085 | 0.371695 | 0.808235 |
| 4 | 173,302 | 0.804555 | 0.298571 | 0.313048 | 0.793981 |
| 5 | 173,476 | 0.798203 | 0.260446 | 0.273584 | 0.788720 |
| 6 | 173,569 | 0.795015 | 0.230593 | 0.242850 | 0.785800 |
| 7 | 173,614 | 0.792816 | 0.206171 | 0.217054 | 0.784123 |

| Member / ensemble | Final Accuracy | Final Balanced Accuracy | Final Macro-F1 | Final Weighted-F1 | Mean boundary Macro-F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Member 0 | 0.724055 | 0.183431 | 0.182823 | 0.716417 | 0.379807 |
| Member 1 | 0.694040 | 0.170363 | 0.169329 | 0.688205 | 0.382391 |
| Member 2 | 0.713168 | 0.186231 | 0.195754 | 0.712316 | 0.392656 |
| Trung bình member | 0.710421 | 0.180008 | 0.182635 | 0.705646 | 0.384951 |
| Offline ensemble | 0.792816 | 0.206171 | 0.217054 | 0.784123 | 0.422485 |

### Nhận xét

- Ensemble tăng Macro-F1 `+0.034419` so với trung bình member. Mean boundary Macro-F1 là `0.422485`.
- Từ task 6 sang 7, Macro-F1 giảm `0.025796`, tương đương 2,58 điểm phần trăm. Nhóm task 7 có 20 lớp mới và 45 test mẫu.
- Macro-F1 giảm liên tục từ `0.907050` ở task 0 xuống `0.217054` ở task 7. Accuracy cuối `0.792816` và Balanced Accuracy `0.206171` cho thấy chất lượng dự đoán chênh lệch lớn giữa các lớp.
- Final Macro-F1 cao hơn TIES–DDINet `0.013974` và Task Arithmetic–DDINet `0.015032`. Chênh lệch với ER–DDINet là `-0.304080`.
- Expert task 7 của M0 đạt best validation Macro-F1 `0.850952` khi đánh giá trên nhóm lớp của task expert. F1 này có phạm vi lớp khác với full all-seen test sau merge.
- BWT và forgetting được ghi `n.a.` theo phạm vi metric đã xuất trong ZIP. Bảng task-group hoặc per-class F1 cần bổ sung để định lượng mức quên theo nhóm lớp.

### Nguồn và kiểm tra

- ZIP `magmax_ddinet_head_to_tail_fixed_head_full_seed0_20261110T190322Z_metrics_review.zip`, SHA256 `68501867f53386d47470efafef1f487ccbde5a8b29d50e707fc79c525bf81a0d`.
- Đã kiểm tra CRC toàn bộ ZIP, đối chiếu summary với 8 dòng `ensemble_metrics.csv`, đủ ba `run_summary.json` và `merged_metrics.csv`. Model, recipe, task-file hash, fingerprint manifest hash và implementation hashes khớp giữa ba member. Các merge audit chỉ liệt kê expert tới boundary tương ứng và ghi `future_experts_used=false`.
- Nguồn gồm `member_*/run_config.json`, `expert_epochs.json`, `merged/task_*/merge_audit.json`, các summary và CSV metric trong gói.
- Config M0 ghi commit `91d1e6d6442b209795f6943967b5505709ac4f5a`, `dirty=true`. Manifest DDINet đánh dấu hidden widths và một số lựa chọn fingerprint là adaptation/provisional. Khi tái lập cần đối chiếu implementation hashes cùng cấu hình đã lưu.

## OPCM + DDINet — 20261111T092045Z

### Cấu hình

- Protocol `ORDERED / head_to_tail`, task layout `38, 20, 20, 20, 20, 20, 20, 20`, ba frozen stratified-fold member, experiment seed `0`, fold seed `42`.
- Backbone `ddinet_multiclass_cil` dùng Morgan 1024-bit/drug, ordered concatenation thành 2048 chiều, hidden `1024 → 512 → 256 → 128`, ReLU, BatchNorm và dropout `0.3`. Preprocessing `raw_identity`, policy batch một mẫu `running_stats_for_singleton_v1`.
- Shared fixed head 178 lớp, row ownership theo task. Tại boundary t, đánh giá trên toàn bộ lớp đã thấy tới task t. Merge learned backbone parameters và gộp BatchNorm moments theo số mẫu train từng task.
- Reference là khởi tạo model trước task 0 theo `pretrained_reference=model_initialization_before_task0`.
- OPCM train từng expert từ cùng reference ban đầu. Sau khi train task t, merge ngay expert đó vào model đã gộp, đánh giá boundary t rồi mới train task tiếp theo. Recipe ghi `alpha=0.5`, scale `1.0` và adaptive scaling theo norm task vector.
- Loss expert cross-entropy, optimizer AdamW, learning rate `0.001`, weight decay `0.0001`, microbatch `64`, effective batch `1024`, evaluation batch `512`, tối đa 30 epoch/expert, patience `5`, chọn checkpoint theo validation Macro-F1 của task expert.

### Kết quả

Metric chính là mean-probability offline ensemble trên toàn bộ test của các lớp đã thấy, coverage 100%. Boundary cuối có 178 lớp và 173.614 mẫu.

| Boundary | Test samples | Accuracy | Balanced Accuracy | Macro-F1 | Weighted-F1 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 164,910 | 0.967958 | 0.974745 | 0.971231 | 0.967909 |
| 1 | 169,606 | 0.916483 | 0.611559 | 0.607416 | 0.903636 |
| 2 | 171,826 | 0.730489 | 0.253220 | 0.270039 | 0.706585 |
| 3 | 172,875 | 0.471300 | 0.095048 | 0.100074 | 0.436150 |
| 4 | 173,302 | 0.276656 | 0.031334 | 0.025628 | 0.178793 |
| 5 | 173,476 | 0.221195 | 0.017231 | 0.008894 | 0.102550 |
| 6 | 173,569 | 0.191797 | 0.012008 | 0.005927 | 0.082626 |
| 7 | 173,614 | 0.167607 | 0.008698 | 0.003996 | 0.066847 |

| Member / ensemble | Final Accuracy | Final Balanced Accuracy | Final Macro-F1 | Final Weighted-F1 | Mean boundary Macro-F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Member 0 | 0.166392 | 0.018907 | 0.004787 | 0.081434 | 0.252176 |
| Member 1 | 0.146520 | 0.010210 | 0.003778 | 0.059270 | 0.243301 |
| Member 2 | 0.176587 | 0.009646 | 0.003261 | 0.059981 | 0.239864 |
| Trung bình member | 0.163167 | 0.012921 | 0.003942 | 0.066895 | 0.245114 |
| Offline ensemble | 0.167607 | 0.008698 | 0.003996 | 0.066847 | 0.249151 |

### Nhận xét

- Ensemble tăng Macro-F1 `+0.000054` so với trung bình member. Mean boundary Macro-F1 là `0.249151`.
- Từ task 6 sang 7, Macro-F1 giảm `0.001931`, tương đương 0,19 điểm phần trăm. Nhóm task 7 có 20 lớp mới và 45 test mẫu.
- Macro-F1 giảm từ `0.971231` ở task 0 xuống `0.607416` ở task 1, `0.270039` ở task 2 và `0.003996` ở task 7. Suy giảm đã diễn ra qua nhiều boundary trước task cuối.
- Ba member đều có Final Macro-F1 dưới `0.005`. Cả Accuracy, Balanced Accuracy và Weighted-F1 cuối đều thấp cùng Macro-F1.
- Audit M0 tại boundary 7 ghi `lambda_t=5.914409`, average task-vector norm `31.718715`. Task-vector norm task 0 là `185.626188`, còn task 1–7 nằm trong khoảng `6.015616–14.168915`. Các giá trị mô tả mức chênh lệch độ lớn vector giữa các task trong run.
- Expert task 7 của M0 đạt best validation Macro-F1 `0.854167` trên nhóm lớp task expert. Full all-seen test sau merge có Macro-F1 `0.004787` ở cùng member. Việc xác định nguyên nhân cần thêm đánh giá expert và merged model trên cùng tập mẫu và cùng phạm vi lớp.
- BWT và forgetting được ghi `n.a.` theo phạm vi metric đã xuất trong ZIP. Bảng task-group hoặc per-class F1 cần bổ sung để định lượng mức quên theo nhóm lớp.

### Nguồn và kiểm tra

- ZIP `opcm_ddinet_head_to_tail_fixed_head_full_seed0_20261111T092045Z_metrics_review.zip`, SHA256 `3fa68f447c5d48ac0dda8d88b5d18a6f3569e8ac5a39d7e94f3e15ee570ad4f2`.
- Đã kiểm tra CRC toàn bộ ZIP, đối chiếu summary với 8 dòng `ensemble_metrics.csv`, đủ ba `run_summary.json` và `merged_metrics.csv`. Model, recipe, task-file hash, fingerprint manifest hash và implementation hashes khớp giữa ba member. Các merge audit chỉ liệt kê expert tới boundary tương ứng và ghi `future_experts_used=false`.
- Nguồn gồm `member_*/run_config.json`, `expert_epochs.json`, `merged/task_*/merge_audit.json`, các summary và CSV metric trong gói.
- Config M0 ghi commit `91d1e6d6442b209795f6943967b5505709ac4f5a`, `dirty=true`. Manifest DDINet đánh dấu hidden widths và một số lựa chọn fingerprint là adaptation/provisional. Khi tái lập cần đối chiếu implementation hashes cùng cấu hình đã lưu.

