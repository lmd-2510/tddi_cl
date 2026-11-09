# MagMax — Head-to-Tail — Fixed Head Results

## Nguồn kết quả

- Run: `magmax_head_to_tail_fixed_head_full_seed0_20261108T182538Z`
- Gói kết quả: `magmax_head_to_tail_fixed_head_full_seed0_20261108T182538Z_metrics_review.zip`
- SHA256 ZIP: `3B4994FE837332AC1B7E02B3D47B61116D4EEA9BCAC9CC780CC2DCB244A7D404`
- Trạng thái: hoàn thành 8 boundary, 3 member và offline ensemble.

## Cấu hình chính

| Thành phần | Thiết lập |
| --- | --- |
| Method | MagMax |
| Protocol | `ORDERED / head_to_tail` |
| Backbone | `tddi_paper_member` |
| Head | Fixed 178-class head, ownership theo task |
| Task layout | `38, 20, 20, 20, 20, 20, 20, 20` |
| Ensemble | 3 frozen stratified-fold members |
| Experiment seed / fold seed | `0 / 42` |
| Preprocessing | `task0_standard_frozen` |
| Merge scope | Chỉ learned backbone parameters; head rows được stitch theo ownership |
| Expert schedule | Sequential fine-tuning từ task trước rồi MagMax merge |
| Future experts | Không dùng future expert tại boundary trước |
| MagMax recipe | `scale=0.5` |
| Selection | Elementwise maximum absolute task vector |
| Optimizer | AdamW, learning rate `0.001`, weight decay `0.0001` |
| Batch | Microbatch `64`, effective batch `1024` |
| Training | Tối đa 30 epoch/task, patience `5` |

## Kết quả chính — offline ensemble

Các metric là full-coverage test metric tại boundary cuối (task 7).

| Phạm vi | Accuracy | Balanced Accuracy | Macro-F1 | Weighted-F1 |
| --- | ---: | ---: | ---: | ---: |
| Task 7 / kết quả cuối | 0.585074 | 0.115060 | **0.132647** | 0.591144 |
| Trung bình Macro-F1 của 8 boundary | — | — | **0.272200** | — |

## Diễn biến theo boundary

| Boundary | Số class đã thấy | Test samples | Accuracy | Balanced Accuracy | Macro-F1 | Weighted-F1 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 38 | 164,910 | 0.610751 | 0.559115 | 0.603713 | 0.629777 |
| 1 | 58 | 169,606 | 0.598965 | 0.362949 | 0.399520 | 0.610993 |
| 2 | 78 | 171,826 | 0.586134 | 0.261685 | 0.293934 | 0.596565 |
| 3 | 98 | 172,875 | 0.580633 | 0.205860 | 0.233442 | 0.589225 |
| 4 | 118 | 173,302 | 0.581095 | 0.171770 | 0.196517 | 0.588562 |
| 5 | 138 | 173,476 | 0.582547 | 0.147392 | 0.169191 | 0.589305 |
| 6 | 158 | 173,569 | 0.584309 | 0.129225 | 0.148635 | 0.590711 |
| 7 | 178 | 173,614 | 0.585074 | 0.115060 | 0.132647 | 0.591144 |

## Member cuối

| Member | Final Macro-F1 | Mean task Macro-F1 |
| ---: | ---: | ---: |
| 0 | 0.125620 | 0.250101 |
| 1 | 0.141251 | 0.253214 |
| 2 | 0.127034 | 0.240971 |
| Trung bình member | 0.131301 | 0.248095 |
| Offline ensemble | **0.132647** | **0.272200** |

Ensemble chỉ tăng khoảng `+0.001346` Macro-F1 so với trung bình ba member ở task 7.

## So sánh với các merging run trước

| Method | Final Accuracy | Final Balanced Accuracy | Final Macro-F1 | Final Weighted-F1 |
| --- | ---: | ---: | ---: | ---: |
| Task Arithmetic | 0.819018 | 0.184912 | 0.208722 | 0.802104 |
| TIES | 0.854148 | 0.208110 | 0.231388 | 0.837363 |
| MagMax | 0.585074 | 0.115060 | **0.132647** | 0.591144 |
| OPCM | 0.437949 | 0.070953 | 0.082172 | 0.425959 |

MagMax tốt hơn OPCM khoảng `+0.050474` Macro-F1 ở task 7, nhưng thấp hơn TIES `0.098741` và Task Arithmetic `0.076075`.

## Nhận xét

1. MagMax tốt hơn OPCM ở boundary cuối trên cả Accuracy, Balanced Accuracy, Macro-F1 và Weighted-F1.
2. Tuy nhiên Macro-F1 vẫn giảm từ `0.603713` ở task 0 xuống `0.132647` ở task 7, tức giảm khoảng `0.471067`.
3. Bước task 6→7 giảm thêm `0.015988` Macro-F1 (`0.148635 → 0.132647`). Sụt giảm cuối không phải nguyên nhân duy nhất; forgetting/thiên lệch đã tích lũy từ các boundary trước.
4. Chênh lệch lớn giữa Weighted-F1 `0.591144` và Macro-F1 `0.132647`, cùng Balanced Accuracy `0.115060`, cho thấy MagMax vẫn ưu tiên các class head có nhiều sample.
5. Mean task Macro-F1 `0.272200` thấp hơn OPCM `0.278576`, dù MagMax có final Macro-F1 cao hơn. Điều này cho thấy MagMax cải thiện boundary cuối nhưng không ổn định hơn trên toàn bộ trajectory.
6. MagMax là sequential model-merging run, không có replay buffer hoặc loss chống forgetting như ER/EWC; BWT/forgetting chuẩn không được báo cáo và không nên so sánh trực tiếp với các method CL replay/regularization.

## Kết luận

MagMax là merging method tốt nhất trong hai run OPCM/MagMax ở task 7, nhưng vẫn chưa cạnh tranh được với TIES, Task Arithmetic hoặc ER về Macro-F1. Nút thắt chính vẫn là long-tail/intra-task imbalance và việc merge backbone không đủ để bảo vệ recall trung bình của 178 class.

