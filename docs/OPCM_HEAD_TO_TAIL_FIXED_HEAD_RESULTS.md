# OPCM — Head-to-Tail — Fixed Head Results

## Nguồn kết quả

- Run: `opcm_head_to_tail_fixed_head_full_seed0_20261108T141808Z`
- Gói kết quả: `opcm_head_to_tail_fixed_head_full_seed0_20261108T141808Z_metrics_review.zip`
- SHA256 ZIP: `9C3AE42537192A0BEECF104FDDF497870641D41C41508EBBEEBF02D4715C7E54`
- Trạng thái: hoàn thành 8 boundary, 3 member và offline ensemble.

## Cấu hình chính

| Thành phần | Thiết lập |
| --- | --- |
| Method | OPCM |
| Protocol | `ORDERED / head_to_tail` |
| Backbone | `tddi_paper_member` |
| Head | Fixed 178-class head, ownership theo task |
| Task layout | `38, 20, 20, 20, 20, 20, 20, 20` |
| Ensemble | 3 frozen stratified-fold members |
| Experiment seed / fold seed | `0 / 42` |
| Preprocessing | `task0_standard_frozen` |
| Merge scope | Chỉ learned backbone parameters; head rows được stitch theo ownership |
| Expert schedule | Expert hiện tại được train rồi merge tuần tự; không dùng future expert |
| Expert initialization | Các expert được tạo từ shared task-0 initialization |
| OPCM recipe | `alpha=0.5`, `scale=1.0` |
| Projection | Full SVD, loại diagonal và dominant subspace |
| Adaptive scaling | Theo trung bình norm của task vector đã thấy |
| Optimizer | AdamW, learning rate `0.001`, weight decay `0.0001` |
| Batch | Microbatch `64`, effective batch `1024` |
| Training | Tối đa 30 epoch/task, patience `5` |

## Kết quả chính — offline ensemble

Các metric là full-coverage test metric tại boundary cuối (task 7).

| Phạm vi | Accuracy | Balanced Accuracy | Macro-F1 | Weighted-F1 |
| --- | ---: | ---: | ---: | ---: |
| Task 7 / kết quả cuối | 0.437949 | 0.070953 | **0.082172** | 0.425959 |
| Trung bình Macro-F1 của 8 boundary | — | — | **0.278576** | — |

## Diễn biến theo boundary

| Boundary | Số class đã thấy | Test samples | Accuracy | Balanced Accuracy | Macro-F1 | Weighted-F1 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 38 | 164,910 | 0.940143 | 0.947025 | 0.947408 | 0.939943 |
| 1 | 58 | 169,606 | 0.616924 | 0.389359 | 0.407236 | 0.626038 |
| 2 | 78 | 171,826 | 0.527516 | 0.231488 | 0.252173 | 0.533429 |
| 3 | 98 | 172,875 | 0.491413 | 0.163494 | 0.183002 | 0.492123 |
| 4 | 118 | 173,302 | 0.471703 | 0.124324 | 0.143545 | 0.469210 |
| 5 | 138 | 173,476 | 0.455608 | 0.098124 | 0.116175 | 0.450706 |
| 6 | 158 | 173,569 | 0.447280 | 0.081997 | 0.096900 | 0.436527 |
| 7 | 178 | 173,614 | 0.437949 | 0.070953 | 0.082172 | 0.425959 |

## Member cuối

| Member | Final Macro-F1 | Mean task Macro-F1 |
| ---: | ---: | ---: |
| 0 | 0.066849 | 0.253067 |
| 1 | 0.063989 | 0.246073 |
| 2 | 0.060874 | 0.241265 |
| Trung bình member | 0.063904 | 0.246802 |
| Offline ensemble | **0.082172** | **0.278576** |

Ensemble tăng khoảng `+0.018268` Macro-F1 so với trung bình ba member, nhưng mức tuyệt đối vẫn thấp.

## Nhận xét

1. OPCM giảm rất mạnh từ Macro-F1 `0.947408` ở task 0 xuống `0.082172` ở task 7, tức giảm khoảng `0.865236`.
2. Riêng bước task 6→7 giảm từ `0.096900` xuống `0.082172` (`-0.014727`), nên vấn đề chính là suy giảm tích lũy qua toàn bộ protocol chứ không chỉ một bước cuối.
3. Accuracy `0.437949` và Weighted-F1 `0.425959` cao hơn rất nhiều Macro-F1. Balanced Accuracy chỉ `0.070953`; model vẫn dự đoán thiên mạnh về các class head và gần như bỏ qua nhiều class tail.
4. Projection OPCM làm giảm xung đột task vector theo công thức của method, nhưng không cung cấp replay hoặc class-balancing. Vì vậy nó không giải quyết được intra-task long-tail của dataset.
5. Bản thân OPCM là merge trajectory, không phải replay-based CL trajectory chuẩn; package không có BWT/forgetting chuẩn để so sánh trực tiếp với EWC, ER hoặc A-GEM.

## Kết luận

Trong pipeline hiện tại, OPCM chưa phù hợp với mục tiêu tối ưu Macro-F1. Nó bảo toàn được một phần cấu trúc backbone qua phép chiếu nhưng chất lượng các class mid/tail suy giảm nghiêm trọng khi mở rộng tới 178 lớp. Không nên chọn OPCM làm best configuration hiện tại.

