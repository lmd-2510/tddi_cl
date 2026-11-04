# UPCL — Head-to-Tail — Fixed Head Results

## Nguồn kết quả

- Run: `upcl_head_to_tail_fixed_head_full_seed0_20261103T165325Z`
- Gói kết quả: `upcl_head_to_tail_fixed_head_full_seed0_20261103T165325Z_metrics_review.zip`
- SHA256 ZIP: `CE969863A667E52F58FA9590893BF747CE7E8F2FEBC6BB0121BCECE021B86D94`
- Trạng thái: hoàn thành 8 task, 3 member và offline ensemble.

## Cấu hình chính

| Thành phần | Thiết lập |
| --- | --- |
| Method | UPCL |
| Protocol | `ORDERED / head_to_tail` |
| Backbone | `tddi_paper_member` |
| Head | Fixed 178-class head, mở mask theo task |
| Task layout | `38, 20, 20, 20, 20, 20, 20, 20` |
| Ensemble | 3 frozen stratified-fold members |
| Experiment seed / fold seed | `0 / 42` |
| Preprocessing | `task0_standard_frozen` |
| Buffer | Equal-class fixed budget, `27,778/member` |
| Replay policy | `stratified_fraction_v1`, UPCL replay fraction `0` |
| UPCL temperature | `0.07` |
| Loss | UPCL / iCaRL-style replay policy |
| Optimizer | AdamW, learning rate `0.001`, weight decay `0.0001` |
| Batch | Microbatch `64`, effective batch `1024` |
| Training | Tối đa 30 epoch/task, patience `5` |

## Kết quả chính — offline ensemble

| Phạm vi | Accuracy | Balanced Accuracy | Macro-F1 | Weighted-F1 | Coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| Task 7 / kết quả cuối | 0.322388 | 0.653767 | **0.219061** | 0.381671 | 1.000000 |
| Trung bình 8 task | 0.414806 | 0.744554 | **0.389315** | 0.443500 | 1.000000 |

Threshold metrics trùng với full-coverage metrics vì coverage luôn bằng `1.0`.

## Diễn biến theo task

| Task | Nhóm | Class mới | Test mẫu mới | Tổng test đã thấy | Accuracy | Balanced Accuracy | Macro-F1 | Weighted-F1 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | Head | 38 | 164,910 | 164,910 | 0.534449 | 0.798542 | 0.673722 | 0.440291 |
| 1 | Mid | 20 | 4,696 | 169,606 | 0.512594 | 0.813017 | 0.531951 | 0.513106 |
| 2 | Mid | 20 | 2,220 | 171,826 | 0.464115 | 0.801539 | 0.430849 | 0.497975 |
| 3 | Mid | 20 | 1,049 | 172,875 | 0.419980 | 0.739717 | 0.377302 | 0.471029 |
| 4 | Mid | 20 | 427 | 173,302 | 0.383365 | 0.730216 | 0.339732 | 0.440801 |
| 5 | Mid | 20 | 174 | 173,476 | 0.352343 | 0.713246 | 0.287054 | 0.412554 |
| 6 | Mid | 20 | 93 | 173,569 | 0.329212 | 0.706386 | 0.254851 | 0.390569 |
| 7 | Tail | 20 | 45 | 173,614 | 0.322388 | 0.653767 | 0.219061 | 0.381671 |

## Continual-learning metrics

| Metric | Giá trị |
| --- | ---: |
| Mean BWT trên task cũ | **0.147812** |
| Mean forgetting trên task cũ | **0.029976** |
| Member task-forgetting, mean ± std | 0.032558 ± 0.003153 |
| Member class-forgetting cuối, mean ± std | 0.086812 ± 0.002065 |
| Zero-F1 classes cuối | 32 / 29 / 32 |

UPCL có BWT dương và forgetting trung bình thấp. Tuy nhiên, điều này không đồng nghĩa với Macro-F1 cuối cao: model giữ được một phần kiến thức cũ nhưng học các class tail mới rất kém.

## Ensemble diversity

- Pooled mean pairwise disagreement: `0.474970`.
- Disagreement tăng từ `0.270503` ở task 0 lên `0.566049` ở task 7.
- Diversity cao hơn ở các task sau nhưng không chuyển thành Macro-F1 cao hơn; các member cùng gặp khó khăn trên các class tail.

## Nhận xét

1. UPCL đạt Balanced Accuracy cuối `0.653767`, tốt hơn nhiều so với Accuracy `0.322388`. Điều này cho thấy recall giữa các class tương đối cân bằng hơn, nhưng precision thấp làm Macro-F1 chỉ còn `0.219061`.
2. Macro-F1 giảm từ `0.673722` ở task 0 xuống `0.219061` ở task 7. Số test mẫu của class mới giảm cực mạnh: từ `164,910` ở task 0 xuống chỉ `45` ở task 7.
3. Task 6→7 giảm `0.035790`, nhỏ hơn các bước suy giảm trước đó. Vấn đề chính là long-tail và độ khó học class mới, không chỉ là catastrophic forgetting.
4. UPCL có ưu điểm rõ về ổn định: BWT dương `0.147812` và forgetting thấp `0.029976`. Tuy nhiên, replay/class-uniform buffer hiện tại chưa đủ để nâng chất lượng dự đoán tail lên mức Macro-F1 tốt.

## Kết luận

UPCL là method giữ kiến thức cũ tốt nhất trong hai run được thêm lần này, nhưng chưa giải quyết được mục tiêu Macro-F1 cuối. Kết quả phù hợp với nhận định rằng stability và performance trên class tail là hai mục tiêu khác nhau: UPCL cải thiện stability, còn intra-task imbalance vẫn là nút thắt chính.

