# EWC + DDINet — Head-to-Tail — Fixed Head Results

## Nguồn kết quả

- Run `ewc_ddinet_head_to_tail_fixed_head_full_seed0_20261109T083845Z`.
- Gói kết quả `ewc_ddinet_head_to_tail_fixed_head_full_seed0_20261109T083845Z_metrics_review.zip`.
- SHA256 ZIP `b4adfae4b2eb9050fc9b065ed2e7ac1b340fda1233221a1a196bb3058cf6d65a`.
- Cả ba member đã hoàn thành task 7, với offline ensemble cho đủ 8 boundary.
- Đã kiểm tra kích thước và SHA256 của 131 file được liệt kê trong `PACKAGE_MANIFEST.json`, tất cả đều khớp.

## Cấu hình chính

| Thành phần | Thiết lập |
| --- | --- |
| Method | EWC, CE cộng cumulative EWC penalty |
| Protocol | `ORDERED / head_to_tail` |
| Backbone | `ddinet_multiclass_cil` |
| Input | Morgan 1024-bit mỗi drug, ordered concatenation thành 2048 chiều |
| Fingerprint recipe | Radius 2, chirality false, theo cấu hình adapter hiện có |
| Hidden layers | `1024 → 512 → 256 → 128` |
| Activation / normalization | ReLU / BatchNorm trong backbone |
| Dropout | `0.3` |
| Head | Fixed linear head 178 lớp, mask các lớp chưa thấy |
| Task layout | `38, 20, 20, 20, 20, 20, 20, 20` |
| Ensemble | 3 frozen stratified-fold members, chạy tuần tự |
| Experiment seed / fold seed | `0 / 42` |
| Preprocessing | `raw_identity`, model dùng Morgan fingerprint cache |
| EWC coefficient | `lambda=1000` |
| Optimizer | AdamW, learning rate `0.001`, weight decay `0.0001` |
| Batch | Microbatch `64`, effective batch `1024`, evaluation batch `512` |
| Training / selection | Tối đa 30 epoch/task, patience `5`, chọn theo all-seen validation Macro-F1 |
| Exemplar memory | `0` |
| Ensemble evaluation | Mean probability trên test, ghép held-out folds cho OOF |

Manifest model đánh dấu hidden widths, ordered concatenation, radius 2 và chirality false là lựa chọn adaptation/provisional. Đây là DDINet được điều chỉnh cho multiclass CIL theo recipe của repo.

Cả ba member có cùng lịch early stopping. Task 0 chạy 30 epoch và chọn epoch 30. Task 1 chạy 7 epoch và chọn epoch 2. Task 2–7 đều chạy 6 epoch và chọn epoch 1.

## Kết quả chính — offline ensemble

Metric chính dùng toàn bộ test samples thuộc các lớp đã thấy, với full coverage.

| Phạm vi | Accuracy | Balanced Accuracy | Macro-F1 | Weighted-F1 |
| --- | ---: | ---: | ---: | ---: |
| Task 7 / kết quả cuối | 0.870310 | 0.251004 | **0.223290** | 0.873725 |
| Trung bình 8 boundary | 0.895055 | 0.502772 | **0.474474** | 0.896056 |

Threshold evaluation ở task 7 có coverage `1.0` và cùng Macro-F1 `0.223290`. Coverage trung bình qua 8 boundary là `0.855122`.

## Diễn biến theo task

| Task | Class mới | Test mẫu mới | Tổng test mẫu đã thấy | Accuracy | Balanced Accuracy | Macro-F1 | Weighted-F1 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 38 | 164,910 | 164,910 | 0.967679 | 0.974033 | 0.970795 | 0.967607 |
| 1 | 20 | 4,696 | 169,606 | 0.910375 | 0.781405 | 0.759695 | 0.914545 |
| 2 | 20 | 2,220 | 171,826 | 0.904729 | 0.576900 | 0.544406 | 0.902039 |
| 3 | 20 | 1,049 | 172,875 | 0.880926 | 0.460586 | 0.419078 | 0.881090 |
| 4 | 20 | 427 | 173,302 | 0.875218 | 0.374862 | 0.336407 | 0.875848 |
| 5 | 20 | 174 | 173,476 | 0.877032 | 0.320933 | 0.288865 | 0.877211 |
| 6 | 20 | 93 | 173,569 | 0.874171 | 0.282450 | 0.253259 | 0.876387 |
| 7 | 20 | 45 | 173,614 | 0.870310 | 0.251004 | 0.223290 | 0.873725 |

Macro-F1 giảm `0.029969`, tương đương khoảng 3,00 điểm phần trăm, từ task 6 sang task 7. Task 7 bổ sung 20 lớp nhưng chỉ có 45 test samples, so với 164.910 test samples của nhóm task 0.

## Kết quả từng member ở task 7

| Member | Accuracy | Balanced Accuracy | Macro-F1 | Weighted-F1 | Zero-F1 classes |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.808305 | 0.228353 | 0.205802 | 0.833802 | 121 |
| 1 | 0.829236 | 0.222840 | 0.205011 | 0.842831 | 120 |
| 2 | 0.797251 | 0.228360 | 0.202641 | 0.818861 | 117 |
| Trung bình member | 0.811597 | 0.226518 | 0.204485 | 0.831831 | — |
| Offline ensemble | **0.870310** | **0.251004** | **0.223290** | **0.873725** | — |

Ensemble tăng Macro-F1 `+0.018805` so với trung bình member, tương đương 1,88 điểm phần trăm. Cả ba member đều có Accuracy và Macro-F1 bằng 0 trên nhóm lớp mới task 7, trên cả validation và test.

## BWT, forgetting và khả năng học lớp mới

| Metric | Giá trị |
| --- | ---: |
| Ensemble mean BWT, old tasks | **-0.017201** |
| Ensemble mean forgetting, old tasks | **0.017499** |
| Member task forgetting, mean ± population std | 0.018968 ± 0.004370 |
| Member class forgetting cuối, mean ± population std | 0.037134 ± 0.004500 |

Ma trận task-group Macro-F1 cho thấy mức học và giữ lại kiến thức rất khác nhau giữa các nhóm.

| Nhóm lớp | Macro-F1 ngay sau khi học | Best Macro-F1 | Macro-F1 tại boundary 7 |
| --- | ---: | ---: | ---: |
| Task 0 | 0.970795 | 0.970795 | 0.915715 |
| Task 1 | 0.591322 | 0.591322 | 0.525341 |
| Task 2 | 0.000000 | 0.002083 | 0.000654 |
| Task 3 | 0.000000 | 0.000000 | 0.000000 |
| Task 4 | 0.000000 | 0.000000 | 0.000000 |
| Task 5 | 0.000000 | 0.000000 | 0.000000 |
| Task 6 | 0.000000 | 0.000000 | 0.000000 |
| Task 7 | 0.000000 | 0.000000 | 0.000000 |

Forgetting trung bình thấp một phần vì nhiều nhóm có F1 bằng 0 từ lúc vừa học, nên mức suy giảm đo được cũng bằng 0. Task 0 và task 1 có forgetting lần lượt `0.055080` và `0.065980`. Các số này cần được đọc cùng khả năng tiếp thu lớp mới.

Một phép phân rã số học giúp giải thích bước 6→7. Nếu F1 của 158 lớp cũ giữ nguyên và 20 lớp mới đều có F1 bằng 0, Macro-F1 sẽ thành `0.253259 × 158 / 178 = 0.224803`. Kết quả thực tế là `0.223290`, thấp hơn giá trị giả định khoảng `0.001513`. Phần lớn mức giảm ở bước này phù hợp với việc bổ sung các lớp mới có F1 bằng 0 vào trung bình macro. Đây là phân rã metric dưới giả định giữ nguyên lớp cũ.

## Đối chiếu với EWC + TDDI

| Backbone | Final Accuracy | Final Balanced Accuracy | Final Macro-F1 | Final Weighted-F1 | BWT | Forgetting |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TDDI, run `20261102T071912Z` | 0.569257 | 0.548442 | 0.377209 | 0.668461 | -0.210327 | 0.217414 |
| DDINet, run hiện tại | 0.870310 | 0.251004 | 0.223290 | 0.873725 | -0.017201 | 0.017499 |

DDINet thấp hơn TDDI `0.153919` Macro-F1, tương đương 15,39 điểm phần trăm. Accuracy và Weighted-F1 của DDINet cao hơn, trong khi recall trung bình trên các lớp thấp hơn rõ rệt. Hai run sử dụng hai representation khác nhau, gồm QSAR descriptors cho TDDI và Morgan fingerprints cho DDINet, cùng kiến trúc và normalization riêng của mỗi backbone.

## Nhận xét và kết luận

1. EWC + DDINet đạt Macro-F1 cuối **0.223290**, với kết quả tốt tập trung ở các nhóm task 0–1.
2. Ma trận task-group ghi nhận khả năng học lớp mới suy giảm từ task 2. Các nhóm task 3–7 có Macro-F1 bằng 0 ở mọi boundary đã được đánh giá.
3. Accuracy `0.870310` và Weighted-F1 `0.873725` chịu ảnh hưởng lớn từ số mẫu của các lớp head. Balanced Accuracy `0.251004` và 117–121 zero-F1 classes/member thể hiện chất lượng chưa đồng đều giữa các lớp.
4. Early stopping chọn epoch 1 cho mọi task từ 2 đến 7. Đây là dấu hiệu cần đối chiếu với cân bằng giữa học lớp mới và giữ lớp cũ. Gói kết quả hiện tại chưa đủ để quy nguyên nhân riêng cho hệ số EWC, representation hoặc tiêu chí chọn checkpoint.
5. Theo mục tiêu Macro-F1, run này thấp hơn EWC + TDDI đã ghi nhận trước đó. BWT/forgetting cần được trình bày cùng điểm từng nhóm để phản ánh đúng tình trạng nhiều lớp mới chưa được học hiệu quả.

## Ghi nhận provenance

`full_manifest.json` ghi policy `reviewed_ddinet_checkpoint_reader_fix_v1`. M0 dùng hash loader trước bản sửa DDINet, M1/M2 dùng hash sau bản sửa. Manifest vẫn lưu hash gốc của từng member. Các số liệu trong tài liệu được lấy trực tiếp từ package sau khi offline ensemble hoàn tất.

