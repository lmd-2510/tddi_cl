# ER + DDINet — Head-to-Tail — Fixed Head Results

## Nguồn kết quả

- Run `er_ddinet_head_to_tail_fixed_head_full_seed0_20261109T140200Z`.
- Gói kết quả `er_ddinet_head_to_tail_fixed_head_full_seed0_20261109T140200Z_metrics_review.zip`.
- SHA256 ZIP `1ceaf99492bc92a466361168ce8af20f1f886fba71562297d6cf8a5a976a49f2`.
- Ba member hoàn thành task 7, offline ensemble đủ 8 boundary.
- Đã đối chiếu kích thước và SHA256 của 131 file trong `PACKAGE_MANIFEST.json`, tất cả khớp.

## Cấu hình chính

| Thành phần | Thiết lập |
| --- | --- |
| Method | ER |
| Protocol | `ORDERED / head_to_tail` |
| Backbone | `ddinet_multiclass_cil` |
| Input | Morgan 1024-bit mỗi drug, ordered concatenation thành 2048 chiều |
| Fingerprint recipe | Radius 2, chirality false theo adapter của repo |
| Hidden layers | `1024 → 512 → 256 → 128` |
| Activation / normalization / dropout | ReLU / BatchNorm / `0.3` |
| Head | Fixed 178-class head, mask lớp chưa thấy |
| Task layout | `38, 20, 20, 20, 20, 20, 20, 20` |
| Ensemble | 3 frozen stratified-fold members, mean probability trên test |
| Experiment seed / fold seed | `0 / 42` |
| Preprocessing | `raw_identity`, model nhận Morgan fingerprint cache |
| Loss | Cross-entropy trên current và replay |
| Memory | `reservoir_uniform_v1`, lấy mẫu đều trên stream mẫu đã thấy |
| Budget / sử dụng cuối | 27.778 exemplar/member, 83.334 cho ba member |
| Replay | Current/replay 1:1, uniform sampling with replacement |
| Sampler | `er_uniform_reservoir_equal_batch_v1`, repeat cap `1` |
| Optimizer | AdamW, learning rate `0.001`, weight decay `0.0001` |
| Batch | Microbatch `64`, effective batch `1024`, evaluation batch `512` |
| Training / selection | Tối đa 30 epoch/task, patience `5`, chọn theo all-seen validation Macro-F1 |

Manifest DDINet đánh dấu hidden widths, ordered concatenation, radius và chirality là các lựa chọn adaptation/provisional. Báo cáo mô tả cấu hình thực tế được lưu trong run.

## Kết quả chính — offline ensemble

Các metric sau dùng toàn bộ test samples của lớp đã thấy, coverage 100%. Boundary 7 gồm 178 lớp và 173.614 mẫu.

| Phạm vi | Accuracy | Balanced Accuracy | Macro-F1 | Weighted-F1 |
| --- | ---: | ---: | ---: | ---: |
| Task 7 / kết quả cuối | 0.934694 | 0.554412 | **0.521134** | 0.936575 |
| Trung bình 8 boundary | 0.941861 | 0.821064 | **0.764030** | 0.944061 |

Bảng threshold riêng ở task 7 có Macro-F1 `0.396415` với coverage `0.841505`. Bảng so sánh chung sử dụng full-test Macro-F1 `0.521134`.

## Diễn biến theo task

| Boundary | Class đã thấy | Test mẫu mới | Tổng test đã thấy | Accuracy | Balanced Accuracy | Macro-F1 | Weighted-F1 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 38 | 164,910 | 164,910 | 0.967316 | 0.974170 | 0.971211 | 0.967230 |
| 1 | 58 | 4,696 | 169,606 | 0.951022 | 0.963525 | 0.912526 | 0.951965 |
| 2 | 78 | 2,220 | 171,826 | 0.945579 | 0.962330 | 0.883107 | 0.947299 |
| 3 | 98 | 1,049 | 172,875 | 0.933322 | 0.919777 | 0.815748 | 0.938673 |
| 4 | 118 | 427 | 173,302 | 0.932886 | 0.839332 | 0.754200 | 0.937165 |
| 5 | 138 | 174 | 173,476 | 0.936129 | 0.727751 | 0.668213 | 0.937855 |
| 6 | 158 | 93 | 173,569 | 0.933940 | 0.627217 | 0.586100 | 0.935726 |
| 7 | 178 | 45 | 173,614 | 0.934694 | 0.554412 | 0.521134 | 0.936575 |

Macro-F1 giảm `0.064966`, tương đương 6,50 điểm phần trăm từ task 6 sang 7. Nhóm task 7 bổ sung 20 lớp với 45 test mẫu.

## Kết quả từng member ở task 7

| Member | Accuracy | Balanced Accuracy | Macro-F1 | Weighted-F1 | Zero-F1 classes |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.905659 | 0.513724 | 0.468846 | 0.917070 | 59 |
| 1 | 0.893563 | 0.468110 | 0.436855 | 0.909558 | 69 |
| 2 | 0.898724 | 0.483268 | 0.435731 | 0.913066 | 65 |
| Trung bình member | 0.899315 | 0.488367 | 0.447144 | 0.913231 | — |
| Offline ensemble | **0.934694** | **0.554412** | **0.521134** | **0.936575** | — |

Chênh lệch ensemble so với Macro-F1 trung bình member là `+0.073990`.

## BWT, forgetting và khả năng học lớp mới

| Metric | Giá trị |
| --- | ---: |
| Ensemble mean BWT trên task cũ | **-0.097410** |
| Ensemble mean forgetting trên task cũ | **0.097410** |

| Nhóm lớp | F1 ngay sau khi học | Best F1 | F1 tại boundary 7 |
| --- | ---: | ---: | ---: |
| Task 0 | 0.971211 | 0.971211 | 0.946436 |
| Task 1 | 0.996564 | 0.996564 | 0.975855 |
| Task 2 | 0.994690 | 0.994690 | 0.967284 |
| Task 3 | 0.925056 | 0.925056 | 0.590500 |
| Task 4 | 0.700895 | 0.700895 | 0.495804 |
| Task 5 | 0.445281 | 0.445281 | 0.375946 |
| Task 6 | 0.000000 | 0.000000 | 0.000000 |
| Task 7 | 0.033333 | 0.033333 | 0.033333 |

Bảng nhóm và BWT/forgetting lấy từ các CSV đã xuất. Code báo cáo lọc mẫu theo nhãn thật thuộc nhóm task rồi tính F1 trên nhóm đó, dùng dự đoán all-seen. False positive từ mẫu có nhãn thật thuộc nhóm khác nằm ngoài tập con này. Vì vậy trung bình các F1 theo nhóm có thể khác Macro-F1 toàn cục và số cao theo nhóm cần được đọc cùng bảng full-test.

## Nhận xét

- Đây là run có Final Macro-F1 cao nhất trong năm gói DDINet vừa tổng hợp, đạt `0.521134`. So với ER–TDDI `0.500563`, chênh lệch là `+0.020571`, tương đương 2,06 điểm phần trăm.
- Task 6 sang 7 giảm Macro-F1 `0.064965`, trong khi Accuracy tăng nhẹ. Test nhóm mới task 7 chỉ có 45 mẫu cho 20 lớp. Nhóm task 6 đã có F1 bằng 0 ngay lúc vừa học và nhóm task 7 chỉ đạt `0.033333`.
- Forgetting lớn nhất nằm ở nhóm task 3, khoảng `0.334556`, tiếp theo là task 4 `0.205091`. Nhóm task 0–2 vẫn giữ F1 theo nhóm cuối trên `0.94`.
- ER cải thiện rõ so với A-GEM và EWC trên DDINet trong các run đang có. Kết quả vẫn để lại 59 / 69 / 65 lớp F1 bằng 0 ở ba member, nên Accuracy cao chưa đồng nghĩa học đều tất cả lớp.

## Phạm vi so sánh và truy vết

- Kết quả thuộc một experiment seed với ba fold member. Chênh lệch giữa run là quan sát trên seed này.
- Đối chiếu DDINet với TDDI bao gồm thay đổi kiến trúc và representation đầu vào. So sánh method còn bao gồm khác biệt memory, loss và head theo cấu hình đã ghi.
- Config M0 ghi Git commit `0b6666ffd057ed3df4a1b68b68822a642b9869d8` và `dirty=true`. Khi tái lập cần đối chiếu thêm implementation hashes trong gói với phiên bản code thực tế.

## File nguồn để đối chiếu

- `final_results/ensemble3_ordered_task_summary.csv`.
- `final_results/ensemble3_ordered_bwt_summary.csv`, `ensemble3_ordered_forgetting_summary.csv`.
- `final_results/ensemble3_ordered_member_forgetting_summary.csv`.
- `member_*/run_config.json`, `run_summary.json`, `task_7/metrics.json`.
