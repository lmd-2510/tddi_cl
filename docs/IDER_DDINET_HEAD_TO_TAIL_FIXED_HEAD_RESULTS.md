# IDER + DDINet — Head-to-Tail — Fixed Head Results

## Nguồn kết quả

- Run `ider_ddinet_head_to_tail_fixed_head_full_seed0_20261109T173342Z`.
- Gói kết quả `ider_ddinet_head_to_tail_fixed_head_full_seed0_20261109T173342Z_metrics_review.zip`.
- SHA256 ZIP `8a94cc6d63836144d0aae8bd8fa109fc5ee3bd769fcfcd65fab31dfbf444471d`.
- Ba member hoàn thành task 7, offline ensemble đủ 8 boundary.
- Đã đối chiếu kích thước và SHA256 của 131 file trong `PACKAGE_MANIFEST.json`, tất cả khớp.

## Cấu hình chính

| Thành phần | Thiết lập |
| --- | --- |
| Method | IDER |
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
| Loss | `ider`, supervised và idempotence/distillation trên current và memory |
| Adapter | Additive projection trước shared fixed classifier, riêng cho IDER |
| Memory | `equal_class_fixed_budget`, ranking `sample_normalized_class_mean` |
| Budget / sử dụng cuối | 27.778 exemplar/member, 83.334 cho ba member |
| Hệ số IDER | `a=0.5`, `b=0.4`, `c=0.5` |
| Replay riêng IDER | Replay batch `64`, uniform probability `1.0` |
| Optimizer | AdamW, learning rate `0.001`, weight decay `0.0001` |
| Batch | Microbatch `64`, effective batch `1024`, evaluation batch `512` |
| Training / selection | Tối đa 30 epoch/task, patience `5`, chọn theo all-seen validation Macro-F1 |

Manifest DDINet đánh dấu hidden widths, ordered concatenation, radius và chirality là các lựa chọn adaptation/provisional. Báo cáo mô tả cấu hình thực tế được lưu trong run.

## Kết quả chính — offline ensemble

Các metric sau dùng toàn bộ test samples của lớp đã thấy, coverage 100%. Boundary 7 gồm 178 lớp và 173.614 mẫu.

| Phạm vi | Accuracy | Balanced Accuracy | Macro-F1 | Weighted-F1 |
| --- | ---: | ---: | ---: | ---: |
| Task 7 / kết quả cuối | 0.820953 | 0.417082 | **0.301342** | 0.854357 |
| Trung bình 8 boundary | 0.841129 | 0.677550 | **0.548116** | 0.873906 |

## Diễn biến theo task

| Boundary | Class đã thấy | Test mẫu mới | Tổng test đã thấy | Accuracy | Balanced Accuracy | Macro-F1 | Weighted-F1 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 38 | 164,910 | 164,910 | 0.969019 | 0.973856 | 0.971938 | 0.968973 |
| 1 | 58 | 4,696 | 169,606 | 0.926182 | 0.858813 | 0.777421 | 0.934865 |
| 2 | 78 | 2,220 | 171,826 | 0.828774 | 0.793706 | 0.607977 | 0.870198 |
| 3 | 98 | 1,049 | 172,875 | 0.731505 | 0.753008 | 0.541728 | 0.804970 |
| 4 | 118 | 427 | 173,302 | 0.805565 | 0.629220 | 0.452498 | 0.845428 |
| 5 | 138 | 174 | 173,476 | 0.829469 | 0.531679 | 0.396655 | 0.861683 |
| 6 | 158 | 93 | 173,569 | 0.817565 | 0.463039 | 0.335365 | 0.850774 |
| 7 | 178 | 45 | 173,614 | 0.820953 | 0.417082 | 0.301342 | 0.854357 |

Macro-F1 giảm `0.034024`, tương đương 3,40 điểm phần trăm từ task 6 sang 7. Nhóm task 7 bổ sung 20 lớp với 45 test mẫu.

## Kết quả từng member ở task 7

| Member | Accuracy | Balanced Accuracy | Macro-F1 | Weighted-F1 | Zero-F1 classes |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.744796 | 0.404323 | 0.272745 | 0.805524 | 83 |
| 1 | 0.752854 | 0.388894 | 0.273161 | 0.810008 | 86 |
| 2 | 0.773215 | 0.374189 | 0.268254 | 0.819171 | 88 |
| Trung bình member | 0.756955 | 0.389135 | 0.271386 | 0.811568 | — |
| Offline ensemble | **0.820953** | **0.417082** | **0.301342** | **0.854357** | — |

Chênh lệch ensemble so với Macro-F1 trung bình member là `+0.029956`.

## BWT, forgetting và khả năng học lớp mới

| Metric | Giá trị |
| --- | ---: |
| Ensemble mean BWT trên task cũ | **-0.015086** |
| Ensemble mean forgetting trên task cũ | **0.030465** |

| Nhóm lớp | F1 ngay sau khi học | Best F1 | F1 tại boundary 7 |
| --- | ---: | ---: | ---: |
| Task 0 | 0.971938 | 0.971938 | 0.902964 |
| Task 1 | 0.746215 | 0.853862 | 0.841798 |
| Task 2 | 0.700580 | 0.700580 | 0.635707 |
| Task 3 | 0.783923 | 0.783923 | 0.716581 |
| Task 4 | 0.000000 | 0.000000 | 0.000000 |
| Task 5 | 0.000000 | 0.000000 | 0.000000 |
| Task 6 | 0.000000 | 0.000000 | 0.000000 |
| Task 7 | 0.000000 | 0.000000 | 0.000000 |

Bảng nhóm và BWT/forgetting lấy từ các CSV đã xuất. Code báo cáo lọc mẫu theo nhãn thật thuộc nhóm task rồi tính F1 trên nhóm đó, dùng dự đoán all-seen. False positive từ mẫu có nhãn thật thuộc nhóm khác nằm ngoài tập con này. Vì vậy trung bình các F1 theo nhóm có thể khác Macro-F1 toàn cục và số cao theo nhóm cần được đọc cùng bảng full-test.

## Nhận xét

- Final Macro-F1 `0.301342` thấp hơn ER–DDINet `0.219792`, dù mean BWT chỉ `-0.015086` và forgetting `0.030465`.
- Nhóm task 4–7 có F1 bằng 0 ngay sau khi học và tiếp tục bằng 0 ở boundary cuối. Forgetting nhỏ ở các nhóm này đi cùng khả năng tiếp thu lớp mới yếu.
- Nhóm task 0–3 vẫn giữ F1 theo nhóm cuối khoảng `0.903 / 0.842 / 0.636 / 0.717`. Nhóm task 1 còn cải thiện so với thời điểm vừa học, góp phần làm mean BWT gần 0.
- So với IDER–TDDI `0.376521`, run DDINet thấp hơn `0.075179`, tương đương 7,52 điểm phần trăm. Package chưa cô lập ảnh hưởng của adapter, representation và lựa chọn memory.

## Phạm vi so sánh và truy vết

- Kết quả thuộc một experiment seed với ba fold member. Chênh lệch giữa run là quan sát trên seed này.
- Đối chiếu DDINet với TDDI bao gồm thay đổi kiến trúc và representation đầu vào. So sánh method còn bao gồm khác biệt memory, loss và head theo cấu hình đã ghi.
- Config M0 ghi Git commit `8765e63b0a34d401efca0f178a6f72776a365888` và `dirty=true`. Khi tái lập cần đối chiếu thêm implementation hashes trong gói với phiên bản code thực tế.

## File nguồn để đối chiếu

- `final_results/ensemble3_ordered_task_summary.csv`.
- `final_results/ensemble3_ordered_bwt_summary.csv`, `ensemble3_ordered_forgetting_summary.csv`.
- `final_results/ensemble3_ordered_member_forgetting_summary.csv`.
- `member_*/run_config.json`, `run_summary.json`, `task_7/metrics.json`.
