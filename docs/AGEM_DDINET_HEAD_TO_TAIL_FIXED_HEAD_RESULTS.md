# A-GEM + DDINet — Head-to-Tail — Fixed Head Results

## Nguồn kết quả

- Run `agem_ddinet_head_to_tail_fixed_head_full_seed0_20261109T113318Z`.
- Gói kết quả `agem_ddinet_head_to_tail_fixed_head_full_seed0_20261109T113318Z_metrics_review.zip`.
- SHA256 ZIP `9052d262d6a173110a88b0829165f1dc74cf9934c924fa746ad9c135bbb8e96f`.
- Ba member hoàn thành task 7, offline ensemble đủ 8 boundary.
- Đã đối chiếu kích thước và SHA256 của 131 file trong `PACKAGE_MANIFEST.json`, tất cả khớp.

## Cấu hình chính

| Thành phần | Thiết lập |
| --- | --- |
| Method | A-GEM |
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
| Loss | Cross-entropy và A-GEM gradient projection |
| Memory | `agem_task_uniform_v1`, quota cố định theo task, chọn `seeded_task_uniform` |
| Budget | Tối đa 27.778 exemplar/member, 83.334 cho ba member |
| Memory thực tế cuối | 15.199 / 15.197 / 15.198 exemplar ở M0 / M1 / M2 |
| Reference batch | `256`, dùng gradient trên memory làm ràng buộc cho gradient hiện tại |
| Optimizer | AdamW, learning rate `0.001`, weight decay `0.0001` |
| Batch | Microbatch `64`, effective batch `1024`, evaluation batch `512` |
| Training / selection | Tối đa 30 epoch/task, patience `5`, chọn theo all-seen validation Macro-F1 |

Manifest DDINet đánh dấu hidden widths, ordered concatenation, radius và chirality là các lựa chọn adaptation/provisional. Báo cáo mô tả cấu hình thực tế được lưu trong run.

## Kết quả chính — offline ensemble

Các metric sau dùng toàn bộ test samples của lớp đã thấy, coverage 100%. Boundary 7 gồm 178 lớp và 173.614 mẫu.

| Phạm vi | Accuracy | Balanced Accuracy | Macro-F1 | Weighted-F1 |
| --- | ---: | ---: | ---: | ---: |
| Task 7 / kết quả cuối | 0.867741 | 0.304916 | **0.256433** | 0.880240 |
| Trung bình 8 boundary | 0.901496 | 0.534275 | **0.511796** | 0.906389 |

## Diễn biến theo task

| Boundary | Class đã thấy | Test mẫu mới | Tổng test đã thấy | Accuracy | Balanced Accuracy | Macro-F1 | Weighted-F1 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 38 | 164,910 | 164,910 | 0.967679 | 0.974033 | 0.970795 | 0.967607 |
| 1 | 58 | 4,696 | 169,606 | 0.930079 | 0.791720 | 0.780052 | 0.931081 |
| 2 | 78 | 2,220 | 171,826 | 0.912889 | 0.601533 | 0.575753 | 0.911427 |
| 3 | 98 | 1,049 | 172,875 | 0.894305 | 0.503677 | 0.474064 | 0.895754 |
| 4 | 118 | 427 | 173,302 | 0.879159 | 0.418629 | 0.393637 | 0.888073 |
| 5 | 138 | 174 | 173,476 | 0.879430 | 0.361372 | 0.343481 | 0.888688 |
| 6 | 158 | 93 | 173,569 | 0.880687 | 0.318319 | 0.300152 | 0.888238 |
| 7 | 178 | 45 | 173,614 | 0.867741 | 0.304916 | 0.256433 | 0.880240 |

Macro-F1 giảm `0.043719`, tương đương 4,37 điểm phần trăm từ task 6 sang 7. Nhóm task 7 bổ sung 20 lớp với 45 test mẫu.

## Kết quả từng member ở task 7

| Member | Accuracy | Balanced Accuracy | Macro-F1 | Weighted-F1 | Zero-F1 classes |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.747670 | 0.289682 | 0.230968 | 0.805655 | 109 |
| 1 | 0.842726 | 0.231632 | 0.216781 | 0.853039 | 118 |
| 2 | 0.766033 | 0.256921 | 0.235730 | 0.819516 | 97 |
| Trung bình member | 0.785476 | 0.259412 | 0.227826 | 0.826070 | — |
| Offline ensemble | **0.867741** | **0.304916** | **0.256433** | **0.880240** | — |

Chênh lệch ensemble so với Macro-F1 trung bình member là `+0.028607`.

## BWT, forgetting và khả năng học lớp mới

| Metric | Giá trị |
| --- | ---: |
| Ensemble mean BWT trên task cũ | **-0.039699** |
| Ensemble mean forgetting trên task cũ | **0.069837** |

| Nhóm lớp | F1 ngay sau khi học | Best F1 | F1 tại boundary 7 |
| --- | ---: | ---: | ---: |
| Task 0 | 0.970795 | 0.970795 | 0.931095 |
| Task 1 | 0.579481 | 0.609858 | 0.457875 |
| Task 2 | 0.055049 | 0.171838 | 0.171838 |
| Task 3 | 0.033062 | 0.033062 | 0.011111 |
| Task 4 | 0.234862 | 0.298664 | 0.037723 |
| Task 5 | 0.000000 | 0.000000 | 0.000000 |
| Task 6 | 0.014286 | 0.014286 | 0.000000 |
| Task 7 | 0.410952 | 0.410952 | 0.410952 |

Bảng nhóm và BWT/forgetting lấy từ các CSV đã xuất. Code báo cáo lọc mẫu theo nhãn thật thuộc nhóm task rồi tính F1 trên nhóm đó, dùng dự đoán all-seen. False positive từ mẫu có nhãn thật thuộc nhóm khác nằm ngoài tập con này. Vì vậy trung bình các F1 theo nhóm có thể khác Macro-F1 toàn cục và số cao theo nhóm cần được đọc cùng bảng full-test.

## Nhận xét

- Macro-F1 cuối thấp hơn ER–DDINet `0.264701`, dù Accuracy vẫn đạt `0.867741`. Có 109 / 118 / 97 lớp F1 bằng 0 ở ba member.
- Nhóm task 1 và task 4 có forgetting lần lượt `0.151983` và `0.260941`. Nhóm task 5 có F1 bằng 0 ngay sau khi học, nhóm task 6 từ `0.014286` về 0. Kết quả gồm cả học yếu lớp mới và suy giảm lớp cũ.
- Task 7 có F1 theo nhóm `0.410952`, nhưng từng member chênh lệch rõ. Current-group test F1 của M0 / M1 / M2 là `0.591515 / 0.014286 / 0.000000`.
- Memory thực tế chỉ khoảng 15,2 nghìn mẫu/member vì quota theo task bị giới hạn bởi số mẫu có sẵn. Budget tối đa bằng các run replay khác nhưng số exemplar thực sự dùng khác nhau.

## Phạm vi so sánh và truy vết

- Kết quả thuộc một experiment seed với ba fold member. Chênh lệch giữa run là quan sát trên seed này.
- Đối chiếu DDINet với TDDI bao gồm thay đổi kiến trúc và representation đầu vào. So sánh method còn bao gồm khác biệt memory, loss và head theo cấu hình đã ghi.
- Config M0 ghi Git commit `9859d5022089cefdca32641fa9e642f54fa76120` và `dirty=true`. Khi tái lập cần đối chiếu thêm implementation hashes trong gói với phiên bản code thực tế.

## File nguồn để đối chiếu

- `final_results/ensemble3_ordered_task_summary.csv`.
- `final_results/ensemble3_ordered_bwt_summary.csv`, `ensemble3_ordered_forgetting_summary.csv`.
- `final_results/ensemble3_ordered_member_forgetting_summary.csv`.
- `member_*/run_config.json`, `run_summary.json`, `task_7/metrics.json`.
