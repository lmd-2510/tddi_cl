# UPCL + DDINet — Head-to-Tail — Fixed Head Results

## Nguồn kết quả

- Run `upcl_ddinet_head_to_tail_fixed_head_full_seed0_20261110T081853Z`.
- Gói kết quả `upcl_ddinet_head_to_tail_fixed_head_full_seed0_20261110T081853Z_metrics_review.zip`.
- SHA256 ZIP `76a43707839fe307c0b0b270a1ef7a38455cae62558f3ea07dd5dec06b2f6f59`.
- Ba member hoàn thành task 7, offline ensemble đủ 8 boundary.
- Đã đối chiếu kích thước và SHA256 của 131 file trong `PACKAGE_MANIFEST.json`, tất cả khớp.

## Cấu hình chính

| Thành phần | Thiết lập |
| --- | --- |
| Method | UPCL |
| Protocol | `ORDERED / head_to_tail` |
| Backbone | `ddinet_multiclass_cil` |
| Input | Morgan 1024-bit mỗi drug, ordered concatenation thành 2048 chiều |
| Fingerprint recipe | Radius 2, chirality false theo adapter của repo |
| Hidden layers | `1024 → 512 → 256 → 256` |
| Activation / normalization / dropout | ReLU / BatchNorm / `0.3` |
| Head | Fixed 178-class head, mask lớp chưa thấy |
| Task layout | `38, 20, 20, 20, 20, 20, 20, 20` |
| Ensemble | 3 frozen stratified-fold members, mean probability trên test |
| Experiment seed / fold seed | `0 / 42` |
| Preprocessing | `raw_identity`, model nhận Morgan fingerprint cache |
| Loss | Uniform-prototype contrastive loss và feature knowledge distillation |
| Classifier UPCL | Frozen unit-norm prototypes đặt tại các row của shared fixed head |
| Temperature | `0.07`, giá trị được ghi trong config của run |
| Memory | `equal_class_fixed_budget`, ranking `sample_normalized_class_mean` |
| Budget / sử dụng cuối | 27.778 exemplar/member, 83.334 cho ba member |
| Replay | Full-current + full-memory sampler trong mỗi epoch |
| BatchNorm với batch một mẫu | `running_stats_for_singleton_v1` |
| Optimizer | AdamW, learning rate `0.001`, weight decay `0.0001` |
| Batch | Microbatch `64`, effective batch `1024`, evaluation batch `512` |
| Training / selection | Tối đa 30 epoch/task, patience `5`, chọn theo all-seen validation Macro-F1 |

Manifest DDINet đánh dấu hidden widths, ordered concatenation, radius và chirality là các lựa chọn adaptation/provisional. Báo cáo mô tả cấu hình thực tế được lưu trong run.

Hidden cuối 256 chiều là thay đổi riêng cho UPCL đã được chấp thuận. Trường `replay_fraction=0.0` thuộc CLI chung, trong khi sampler UPCL vẫn ghép current và memory vào dữ liệu mỗi epoch.

## Kết quả chính — offline ensemble

Các metric sau dùng toàn bộ test samples của lớp đã thấy, coverage 100%. Boundary 7 gồm 178 lớp và 173.614 mẫu.

| Phạm vi | Accuracy | Balanced Accuracy | Macro-F1 | Weighted-F1 |
| --- | ---: | ---: | ---: | ---: |
| Task 7 / kết quả cuối | 0.476638 | 0.560851 | **0.284112** | 0.559114 |
| Trung bình 8 boundary | 0.628352 | 0.761689 | **0.518095** | 0.687316 |

## Diễn biến theo task

| Boundary | Class đã thấy | Test mẫu mới | Tổng test đã thấy | Accuracy | Balanced Accuracy | Macro-F1 | Weighted-F1 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 38 | 164,910 | 164,910 | 0.841101 | 0.950918 | 0.919837 | 0.809935 |
| 1 | 58 | 4,696 | 169,606 | 0.838148 | 0.942547 | 0.757217 | 0.874115 |
| 2 | 78 | 2,220 | 171,826 | 0.746179 | 0.867567 | 0.623885 | 0.800968 |
| 3 | 98 | 1,049 | 172,875 | 0.570557 | 0.854295 | 0.494589 | 0.652220 |
| 4 | 118 | 427 | 173,302 | 0.537420 | 0.713002 | 0.406713 | 0.618026 |
| 5 | 138 | 174 | 173,476 | 0.521386 | 0.612460 | 0.342887 | 0.604347 |
| 6 | 158 | 93 | 173,569 | 0.495388 | 0.591872 | 0.315520 | 0.579800 |
| 7 | 178 | 45 | 173,614 | 0.476638 | 0.560851 | 0.284112 | 0.559114 |

Macro-F1 giảm `0.031408`, tương đương 3,14 điểm phần trăm từ task 6 sang 7. Nhóm task 7 bổ sung 20 lớp với 45 test mẫu.

## Kết quả từng member ở task 7

| Member | Accuracy | Balanced Accuracy | Macro-F1 | Weighted-F1 | Zero-F1 classes |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.457740 | 0.528502 | 0.273695 | 0.543355 | 66 |
| 1 | 0.447833 | 0.496836 | 0.269871 | 0.534672 | 71 |
| 2 | 0.436169 | 0.589168 | 0.274773 | 0.512932 | 54 |
| Trung bình member | 0.447247 | 0.538168 | 0.272780 | 0.530319 | — |
| Offline ensemble | **0.476638** | **0.560851** | **0.284112** | **0.559114** | — |

Chênh lệch ensemble so với Macro-F1 trung bình member là `+0.011333`.

## BWT, forgetting và khả năng học lớp mới

| Metric | Giá trị |
| --- | ---: |
| Ensemble mean BWT trên task cũ | **0.157033** |
| Ensemble mean forgetting trên task cũ | **0.028480** |

| Nhóm lớp | F1 ngay sau khi học | Best F1 | F1 tại boundary 7 |
| --- | ---: | ---: | ---: |
| Task 0 | 0.919837 | 0.919837 | 0.739009 |
| Task 1 | 0.976608 | 0.989234 | 0.970703 |
| Task 2 | 0.752897 | 0.993021 | 0.993021 |
| Task 3 | 0.749113 | 0.989341 | 0.989341 |
| Task 4 | 0.000000 | 0.711915 | 0.711915 |
| Task 5 | 0.000000 | 0.093697 | 0.093697 |
| Task 6 | 0.000000 | 0.000000 | 0.000000 |
| Task 7 | 0.000000 | 0.000000 | 0.000000 |

Bảng nhóm và BWT/forgetting lấy từ các CSV đã xuất. Code báo cáo lọc mẫu theo nhãn thật thuộc nhóm task rồi tính F1 trên nhóm đó, dùng dự đoán all-seen. False positive từ mẫu có nhãn thật thuộc nhóm khác nằm ngoài tập con này. Vì vậy trung bình các F1 theo nhóm có thể khác Macro-F1 toàn cục và số cao theo nhóm cần được đọc cùng bảng full-test.

## Nhận xét

- Balanced Accuracy cuối `0.560851` cao hơn ER–DDINet `0.554412`, nhưng Macro-F1 chỉ `0.284112` và Accuracy `0.476638`. Recall trung bình theo lớp khá cao trong khi precision và chất lượng tổng thể còn hạn chế.
- Mean BWT dương `0.157033` chủ yếu đi cùng cải thiện của các nhóm task 2–5 sau thời điểm vừa học. Nhóm task 0 vẫn giảm F1 theo nhóm từ `0.919837` xuống `0.739009`.
- Nhóm task 6 và 7 có F1 bằng 0 ngay sau khi học và tại boundary cuối. Nhóm task 5 cải thiện từ 0 lên `0.093697`. BWT dương cần được đọc cùng các nhóm chưa học được.
- Run này dùng hidden cuối 256 chiều để đáp ứng 178 prototype trực giao. Đối chiếu với các DDINet hidden 128 chiều bao gồm cả thay đổi kiến trúc này.
- So với UPCL–TDDI `0.219061`, Macro-F1 tăng `0.065051`, tương đương 6,51 điểm phần trăm. Kết quả cuối vẫn thấp hơn ER–DDINet.

## Phạm vi so sánh và truy vết

- Kết quả thuộc một experiment seed với ba fold member. Chênh lệch giữa run là quan sát trên seed này.
- Đối chiếu DDINet với TDDI bao gồm thay đổi kiến trúc và representation đầu vào. So sánh method còn bao gồm khác biệt memory, loss và head theo cấu hình đã ghi.
- Config M0 ghi Git commit `7671fc84539157bd1a8925536b6e80ca6dbb2cec` và `dirty=true`. Khi tái lập cần đối chiếu thêm implementation hashes trong gói với phiên bản code thực tế.

## File nguồn để đối chiếu

- `final_results/ensemble3_ordered_task_summary.csv`.
- `final_results/ensemble3_ordered_bwt_summary.csv`, `ensemble3_ordered_forgetting_summary.csv`.
- `final_results/ensemble3_ordered_member_forgetting_summary.csv`.
- `member_*/run_config.json`, `run_summary.json`, `task_7/metrics.json`.
