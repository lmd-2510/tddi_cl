# Task Arithmetic + DDINet — Head-to-Tail — Fixed Head Results

## Nguồn kết quả

- Run `task_arithmetic_ddinet_head_to_tail_fixed_head_full_seed0_20261110T151047Z`.
- Gói kết quả `task_arithmetic_ddinet_head_to_tail_fixed_head_full_seed0_20261110T151047Z_metrics_review.zip`.
- SHA256 ZIP `c26cf7e576a3eeb68a64b6dd1a7db5b0372c8455b023ba51e21a560bd732169a`.
- Ba member hoàn thành task 7, offline ensemble đủ 8 boundary.
- Đã kiểm tra CRC toàn bộ ZIP, đọc được các file kết quả. Gói này không có `PACKAGE_MANIFEST.json` để đối chiếu SHA256 từng file.

## Cấu hình chính

| Thành phần | Thiết lập |
| --- | --- |
| Method | Task Arithmetic |
| Protocol | `ORDERED / head_to_tail` |
| Backbone | `ddinet_multiclass_cil` |
| Input | Morgan 1024-bit mỗi drug, ordered concatenation thành 2048 chiều |
| Fingerprint recipe | Radius 2, chirality false theo adapter của repo |
| Hidden layers | `1024 → 512 → 256 → 128` |
| Activation / normalization / dropout | ReLU / BatchNorm / `0.3` |
| Head | Fixed 178-class head, row ownership theo task, eval trên toàn bộ lớp đã thấy |
| Task layout | `38, 20, 20, 20, 20, 20, 20, 20` |
| Ensemble | 3 frozen stratified-fold members, mean probability trên test |
| Experiment seed / fold seed | `0 / 42` |
| Preprocessing | `raw_identity`, model nhận Morgan fingerprint cache |
| Loss expert | Cross-entropy |
| Base / expert | Task 0 là base, mỗi expert task 1–7 reset độc lập từ checkpoint task 0 |
| Merge recipe | `scale=0.4`, learned backbone parameters only |
| Head ownership | Row thuộc task nào lấy từ expert của task đó, row task 0 lấy từ base |
| Boundary | Train experts rồi tính lại từng boundary với đúng expert đã thấy tới boundary đó |
| BatchNorm buffers | `weighted_pooled_expert_moments_v1`, gộp moments có trọng số theo số mẫu train từng task |
| BatchNorm với batch một mẫu | `running_stats_for_singleton_v1` |
| Calibration samples | Không lưu sample calibration theo merge audit |
| Optimizer | AdamW, learning rate `0.001`, weight decay `0.0001` |
| Batch | Microbatch `64`, effective batch `1024`, evaluation batch `512` |
| Training / selection | Tối đa 30 epoch/expert, patience `5`, chọn theo validation Macro-F1 của task expert |

Manifest DDINet đánh dấu hidden widths, ordered concatenation, radius và chirality là các lựa chọn adaptation/provisional. Báo cáo mô tả cấu hình thực tế được lưu trong run.

## Kết quả chính — offline ensemble

Các metric sau dùng toàn bộ test samples của lớp đã thấy, coverage 100%. Boundary 7 gồm 178 lớp và 173.614 mẫu.

| Phạm vi | Accuracy | Balanced Accuracy | Macro-F1 | Weighted-F1 |
| --- | ---: | ---: | ---: | ---: |
| Task 7 / kết quả cuối | 0.891760 | 0.206883 | **0.202022** | 0.871610 |
| Trung bình 8 boundary | 0.914934 | 0.440401 | **0.432453** | 0.898965 |

## Diễn biến theo task

| Boundary | Class đã thấy | Test mẫu mới | Tổng test đã thấy | Accuracy | Balanced Accuracy | Macro-F1 | Weighted-F1 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 38 | 164,910 | 164,910 | 0.967958 | 0.974745 | 0.971231 | 0.967909 |
| 1 | 58 | 4,696 | 169,606 | 0.938404 | 0.656740 | 0.644022 | 0.927072 |
| 2 | 78 | 2,220 | 171,826 | 0.920577 | 0.485425 | 0.472453 | 0.903757 |
| 3 | 98 | 1,049 | 172,875 | 0.908003 | 0.381088 | 0.371199 | 0.888821 |
| 4 | 118 | 427 | 173,302 | 0.901576 | 0.315480 | 0.307852 | 0.881645 |
| 5 | 138 | 174 | 173,476 | 0.897375 | 0.268769 | 0.262221 | 0.877272 |
| 6 | 158 | 93 | 173,569 | 0.893817 | 0.234076 | 0.228626 | 0.873637 |
| 7 | 178 | 45 | 173,614 | 0.891760 | 0.206883 | 0.202022 | 0.871610 |

Macro-F1 giảm `0.026605`, tương đương 2,66 điểm phần trăm từ task 6 sang 7. Nhóm task 7 bổ sung 20 lớp với 45 test mẫu.

## Kết quả từng member ở task 7

| Member | Accuracy | Balanced Accuracy | Macro-F1 | Weighted-F1 | Zero-F1 classes |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.867586 | 0.203692 | 0.205166 | 0.849180 | n.a. |
| 1 | 0.842046 | 0.200727 | 0.199695 | 0.826029 | n.a. |
| 2 | 0.849822 | 0.200041 | 0.200318 | 0.832995 | n.a. |
| Trung bình member | 0.853151 | 0.201487 | 0.201726 | 0.836068 | — |
| Offline ensemble | **0.891760** | **0.206883** | **0.202022** | **0.871610** | — |

Chênh lệch ensemble so với Macro-F1 trung bình member là `+0.000295`.

## Continual-learning metrics

Gói kết quả không có BWT, forgetting hoặc ma trận task-group. Các trường này được ghi `n.a.` trong bảng tổng hợp. Boundary audit ghi `future_experts_used=false`, với task 0 làm base và các expert task 1–7 được fine-tune độc lập.

## Nhận xét

- Final Accuracy `0.891760` và Weighted-F1 `0.871610` cao hơn nhiều so với Macro-F1 `0.202022` và Balanced Accuracy `0.206883`. Chất lượng dự đoán giữa các lớp còn chênh lệch lớn.
- Macro-F1 giảm liên tục qua tám boundary. Task 6 sang 7 giảm `0.026605`, nhỏ hơn mức giảm ở các boundary đầu. Vấn đề đã xuất hiện từ boundary 1.
- Offline ensemble gần như giữ nguyên Macro-F1 trung bình member. Mean probability có thể cải thiện Accuracy nhưng chưa tạo cải thiện tương ứng cho Macro-F1 trong run này.
- So với Task Arithmetic–TDDI `0.208722`, Macro-F1 thấp hơn `0.006700`, tương đương 0,67 điểm phần trăm.
- Không có task-group matrix, BWT và forgetting trong gói này. Chưa xác định được lớp hoặc nhóm task quên nhiều nhất từ các bảng đã xuất. Sự không tương thích giữa biểu diễn expert, head và thống kê BatchNorm là giả thuyết cần đối chứng thêm.

## Phạm vi so sánh và truy vết

- Kết quả thuộc một experiment seed với ba fold member. Chênh lệch giữa run là quan sát trên seed này.
- Đối chiếu DDINet với TDDI bao gồm thay đổi kiến trúc và representation đầu vào. So sánh method còn bao gồm khác biệt memory, loss và head theo cấu hình đã ghi.
- Config M0 ghi Git commit `914d9c20fdd0e00624f3f59c169f7937faa20504` và `dirty=true`. Khi tái lập cần đối chiếu thêm implementation hashes trong gói với phiên bản code thực tế.

## File nguồn để đối chiếu

- `ensemble_metrics.csv`, `ensemble_summary.json`.
- `member_*/run_config.json`, `run_summary.json`, `merged_metrics.csv`.
- `member_*/merged/task_7/merge_audit.json`.
