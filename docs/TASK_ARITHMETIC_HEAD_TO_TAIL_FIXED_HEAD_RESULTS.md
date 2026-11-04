# Task Arithmetic — Head-to-Tail — Fixed Head Results

## Nguồn kết quả

- Run: `task_arithmetic_head_to_tail_fixed_head_full_seed0_20261104T120427Z`
- Gói kết quả: `task_arithmetic_head_to_tail_fixed_head_full_seed0_20261104T120427Z_metrics_review.zip`
- SHA256 ZIP: `6CF97EA6D1F1E40A91E60422780506B766B6B18DC78F9F5D567BDE15F5D04A8B`
- Trạng thái: hoàn thành 8 boundary, 3 member và offline ensemble.

## Cấu hình chính

| Thành phần | Thiết lập |
| --- | --- |
| Method | Task Arithmetic |
| Protocol | `ORDERED / head_to_tail` |
| Backbone | `tddi_paper_member` |
| Head | Fixed 178-class head, ownership theo task |
| Task layout | `38, 20, 20, 20, 20, 20, 20, 20` |
| Ensemble | 3 frozen stratified-fold members |
| Experiment seed / fold seed | `0 / 42` |
| Preprocessing | `task0_standard_frozen` |
| Merge scope | Learned backbone parameters only; head rows được stitch theo ownership |
| Training recipe | Task 0 là base; mỗi expert task 1–7 reset từ task-0 base |
| Task Arithmetic scale | `0.4` |
| Optimizer | AdamW, learning rate `0.001`, weight decay `0.0001` |
| Batch | Microbatch `64`, effective batch `1024` |
| Training | Tối đa 30 epoch/task, patience `5` |

## Kết quả chính — offline ensemble

Các metric là full-coverage test metric tại boundary cuối (task 7).

| Phạm vi | Accuracy | Balanced Accuracy | Macro-F1 | Weighted-F1 |
| --- | ---: | ---: | ---: | ---: |
| Task 7 / kết quả cuối | 0.819018 | 0.184912 | **0.208722** | 0.802104 |
| Trung bình Macro-F1 của 8 boundary | — | — | **0.446440** | — |

## Diễn biến theo boundary

| Boundary | Số class đã thấy | Test samples | Accuracy | Balanced Accuracy | Macro-F1 | Weighted-F1 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 38 | 164,910 | 0.940143 | 0.947025 | 0.947408 | 0.939943 |
| 1 | 58 | 169,606 | 0.912208 | 0.639875 | 0.647144 | 0.901804 |
| 2 | 78 | 171,826 | 0.892781 | 0.485648 | 0.507759 | 0.878354 |
| 3 | 98 | 172,875 | 0.872359 | 0.378192 | 0.405979 | 0.856053 |
| 4 | 118 | 173,302 | 0.851767 | 0.305206 | 0.335827 | 0.835204 |
| 5 | 138 | 173,476 | 0.840012 | 0.253654 | 0.281331 | 0.823185 |
| 6 | 158 | 173,569 | 0.826772 | 0.211854 | 0.237347 | 0.809925 |
| 7 | 178 | 173,614 | 0.819018 | 0.184912 | 0.208722 | 0.802104 |

## Kết quả từng member ở task 7

| Member | Macro-F1 | Mean boundary Macro-F1 |
| ---: | ---: | ---: |
| 0 | 0.209265 | 0.442839 |
| 1 | 0.218094 | 0.449317 |
| 2 | 0.210340 | 0.440637 |
| Trung bình member | 0.212566 | 0.444264 |
| Offline ensemble | **0.208722** | **0.446440** |

## Nhận xét

1. Task Arithmetic giữ Accuracy và Weighted-F1 khá cao, nhưng Macro-F1 cuối chỉ `0.208722` và Balanced Accuracy chỉ `0.184912`. Điều này cho thấy model vẫn dự đoán tốt các class có nhiều mẫu nhưng rất yếu trên các class ít mẫu.
2. Macro-F1 giảm liên tục từ `0.947408` ở boundary 0 xuống `0.208722` ở boundary 7. Mức giảm từ boundary 6 sang 7 là `0.028625`, không phải cú sụt lớn nhất; chất lượng đã suy giảm dần từ khi số class tăng.
3. Accuracy cao nhưng Macro-F1 thấp là dấu hiệu rõ của long-tail bias: các class head chi phối số lượng test mẫu, còn các class tail có thể có F1 rất thấp hoặc bằng 0.
4. Đây không phải forgetting theo nghĩa CL tuần tự thuần túy, vì các expert được train độc lập từ task-0 base. Do đó package hiện chưa báo cáo BWT/forgetting chuẩn cho run này.
5. Kết quả cho thấy phép cộng task vector với scale `0.4` chưa đủ để tạo backbone chung tương thích cho toàn bộ 178 class. Việc stitch head giúp tránh ghi đè row giữa các task nhưng không giải quyết được lệch biểu diễn giữa các expert.

## Kết luận

Trong cấu hình hiện tại, Task Arithmetic chưa phù hợp nếu mục tiêu chính là Macro-F1 trên toàn bộ 178 class. Nó tạo Accuracy/Weighted-F1 cao do ưu tiên class nhiều mẫu, nhưng không xử lý được intra-task imbalance và sự không tương thích giữa expert khi số class tăng.

