# EWC — Head-to-Tail — Fixed Head Results

## Nguồn kết quả

- Run: `ewc_head_to_tail_fixed_head_full_seed0_20261102T071912Z`
- Gói kết quả: `ewc_head_to_tail_fixed_head_full_seed0_20261102T071912Z_metrics_review.zip`
- SHA256 của ZIP: `d807f6ea3120a2c1ef1452ba6bb8fa3b31b8bf14988a9b58ea2c8892836b00f9`
- Trạng thái: hoàn thành 8 task, 3 member và offline ensemble.

## Cấu hình chính

| Thành phần | Thiết lập |
| --- | --- |
| Method | Original EWC, không replay buffer |
| Protocol | `ORDERED / head_to_tail` |
| Backbone | `tddi_paper_member` |
| Head | Fixed head 178 class, mở mask theo task |
| Task layout | `38, 20, 20, 20, 20, 20, 20, 20` |
| Ensemble | 3 frozen stratified-fold members |
| Experiment seed | `0` |
| Fold seed | `42` |
| Preprocessing | `task0_standard_frozen` |
| Loss | Cross-entropy |
| EWC coefficient | `lambda = 1000` |
| Optimizer | AdamW, learning rate `0.001`, weight decay `0.0001` |
| Batch | Microbatch `64`, effective batch `1024` |
| Training | Tối đa 30 epoch/task, patience `5` |
| Memory budget | `0` |

## Kết quả chính

Các số dưới đây là kết quả test của offline ensemble trên toàn bộ mẫu thuộc các class đã thấy.

| Phạm vi | Accuracy | Balanced Accuracy | Macro-F1 | Weighted-F1 |
| --- | ---: | ---: | ---: | ---: |
| Task 7 / kết quả cuối | 0.569257 | 0.548442 | **0.377209** | 0.668461 |
| Trung bình 8 task | 0.737343 | 0.692398 | **0.575171** | 0.784793 |

## Diễn biến theo task

| Task | Nhóm | Class mới | Test mẫu mới | Tổng test mẫu đã thấy | Accuracy | Balanced Accuracy | Macro-F1 | Weighted-F1 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | Head | 38 | 164,910 | 164,910 | 0.941107 | 0.948290 | 0.948217 | 0.940924 |
| 1 | Mid | 20 | 4,696 | 169,606 | 0.863519 | 0.829761 | 0.764021 | 0.869310 |
| 2 | Mid | 20 | 2,220 | 171,826 | 0.811664 | 0.753198 | 0.641529 | 0.827955 |
| 3 | Mid | 20 | 1,049 | 172,875 | 0.762551 | 0.677644 | 0.535921 | 0.791924 |
| 4 | Mid | 20 | 427 | 173,302 | 0.701284 | 0.626554 | 0.483223 | 0.758837 |
| 5 | Mid | 20 | 174 | 173,476 | 0.654892 | 0.574741 | 0.444978 | 0.730571 |
| 6 | Mid | 20 | 93 | 173,569 | 0.594473 | 0.580551 | 0.406274 | 0.690366 |
| 7 | Tail | 20 | 45 | 173,614 | 0.569257 | 0.548442 | 0.377209 | 0.668461 |

Macro-F1 giảm từ `0.948217` ở task 0 xuống `0.377209` ở task 7. Task 6 → 7 giảm `0.029065`, nhỏ hơn các đợt giảm mạnh ở đầu trajectory nhưng kết quả cuối vẫn thấp do quên tích lũy.

## Continual-learning metrics

| Metric | Giá trị |
| --- | ---: |
| Mean BWT trên task cũ | **-0.210327** |
| Mean forgetting trên task cũ | **0.217414** |
| Member task-forgetting, mean ± std | 0.219289 ± 0.016169 |
| Member class-forgetting cuối, mean ± std | 0.125313 ± 0.007886 |

Các task bị quên mạnh nhất là task 3 (`0.293425`), task 0 (`0.291841`) và task 6 (`0.287943`). BWT âm lớn xác nhận EWC chưa giữ được biểu diễn cũ đủ tốt trên protocol mất cân bằng này.

## Kết quả từng member ở task 7

| Member | Accuracy | Balanced Accuracy | Macro-F1 | Weighted-F1 | Zero-F1 classes |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.448380 | 0.431993 | 0.265717 | 0.553661 | 22 |
| 1 | 0.427702 | 0.471353 | 0.278232 | 0.536223 | 11 |
| 2 | 0.427800 | 0.468288 | 0.289010 | 0.547295 | 10 |
| Trung bình member | 0.434627 | 0.457211 | 0.277653 | 0.545726 | — |
| Offline ensemble | **0.569257** | **0.548442** | **0.377209** | **0.668461** | — |

Offline ensemble tăng Macro-F1 khoảng `+0.099556` so với trung bình ba member, vì vậy ensemble đóng góp đáng kể. Tuy nhiên, mức tăng này không giải quyết được forgetting của từng model.


