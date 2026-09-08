# Kết quả T-DDI Ensemble3 Replay-Distill P3

## 1. Cấu hình thí nghiệm

| Thành phần | Giá trị |
| --- | --- |
| Protocol | P3 `tail_to_head` |
| Method | `replay_distill_fixed_budget_uniform` |
| Backbone | `tddi_paper_member` |
| Số ensemble member | 3 |
| Số task | 8 |
| Experiment seed | 0 |

Ba member có cùng kiến trúc, protocol và class order nhưng dùng member seed riêng.
Offline ensemble được tính bằng trung bình xác suất của ba member.

Nguồn số liệu của tài liệu này:

- `ensemble3_p3_task_summary.csv`;
- `ensemble3_p3_paper_table.csv`;
- `ensemble3_p3_final_report.md`.

## 2. Kết quả trung bình


| Metric | Giá trị |
| --- | ---: |
| Full Accuracy | 0.901963 |
| Full Macro-F1 | 0.829565 |
| Full ECE | 0.062412 |
| Full Brier score | 0.156334 |
| Full NLL | 0.446791 |
| Threshold Accuracy | 0.947416 |
| Threshold Macro-F1 | 0.868886 |
| Average Coverage | 0.846656 |


## 3. Task summary đầy đủ

### 3.1. Metric trên toàn bộ mẫu

| Task | Full Accuracy | Full Macro-F1 | Full ECE | Full Brier | Full NLL | Số mẫu |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.904762 | 0.870852 | 0.053204 | 0.152230 | 0.548971 | 126 |
| 1 | 0.916667 | 0.883842 | 0.030787 | 0.125258 | 0.339580 | 288 |
| 2 | 0.931287 | 0.901499 | 0.059649 | 0.107135 | 0.301915 | 684 |
| 3 | 0.943327 | 0.905299 | 0.097580 | 0.109626 | 0.314623 | 1,641 |
| 4 | 0.950339 | 0.899526 | 0.114048 | 0.101049 | 0.301533 | 3,685 |
| 5 | 0.922102 | 0.870242 | 0.086017 | 0.134002 | 0.350473 | 8,049 |
| 6 | 0.830317 | 0.762089 | 0.012077 | 0.243473 | 0.578702 | 23,756 |
| 7 | 0.816904 | 0.543176 | 0.045930 | 0.277903 | 0.838528 | 173,614 |

### 3.2. Metric sau khi áp dụng confidence threshold

| Task | Score | Threshold | Threshold Accuracy | Threshold Macro-F1 | Coverage | Được giữ | Tổng mẫu |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | `entropy_confidence` | 0.8 | 0.946903 | 0.892907 | 0.896825 | 113 | 126 |
| 1 | `entropy_confidence` | 0.7 | 0.957854 | 0.925503 | 0.906250 | 261 | 288 |
| 2 | `entropy_confidence` | 0.7 | 0.975288 | 0.947838 | 0.887427 | 607 | 684 |
| 3 | `entropy_confidence` | 0.8 | 0.983845 | 0.939204 | 0.754418 | 1,238 | 1,641 |
| 4 | `entropy_confidence` | 0.7 | 0.984157 | 0.946306 | 0.856445 | 3,156 | 3,685 |
| 5 | `entropy_confidence` | 0.8 | 0.975453 | 0.912392 | 0.733880 | 5,907 | 8,049 |
| 6 | `entropy_confidence` | 0.8 | 0.921905 | 0.833183 | 0.768101 | 18,247 | 23,756 |
| 7 | `entropy_confidence` | 0.7 | 0.833919 | 0.553752 | 0.969899 | 168,388 | 173,614 |

## 4. Bảng rút gọn dùng cho báo cáo/paper

Bảng này giữ đúng sáu cột từ `ensemble3_p3_paper_table.csv`.

| Task | Full Macro-F1 | Threshold Macro-F1 | Threshold Accuracy | Coverage | Threshold |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.870852 | 0.892907 | 0.946903 | 0.896825 | 0.8 |
| 1 | 0.883842 | 0.925503 | 0.957854 | 0.906250 | 0.7 |
| 2 | 0.901499 | 0.947838 | 0.975288 | 0.887427 | 0.7 |
| 3 | 0.905299 | 0.939204 | 0.983845 | 0.754418 | 0.8 |
| 4 | 0.899526 | 0.946306 | 0.984157 | 0.856445 | 0.7 |
| 5 | 0.870242 | 0.912392 | 0.975453 | 0.733880 | 0.8 |
| 6 | 0.762089 | 0.833183 | 0.921905 | 0.768101 | 0.8 |
| 7 | 0.543176 | 0.553752 | 0.833919 | 0.969899 | 0.7 |

## 5. Giải thích các cột trong task summary

### `task_id`

Task vừa hoàn thành trong trajectory CIL, đánh số từ 0 đến 7. Ở task `t`, model đã
học các lớp xuất hiện từ task 0 đến task `t`. Vì vậy số mẫu đánh giá tăng dần theo số
lớp đã thấy; task 7 chứa toàn bộ 178 lớp.

### `full_accuracy`

Tỷ lệ dự đoán đúng trên toàn bộ mẫu đánh giá của các lớp đã thấy, trước khi lọc theo
confidence. Giá trị càng cao càng tốt. Accuracy có thể bị chi phối bởi lớp nhiều mẫu,
nên phải đọc cùng Macro-F1 trong dữ liệu mất cân bằng.

### `full_macro_f1`

Tính F1 riêng cho từng lớp rồi lấy trung bình, mỗi lớp có trọng số bằng nhau. Giá trị
càng cao càng tốt. Đây là metric quan trọng để biết model có hoạt động đồng đều trên
cả lớp hiếm và lớp phổ biến hay không.

### `full_ece`

Expected Calibration Error trên toàn bộ mẫu. Trong pipeline này, ECE dùng maximum
ensemble probability làm confidence và so sánh nó với độ chính xác quan sát được
trong các confidence bin. Giá trị càng gần 0 càng tốt. ECE thấp chỉ nói confidence
được hiệu chỉnh tốt; nó không đảm bảo Macro-F1 cao.

### `full_brier_score`

Sai số bình phương giữa vector xác suất dự đoán và nhãn one-hot thật. Metric đánh giá
toàn bộ phân phối xác suất, không chỉ class có xác suất lớn nhất. Giá trị càng thấp
càng tốt.

### `full_nll`

Negative Log-Likelihood của nhãn thật. Dự đoán sai nhưng quá tự tin bị phạt mạnh. Giá
trị càng thấp càng tốt.

### `full_samples`

Số mẫu được dùng để tính nhóm metric `full_*` tại task tương ứng.

### `threshold_score`

Tên confidence score dùng để lọc mẫu. Trong kết quả này score là
`entropy_confidence`, được định nghĩa là:

```text
entropy_confidence = 1 - predictive_entropy / log(C_t)
```

Trong đó `C_t` là số lớp model đã thấy tại task `t`. Giá trị cao nghĩa là phân phối
ensemble tập trung hơn và model ít bất định hơn. Đây không phải là max softmax
probability.

### `threshold_value`

Ngưỡng áp dụng lên `entropy_confidence`. Chỉ những mẫu có score đạt ngưỡng mới được
giữ để tính metric threshold. Theo protocol của study, ngưỡng phải được chọn trên
validation rồi freeze trước khi áp dụng lên test.

### `threshold_accuracy` và `threshold_macro_f1`

Accuracy và Macro-F1 chỉ trên tập mẫu vượt threshold. Hai giá trị này thường cao hơn
metric full-set vì model được phép từ chối các mẫu có uncertainty cao. Chúng không nên
được báo cáo riêng mà không kèm coverage.

### `coverage`

Tỷ lệ mẫu được giữ lại sau threshold:

```text
coverage = selected_count / total_count
```

Coverage cao nghĩa là model đưa ra dự đoán cho nhiều mẫu hơn. Coverage thấp có thể làm
metric threshold đẹp hơn do loại bỏ nhiều ca khó. Vì vậy cần đánh giá đồng thời chất
lượng và mức bao phủ.

### `selected_count` và `total_count`

`selected_count` là số mẫu vượt threshold. `total_count` là tổng số mẫu trước khi lọc.
Hai cột này là số nguyên dùng để kiểm tra trực tiếp phép tính coverage.

## 6. Cách đọc kết quả

- Task 0–5 duy trì Full Macro-F1 từ 0.870242 đến 0.905299. Hiệu năng giảm rõ ở task 6
  và đặc biệt ở task 7.
- Ở task 7, Accuracy là 0.816904 nhưng Macro-F1 chỉ còn 0.543176. Khoảng cách này cho
  thấy hiệu năng giữa các lớp không đồng đều; các lớp nhiều mẫu có thể giữ accuracy
  cao trong khi một số lớp có F1 thấp.
- Threshold nâng Macro-F1 trung bình từ 0.829565 lên 0.868886, tăng khoảng 0.039320,
  với Average Coverage 0.846656.
- Task 7 có coverage 0.969899 nhưng Threshold Macro-F1 chỉ tăng từ 0.543176 lên
  0.553752. Confidence filtering vì vậy không giải quyết được phần lớn suy giảm
  per-class ở task cuối.
- Full ECE thấp nhất ở task 6 là 0.012077 nhưng Full Macro-F1 task này chỉ 0.762089.
  Đây là ví dụ cho thấy calibration tốt và classification tốt là hai khía cạnh khác
  nhau.

Do task 7 có nhiều mẫu hơn hẳn các task trước, coverage gộp theo toàn bộ 211,843 dòng
là 0.934263, trong khi Average Coverage báo cáo là 0.846656 do lấy trung bình đều trên
tám task. Tài liệu sử dụng Average Coverage để khớp báo cáo nguồn.

## 7. Kết luận

Ensemble3 Replay-Distill đạt kết quả mạnh ở sáu task đầu và confidence threshold cải
thiện metric trên phần mẫu được chấp nhận. Điểm cần phân tích tiếp là suy giảm
Macro-F1 ở task 6–7, đặc biệt chênh lệch lớn giữa Accuracy và Macro-F1 tại task cuối.
Task summary chỉ chỉ ra hiện tượng, chưa đủ để kết luận nguyên nhân. Cần kết hợp
class-forgetting, old/current-class breakdown và rarity breakdown từ UE audit để phân
biệt forgetting, class imbalance và lỗi calibration.

Ba file nguồn không chứa artifact hash, member seed hoặc đường dẫn frozen-threshold.
Do đó tài liệu này xác nhận tính nhất quán số học giữa ba file nhưng chưa tự xác minh
provenance của prediction/threshold artifact.

## 8. Tạo lại báo cáo từ artifact đầy đủ

Sau khi pull phiên bản repo mới lên server, dùng `src/eval/build_final_report.py` để
tạo báo cáo mở rộng có Balanced Accuracy, Weighted F1, forgetting và ensemble
diversity. Bước này không train hoặc chạy inference:

```bash
python src/eval/build_final_report.py \
  --full-root outputs/full/tddi_ensemble3_replay_distill_p3_seed0_8tasks_v1 \
  --task-file outputs/tasks/tail_to_head_tasks.json \
  --overwrite
```
