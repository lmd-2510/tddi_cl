# Kết quả T-DDI Ensemble3 Replay-Distill P3

## 1. Phạm vi và cách đọc tài liệu

Tài liệu này tổng hợp kết quả của study:

```text
replay_distill_fixed_budget_uniform
× tddi_paper_member
× ensemble 3 member
× protocol P3 tail_to_head
× experiment seed 0
× 8 task / 178 class
```

Cần phân biệt ba khái niệm:

1. **Kết quả model cuối**: dòng `task_id=7`, model đã học xong cả tám task và được
   đánh giá trên toàn bộ 178 class (`seen_all`). Đây là kết quả chính của run.
2. **Kết quả theo giai đoạn**: dòng `task_id=t` dùng model vừa học xong task `t` và
   đánh giá trên tất cả class đã thấy từ task 0 đến task `t`.
3. **Mean across tasks/stages**: trung bình đều tám dòng theo giai đoạn. Đây là metric
   phụ mô tả toàn bộ quá trình học, không phải hiệu năng của model cuối.

Ba member dùng ba `member_seed` khác nhau để tạo ensemble, nhưng toàn bộ study hiện
chỉ có **một experiment seed là 0**. Vì vậy kết quả này chưa phải `mean ± std` qua
nhiều experiment seed.

## 2. Cấu hình thí nghiệm

| Thành phần | Giá trị |
| --- | --- |
| Protocol | P3 `tail_to_head` |
| Method | `replay_distill_fixed_budget_uniform` |
| Backbone | `tddi_paper_member` |
| Experiment seed | `0` |
| Member IDs | `[0, 1, 2]` |
| Member seeds | `[409845317, 215626784, 3041879697]` |
| Số task | 8 |
| Tổng số class | 178 |

Ba member có cùng dữ liệu, task order và class map nhưng có initialization, dropout
và sampler order riêng. Offline ensemble được tính bằng **mean probabilities**, không
trung bình raw logits.

## 3. Kết quả chính: model cuối sau task 7

Đây là kết quả cần dùng khi trả lời câu hỏi: "Sau khi học xong toàn bộ tám task, model
còn hoạt động tốt thế nào trên tất cả 178 class?"

### 3.1. Full-set test metrics

| Metric | Kết quả |
| --- | ---: |
| Accuracy | 0.816904 |
| Balanced Accuracy | 0.499500 |
| Macro-F1 | 0.543176 |
| Weighted F1 | 0.779396 |
| ECE | 0.045930 |
| Brier score | 0.277903 |
| NLL | 0.838528 |
| Test samples | 173,614 |

Diễn giải ngắn:

- Accuracy `0.816904` cho thấy ensemble dự đoán đúng khoảng 81.69% tổng số mẫu.
- Balanced Accuracy chỉ `0.499500`, thấp hơn Accuracy rất nhiều, cho thấy recall giữa
  các class không đồng đều.
- Macro-F1 `0.543176` xác nhận các class được đối xử không đồng đều; nhiều class cũ
  hoặc hiếm có chất lượng thấp dù Accuracy tổng thể còn cao.
- Weighted F1 `0.779396` gần Accuracy hơn Macro-F1 vì các class nhiều mẫu có trọng số
  lớn hơn.

### 3.2. Selective prediction sau confidence threshold

Threshold được chọn trên validation rồi freeze trước khi áp dụng lên test.

| Thuộc tính | Kết quả |
| --- | ---: |
| Confidence score | `entropy_confidence` |
| Threshold | 0.700000 |
| Accuracy trên mẫu được giữ | 0.833919 |
| Balanced Accuracy trên mẫu được giữ | 0.509733 |
| Macro-F1 trên mẫu được giữ | 0.553752 |
| Weighted F1 trên mẫu được giữ | 0.799582 |
| Coverage | 0.969899 |
| Số mẫu được giữ | 168,388 / 173,614 |

Model giữ lại khoảng 96.99% mẫu và từ chối khoảng 3.01%. Macro-F1 chỉ tăng từ
`0.543176` lên `0.553752`; do đó threshold ở task cuối chỉ cải thiện nhẹ và không giải
quyết được phần lớn lỗi do mất cân bằng hoặc forgetting.

`threshold=0.7` không có nghĩa là `max probability >= 0.7`. Score sử dụng ở đây là:

```text
entropy_confidence = 1 - predictive_entropy / log(C_t)
```

## 4. Kết quả theo từng giai đoạn học

Mỗi dòng dưới đây là một model state khác nhau:

```text
task_id=t
= model vừa train xong task t
= đánh giá seen_all của task 0..t
```

Vì vậy chỉ dòng task 7 là model cuối sau toàn bộ trajectory.

### 4.1. Full-set classification và calibration

| Stage | Seen classes | Accuracy | Balanced Acc. | Macro-F1 | Weighted F1 | ECE | Brier | NLL | Samples |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | task 0 | 0.904762 | 0.884649 | 0.870852 | 0.893163 | 0.053204 | 0.152230 | 0.548971 | 126 |
| 1 | task 0–1 | 0.916667 | 0.887062 | 0.883842 | 0.913332 | 0.030787 | 0.125258 | 0.339580 | 288 |
| 2 | task 0–2 | 0.931287 | 0.901030 | 0.901499 | 0.927987 | 0.059649 | 0.107135 | 0.301915 | 684 |
| 3 | task 0–3 | 0.943327 | 0.903328 | 0.905299 | 0.941117 | 0.097580 | 0.109626 | 0.314623 | 1,641 |
| 4 | task 0–4 | 0.950339 | 0.896760 | 0.899526 | 0.948891 | 0.114048 | 0.101049 | 0.301533 | 3,685 |
| 5 | task 0–5 | 0.922102 | 0.872100 | 0.870242 | 0.919655 | 0.086017 | 0.134002 | 0.350473 | 8,049 |
| 6 | task 0–6 | 0.830317 | 0.741858 | 0.762089 | 0.811962 | 0.012077 | 0.243473 | 0.578702 | 23,756 |
| **7** | **task 0–7 (178 class)** | **0.816904** | **0.499500** | **0.543176** | **0.779396** | **0.045930** | **0.277903** | **0.838528** | **173,614** |

Hiệu năng mạnh ở stage 0–5, giảm ở stage 6 và giảm rõ hơn sau khi học task 7. Đặc biệt,
khoảng cách giữa Accuracy và Macro-F1 tại stage cuối cho thấy hiệu năng trên các class
phổ biến che khuất suy giảm trên nhiều class khác.

### 4.2. Selective prediction theo từng giai đoạn

| Stage | Threshold | Accuracy | Balanced Acc. | Macro-F1 | Weighted F1 | Coverage | Selected / total |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.8 | 0.946903 | 0.907895 | 0.892907 | 0.933821 | 0.896825 | 113 / 126 |
| 1 | 0.7 | 0.957854 | 0.932628 | 0.925503 | 0.955375 | 0.906250 | 261 / 288 |
| 2 | 0.7 | 0.975288 | 0.951196 | 0.947838 | 0.974357 | 0.887427 | 607 / 684 |
| 3 | 0.8 | 0.983845 | 0.945844 | 0.939204 | 0.984196 | 0.754418 | 1,238 / 1,641 |
| 4 | 0.7 | 0.984157 | 0.951525 | 0.946306 | 0.984126 | 0.856445 | 3,156 / 3,685 |
| 5 | 0.8 | 0.975453 | 0.913882 | 0.912392 | 0.974825 | 0.733880 | 5,907 / 8,049 |
| 6 | 0.8 | 0.921905 | 0.821405 | 0.833183 | 0.908443 | 0.768101 | 18,247 / 23,756 |
| **7** | **0.7** | **0.833919** | **0.509733** | **0.553752** | **0.799582** | **0.969899** | **168,388 / 173,614** |

Metric threshold phải luôn được báo cáo cùng coverage. Nếu chỉ đưa Accuracy hoặc
Macro-F1 sau threshold mà bỏ coverage, kết quả có thể gây hiểu nhầm vì model được phép
loại các mẫu khó.

## 5. Trung bình qua tám giai đoạn — kết quả phụ

Mỗi stage có trọng số bằng nhau trong bảng này. Đây không phải metric của model cuối,
không phải mean per-task của model cuối và cũng không phải mean qua nhiều seed.

| Metric | Mean across 8 training stages |
| --- | ---: |
| Full Accuracy | 0.901963 |
| Full Balanced Accuracy | 0.823286 |
| Full Macro-F1 | 0.829565 |
| Full Weighted F1 | 0.891938 |
| Threshold Accuracy | 0.947416 |
| Threshold Balanced Accuracy | 0.866763 |
| Threshold Macro-F1 | 0.868886 |
| Threshold Weighted F1 | 0.939340 |
| Average Coverage | 0.846656 |

Tên phù hợp khi trích dẫn bảng này là **mean cumulative seen-class metric across
training stages** hoặc **average incremental/trajectory metric**. Không nên gọi
`0.829565` là final Macro-F1 của model.

## 6. Model cuối đánh giá riêng từng task group

Bảng này dùng ensemble cuối sau task 7 rồi xem lại nhóm class thuộc từng task ban đầu.
Khác với bảng ở mục 4, tất cả các dòng ở đây đều liên quan đến model cuối.

| Task group | Best Macro-F1 từng đạt | Final Macro-F1 | Forgetting |
| ---: | ---: | ---: | ---: |
| 0 | 0.881662 | 0.725239 | 0.156423 |
| 1 | 0.943949 | 0.727887 | 0.216062 |
| 2 | 0.964134 | 0.774296 | 0.189837 |
| 3 | 0.979296 | 0.521764 | 0.457532 |
| 4 | 0.978278 | 0.462714 | 0.515564 |
| 5 | 0.989506 | 0.270837 | 0.718669 |
| 6 | 0.979982 | 0.191417 | 0.788566 |
| 7 | 0.922776 | 0.922776 | 0.000000 |

Trung bình đơn giản tám giá trị `Final Macro-F1` theo task group là:

```text
Final task-group mean Macro-F1 = 0.574616
```

Giá trị này khác `final seen_all Macro-F1 = 0.543176` vì cách gom mẫu/class khi tính
metric khác nhau. Không được dùng hai số thay thế cho nhau mà không ghi rõ cách tính.

Task 7 có `Final Macro-F1 = 0.922776`, trong khi nhiều old task giảm mạnh. Model vì vậy
học tốt nhóm class mới nhất nhưng giữ kiến thức cũ chưa tốt.

## 7. Forgetting

Forgetting của task `i` được tính:

```text
forgetting_i = best_macro_f1_i - final_macro_f1_i
```

### 7.1. Ensemble forgetting

```text
Mean forgetting trên các old task 0–6 = 0.434665
```

Điều này nghĩa là old task trung bình mất khoảng 43.47 điểm phần trăm Macro-F1 so với
mức tốt nhất từng đạt. Task 7 không được đưa vào `mean_old_tasks`, vì chưa có task nào
sau nó để gây forgetting.

Task 5 và 6 bị ảnh hưởng nặng nhất:

- Task 5: forgetting `0.718669`.
- Task 6: forgetting `0.788566`.

### 7.2. Forgetting của từng member

| Member | Task forgetting mean old tasks | Mean class forgetting cuối | Median class forgetting cuối | Class có F1=0 | Class đánh giá |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.442756 | 0.400877 | 0.428409 | 6 | 178 |
| 1 | 0.439622 | 0.405643 | 0.451279 | 7 | 178 |
| 2 | 0.439809 | 0.392730 | 0.428571 | 5 | 178 |

Tổng hợp ba member:

```text
Member task forgetting     = 0.440729 ± 0.001435 (population std)
Member class forgetting    = 0.399750 ± 0.005332 (population std)
Ensemble task forgetting   = 0.434665
```

Ba member có forgetting gần nhau, cho thấy hiện tượng này không chỉ do một member seed
bất thường. Ensemble cải thiện nhẹ so với trung bình member nhưng chưa khắc phục được
catastrophic forgetting.

## 8. Ensemble diversity và uncertainty

| Stage | Samples | Members | Pairwise disagreement | Mutual information | Member-normalized MI | Total probability variance |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 126 | 3 | 0.100529 | 0.077747 | 0.070768 | 0.045248 |
| 1 | 288 | 3 | 0.064815 | 0.057577 | 0.052404 | 0.029358 |
| 2 | 684 | 3 | 0.059942 | 0.048071 | 0.043754 | 0.026887 |
| 3 | 1,641 | 3 | 0.065001 | 0.053619 | 0.048804 | 0.018328 |
| 4 | 3,685 | 3 | 0.042877 | 0.041434 | 0.037711 | 0.009329 |
| 5 | 8,049 | 3 | 0.054955 | 0.041376 | 0.037658 | 0.015551 |
| 6 | 23,756 | 3 | 0.097267 | 0.053822 | 0.048989 | 0.027855 |
| 7 | 173,614 | 3 | 0.125347 | 0.073043 | 0.066486 | 0.033632 |

### `mean_pairwise_disagreement`

Với ba member có ba cặp `(0,1)`, `(0,2)` và `(1,2)`. Metric là tỷ lệ hai member trong
mỗi cặp dự đoán nhãn khác nhau, rồi lấy trung bình ba cặp. Task 7 có disagreement
`0.125347`, nghĩa là khoảng 12.53% quyết định giữa các cặp member không giống nhau.

### `mutual_information`

```text
MI = H(mean_member_probability) - mean(H(member_probability))
```

MI gần 0 nghĩa là các member có phân phối xác suất tương tự. MI cao hơn thể hiện bất
đồng giữa model và thường được dùng làm tín hiệu epistemic uncertainty.

### `member_normalized_mi`

```text
member_normalized_mi = MI / log(number_of_members)
```

Với ba member, giá trị được chuẩn hóa theo `log(3)`. Các giá trị khoảng 0.038–0.071 cho
thấy có diversity nhưng nhìn chung ba member vẫn khá giống nhau.

### `total_probability_variance`

Tính variance xác suất giữa các member trên từng class rồi cộng qua các class. Giá trị
càng lớn nghĩa là phân phối dự đoán của các member càng khác nhau.

Diversity chỉ chứng minh member khác nhau; nó chưa chứng minh uncertainty phát hiện lỗi
tốt. Để đánh giá UE cần thêm error-detection AUROC/AUPRC, AURC, risk–coverage và so sánh
uncertainty của prediction đúng với prediction sai.

## 9. Giải thích các metric classification và calibration

### Accuracy

Tỷ lệ tổng số mẫu dự đoán đúng. Dễ bị chi phối bởi class phổ biến trong dữ liệu mất cân
bằng.

### Balanced Accuracy

Trung bình recall của từng class, mỗi class có trọng số bằng nhau. Khoảng cách lớn giữa
Accuracy `0.816904` và Balanced Accuracy `0.499500` ở model cuối là dấu hiệu model làm
tốt hơn đáng kể trên một số class so với các class còn lại.

### Macro-F1

Tính F1 riêng cho từng class rồi lấy trung bình đều. Đây là metric quan trọng cho DDI
mất cân bằng vì class hiếm không bị class phổ biến lấn át về trọng số.

### Weighted F1

Tính F1 theo từng class nhưng trọng số tỷ lệ với số mẫu của class. Weighted F1 cao hơn
Macro-F1 cho thấy model hoạt động tốt hơn trên các class nhiều mẫu.

### ECE

Expected Calibration Error so sánh confidence với accuracy quan sát được theo các bin.
Trong pipeline này ECE sử dụng maximum ensemble probability làm confidence. Càng gần 0
càng tốt, nhưng ECE thấp không đảm bảo Accuracy hoặc Macro-F1 cao.

### Brier score

Sai số bình phương giữa toàn bộ vector probability và nhãn one-hot thật. Càng thấp càng
tốt.

### NLL

Negative Log-Likelihood phạt mạnh dự đoán sai nhưng quá tự tin. Càng thấp càng tốt.

### Coverage

```text
coverage = selected_count / total_count
```

Coverage là tỷ lệ mẫu model chấp nhận dự đoán sau threshold. Metric selective cao phải
được đọc cùng coverage.

## 10. Đánh giá tổng thể

### Điểm tích cực

- Full Accuracy cuối đạt `0.816904` trên 173,614 test samples.
- Ba member tạo diversity thật và ensemble cải thiện forgetting nhẹ so với trung bình
  từng member.
- Model học tốt nhóm class task cuối với task-group Macro-F1 `0.922776`.
- Pipeline threshold tuân thủ nguyên tắc chọn trên validation và đánh giá trên test.

### Hạn chế chính

- Final seen-all Macro-F1 chỉ `0.543176` và Balanced Accuracy chỉ `0.499500`.
- Mean old-task forgetting `0.434665` là cao.
- Task 5 và 6 mất lần lượt khoảng 71.87 và 78.86 điểm phần trăm Macro-F1.
- Có 5–7 class đạt F1 bằng 0 tùy member ở thời điểm cuối.
- Threshold task cuối giữ gần 97% mẫu nhưng chỉ cải thiện Macro-F1 khoảng 0.0106.
- Diversity hiện có chưa đủ để kết luận UE phát hiện prediction sai tốt.

Kết luận phù hợp là: Ensemble3 Replay-Distill đạt hiệu năng tốt ở các giai đoạn đầu và
trên nhóm class mới nhất, nhưng khả năng duy trì hiệu năng class-balanced sau toàn bộ
P3 trajectory còn hạn chế. Ensemble giảm forgetting nhẹ nhưng chưa giải quyết được
catastrophic forgetting.

## 11. So sánh với bảng kết quả nhiều seed

Kết quả chính của run này tương ứng với **một giá trị seed 0** trong bảng tổng hợp nhiều
seed:

```text
Macro-F1          = task_id 7 / full_macro_f1
Balanced Accuracy = task_id 7 / full_balanced_accuracy
Accuracy          = task_id 7 / full_accuracy
Weighted F1       = task_id 7 / full_weighted_f1
Forgetting        = ensemble mean_old_tasks
```

Không được dùng ba member như ba experiment seed để tính `mean ± std`. Muốn có bảng
tương đương study 5 seed, cần chạy toàn bộ Ensemble3 với experiment seed 0–4. Mỗi seed
tạo một kết quả model cuối, sau đó mới tính mean và sample standard deviation qua năm
kết quả đó.

## 12. Nguồn dữ liệu và tạo lại báo cáo

Tài liệu được tổng hợp từ:

- `ensemble3_p3_task_summary.csv`;
- `ensemble3_p3_paper_table.csv`;
- `ensemble3_p3_final_report.md`;
- offline ensemble artifacts;
- frozen threshold reports;
- member/task/class forgetting artifacts.

Sau khi pull phiên bản repo mới lên server, có thể dựng lại báo cáo từ artifact đã có
mà không train và không chạy inference:

```bash
python src/eval/report.py \
  --full-root outputs/full/tddi_ensemble3_replay_distill_p3_seed0_8tasks_v1 \
  --task-file study_assets/task_protocols/tail_to_head_tasks.json \
  --overwrite
```

Các file được tạo tại:

```text
outputs/full/tddi_ensemble3_replay_distill_p3_seed0_8tasks_v1/final_results/
```
