# Báo cáo kết quả và phân tích thực nghiệm S01

## 1. Mục đích của báo cáo

S01 được thực hiện để xây dựng nền tảng đánh giá theo từng class cho bài toán Class-Incremental Learning trên dữ liệu T-DDI. Sau mỗi task, mô hình được đánh giá trên toàn bộ các class đã học và lưu lại:

- precision
- recall
- F1
- confidence
- entropy
- số mẫu train và test của từng class
- task mà class bắt đầu xuất hiện
- mức độ quên của từng class qua thời gian

Báo cáo này tổng hợp hai nguồn kết quả:

1. **Kết quả đầy đủ của S01 trên 4 phương pháp × 5 seeds**. Các số liệu được trình bày dưới dạng trung bình ± độ lệch chuẩn.
2. **Phân tích sâu trên seed 0**. Phần này dùng dữ liệu class-level, memory summary và task-level metrics để giải thích rõ hơn nguyên nhân của các kết quả quan sát được.

Các kết luận từ năm seeds có độ tin cậy cao hơn. Các phân tích chỉ dựa trên seed 0 được xem là bằng chứng chẩn đoán ban đầu, dùng để hiểu cơ chế xảy ra bên trong mô hình.

---

## 2. Thiết lập thực nghiệm

S01 gồm bốn phương pháp:

- `sequential`: học lần lượt từng task, không lưu dữ liệu cũ.
- `replay`: lưu một số exemplar của class cũ và đưa chúng vào quá trình học task mới.
- `replay_distill`: kết hợp exemplar replay với knowledge distillation.
- `joint_seen`: sau mỗi task, mô hình được truy cập toàn bộ dữ liệu của tất cả class đã thấy. Đây là phương pháp tham chiếu gần với upper bound.

Cấu trúc task:

- task đầu tiên có 38 class
- mỗi task tiếp theo thêm 20 class
- tổng cộng có 8 task và 178 class

Thực nghiệm đầy đủ gồm:

```text
methods = sequential, replay, replay_distill, joint_seen
seeds   = 0, 1, 2, 3, 4
tasks   = 8
runs    = 20
```

Tổng số dòng class-level được tạo ra là 17.280 cho `class_trajectory.csv` và 17.280 cho `class_forgetting.csv`.

---

## 3. Các chỉ số chính

### 3.1. Accuracy

Accuracy là tỷ lệ tất cả mẫu test được dự đoán đúng:

$$
\mathrm{Accuracy}
=
\frac{\text{số dự đoán đúng}}{\text{tổng số mẫu}}
$$

Vì dữ liệu bị mất cân bằng rất mạnh, Accuracy bị chi phối nhiều bởi các class có nhiều mẫu.

### 3.2. Macro-F1

F1 của một class là trung bình điều hòa của precision và recall:

$$
F1_c
=
\frac{2\,\mathrm{Precision}_c\,\mathrm{Recall}_c}
{\mathrm{Precision}_c+\mathrm{Recall}_c}
$$

Macro-F1 là trung bình F1 của tất cả class:

$$
\mathrm{MacroF1}
=
\frac{1}{|\mathcal C|}
\sum_{c\in\mathcal C}F1_c
$$

Mỗi class có trọng số như nhau, nên Macro-F1 phù hợp để đánh giá cả class phổ biến và class hiếm.

### 3.3. Weighted-F1

Weighted-F1 cũng lấy trung bình F1, nhưng class có nhiều mẫu test sẽ có trọng số lớn hơn. Vì vậy, metric này phản ánh tốt hơn hiệu năng trên các class phổ biến.

### 3.4. Balanced Accuracy

Trong bài toán multiclass, Balanced Accuracy có thể hiểu gần như recall trung bình giữa các class:

$$
\mathrm{BalancedAccuracy}
\approx
\frac{1}{|\mathcal C|}
\sum_{c\in\mathcal C}\mathrm{Recall}_c
$$

Nếu Balanced Accuracy cao nhưng Macro-F1 thấp, mô hình có thể tìm đúng được nhiều mẫu của từng class nhưng đồng thời tạo ra rất nhiều false positive.

### 3.5. Class-wise forgetting

Mức độ quên của class $c$ sau task $t$ được tính bằng:

$$
\mathrm{Forgetting}_{c,t}
=
\max_{\tau\leq t}F1_{c,\tau}
-
F1_{c,t}
$$

Ví dụ, nếu F1 tốt nhất trước đây của một class là 0,80 nhưng sau task cuối chỉ còn 0,30, forgetting của class đó là 0,50.

---

## Phần I. Kết quả tổng thể của S01

### 4. Hiệu năng cuối cùng sau 8 task

Kết quả dưới đây là trung bình ± độ lệch chuẩn trên 5 seeds.

| Phương pháp | Accuracy | Macro-F1 | Weighted-F1 | Balanced Accuracy |
| --- | ---: | ---: | ---: | ---: |
| `joint_seen` | 0,9266 ± 0,0046 | 0,8344 ± 0,0072 | 0,9262 ± 0,0047 | 0,8264 ± 0,0078 |
| `replay` | 0,1905 ± 0,0587 | 0,2682 ± 0,0267 | 0,1452 ± 0,0288 | 0,6383 ± 0,0179 |
| `replay_distill` | 0,2235 ± 0,0431 | 0,2611 ± 0,0217 | 0,1636 ± 0,0270 | 0,6110 ± 0,0166 |
| `sequential` | 0,1368 ± 0,1000 | 0,0341 ± 0,0130 | 0,0604 ± 0,0507 | 0,0785 ± 0,0308 |

Kết quả cho thấy ba mức hiệu năng rất rõ:

1. `joint_seen` cao hơn rất nhiều so với các phương pháp CIL.
2. `replay` và `replay_distill` tốt hơn rõ rệt so với `sequential`.
3. dù có replay, khoảng cách với `joint_seen` vẫn còn rất lớn.

Khoảng cách Macro-F1 giữa `joint_seen` và replay là:

$$
0,8344-0,2682=0,5662
$$

Khoảng cách này cho thấy mô hình và dữ liệu có thể đạt hiệu năng khá tốt khi được xem lại toàn bộ dữ liệu. Phần suy giảm lớn chủ yếu xuất hiện khi dữ liệu cũ bị giới hạn trong quá trình học tăng dần.

---

### 5. Forgetting và khả năng duy trì class

| Phương pháp | Task-level forgetting | Final class-wise forgetting | Số class có F1 = 0 ở task cuối |
| --- | ---: | ---: | ---: |
| `joint_seen` | 0,0229 ± 0,0050 | 0,0574 ± 0,0081 | 4,6 ± 1,5 |
| `replay` | 0,2381 ± 0,0246 | 0,2161 ± 0,0283 | 1,4 ± 0,5 |
| `replay_distill` | 0,2008 ± 0,0285 | 0,1979 ± 0,0331 | 2,0 ± 0,0 |
| `sequential` | 0,8269 ± 0,0796 | 0,4531 ± 0,0478 | 157,2 ± 4,0 |

Có hai kết luận chính:

- `sequential` bị catastrophic forgetting rất mạnh.
- replay giúp gần như tất cả class còn giữ được ít nhất một phần khả năng dự đoán.

Tuy nhiên, số class có F1 khác 0 không đồng nghĩa với việc các class đã được phân biệt tốt. Replay làm class không biến mất hoàn toàn, nhưng Macro-F1 vẫn thấp.

---

## Phần II. Phân tích các vấn đề quan sát được

### 6. Sequential gần như chỉ nhớ các class mới nhất

Sau task cuối, `sequential` có trung bình 157,2 trên tổng số 178 class có F1 bằng 0.

Số class còn F1 lớn hơn 0 chỉ khoảng:

$$
178-157,2=20,8
$$

Mỗi task mới thêm đúng 20 class. Vì vậy, 20,8 class còn hoạt động gần bằng số class của task cuối.

Điều này gợi ý rằng mô hình sequential gần như chỉ nhận diện được các class vừa học gần đây và mất phần lớn kiến thức của các task trước.

#### Cơ sở lý thuyết

Khi chỉ học dữ liệu task mới, gradient được tính từ các class mới:

$$
\theta_t
=
\theta_{t-1}
-
\eta\nabla_\theta\mathcal L_{\mathrm{new}}
$$

Không có thành phần loss nào nhắc mô hình rằng các class cũ vẫn phải được giữ lại. Vì vậy, các tham số và decision boundary được điều chỉnh để phù hợp với class mới, làm hiệu năng trên class cũ giảm nhanh.

Đây là biểu hiện điển hình của catastrophic forgetting.

---

### 7. Replay giải quyết tốt class coverage nhưng chưa giải quyết tốt class discrimination

Ở task cuối:

- `replay` chỉ còn trung bình 1,4 class có F1 bằng 0.
- `replay_distill` chỉ còn 2 class có F1 bằng 0.
- `sequential` có khoảng 157 class có F1 bằng 0.

Điều này cho thấy replay giúp gần như mọi class còn có một số dự đoán đúng.

Tuy nhiên:

- Macro-F1 của replay chỉ là 0,2682.
- Macro-F1 của replay-distill chỉ là 0,2611.

Có thể tách vấn đề thành hai khái niệm:

#### Class coverage

Mô hình có còn dự đoán đúng được ít nhất một số mẫu của class hay không.

#### Class discrimination

Mô hình có phân biệt class đó chính xác với các class khác hay không.

Replay cải thiện class coverage rất mạnh, nhưng class discrimination vẫn yếu. Nói cách khác, mô hình còn nhớ rằng class tồn tại nhưng decision boundary của các class chưa đủ chính xác.

---

### 8. Replay có recall tương đối cao nhưng precision thấp

Kết quả trung bình 5 seeds của `replay`:

- Balanced Accuracy: 0,6383
- Macro-F1: 0,2682
- Accuracy: 0,1905
- Weighted-F1: 0,1452

Balanced Accuracy cao hơn Macro-F1 khoảng:

$$
0,6383-0,2682=0,3701
$$

Khoảng cách này rất lớn.

Vì Balanced Accuracy gần với recall trung bình theo class, kết quả cho thấy mô hình tìm đúng được khá nhiều mẫu thật của nhiều class. Tuy nhiên, F1 thấp chứng tỏ precision của nhiều class thấp.

#### Ví dụ đơn giản

Giả sử class A có 10 mẫu test:

- mô hình dự đoán đúng 8 mẫu
- recall là $8/10=0,8$
- nhưng mô hình dự đoán tổng cộng 100 mẫu là class A
- chỉ 8 trong số 100 prediction là đúng
- precision là $8/100=0,08$

Khi đó:

$$
F1
=
\frac{2\times0,08\times0,8}{0,08+0,8}
\approx0,145
$$

Recall cao nhưng F1 vẫn thấp vì có quá nhiều false positive.

Kết quả S01 cho thấy replay có hành vi tương tự: mô hình dự đoán được nhiều class nhưng gán quá nhiều mẫu vào các class đó.

---

### 9. New-class bias rất mạnh

Phân tích sâu seed 0 cho thấy các class của task cuối chỉ chiếm khoảng 3,66% số mẫu test. Tuy nhiên, tỷ lệ prediction được gán vào các class task cuối rất cao.

| Phương pháp | Tỷ lệ thật của task 7 | Tỷ lệ prediction vào task 7 | Mức over-prediction |
| --- | ---: | ---: | ---: |
| `joint_seen` | 3,66% | 3,72% | 1,02 lần |
| `replay_distill` | 3,66% | 44,62% | 12,20 lần |
| `replay` | 3,66% | 69,01% | 18,86 lần |
| `sequential` | 3,66% | 86,46% | 23,64 lần |

`joint_seen` dự đoán gần đúng với tỷ lệ thật. Ngược lại, các phương pháp CIL gán quá nhiều prediction vào các class mới.

#### Kết quả theo old class và current-task class ở seed 0

| Phương pháp | Nhóm class | Macro-Precision | Macro-Recall | Macro-F1 |
| --- | --- | ---: | ---: | ---: |
| `replay` | Class cũ | 0,321 | 0,630 | 0,304 |
| `replay` | Class task hiện tại | 0,042 | 0,945 | 0,078 |
| `replay_distill` | Class cũ | 0,334 | 0,599 | 0,315 |
| `replay_distill` | Class task hiện tại | 0,053 | 0,903 | 0,095 |

Các class mới có recall rất cao, khoảng 0,90–0,95. Nhưng precision chỉ khoảng 0,04–0,05.

Điều này nghĩa là:

> Mô hình tìm được hầu hết mẫu thật của class mới, nhưng cũng gán rất nhiều mẫu của class cũ thành class mới.

#### Cơ sở lý thuyết

Trong mỗi task:

- class mới có toàn bộ dữ liệu thật
- class cũ chỉ được đại diện bằng memory nhỏ
- gradient từ class mới thường xuất hiện nhiều hơn
- classifier mở rộng vùng quyết định của class mới
- một phần vùng của class cũ bị class mới chiếm lấy

Nếu gọi $N_{\mathrm{new}}$ là số lượt mẫu mới và $N_{\mathrm{old}}$ là số lượt replay, khi:

$$
N_{\mathrm{new}}\gg N_{\mathrm{old}}
$$

loss tổng sẽ chịu ảnh hưởng lớn hơn từ class mới. Kết quả là mô hình có xu hướng dự đoán về class mới, ngay cả với mẫu cũ.

---

### 10. Rare class vốn khó ngay cả khi dùng toàn bộ dữ liệu

Để phân biệt độ khó tự nhiên của rare class với forgetting do CIL, seed 0 của `joint_seen` được phân tích theo số lượng mẫu train.

| Số mẫu train | Số class | Macro-Precision | Macro-Recall | Macro-F1 | Mean forgetting | Class F1 = 0 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ≤5 | 9 | 0,778 | 0,778 | 0,763 | 0,144 | 1 |
| 6–10 | 11 | 0,750 | 0,576 | 0,608 | 0,173 | 1 |
| 11–20 | 25 | 0,738 | 0,663 | 0,680 | 0,134 | 4 |
| 21–50 | 19 | 0,883 | 0,801 | 0,830 | 0,044 | 0 |
| 51–200 | 29 | 0,840 | 0,798 | 0,814 | 0,044 | 0 |
| 201–1000 | 46 | 0,913 | 0,926 | 0,919 | 0,015 | 0 |
| >1000 | 39 | 0,936 | 0,941 | 0,939 | 0,008 | 0 |

Nhìn chung, class có nhiều dữ liệu đạt F1 cao và ổn định hơn.

Tương quan Spearman giữa $\log_{10}(\text{train count})$ và final F1 là khoảng:

$$
\rho=0,295,\qquad p<0,001
$$

Mối tương quan dương cho thấy số lượng mẫu càng lớn thì F1 thường càng cao.

#### Ý nghĩa

Rare class có thể có F1 thấp vì:

- quá ít mẫu để học được pattern ổn định
- một hoặc hai mẫu có thể không đại diện cho toàn bộ class
- class dễ chồng lấn với class khác
- nhãn có thể nhiễu
- test set quá nhỏ làm metric dao động mạnh

Do đó, F1 thấp của rare class không tự động có nghĩa là class bị catastrophic forgetting. Một phần vấn đề đã tồn tại ngay cả trong `joint_seen`.

---

### 11. Tất cả class có F1 bằng 0 trong joint-seen seed 0 đều rất ít dữ liệu

Sáu class có final F1 bằng 0 trong `joint_seen` seed 0 là:

| Class | First task | Train count | Test count | Best F1 | Final F1 | Forgetting |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 104 | 0 | 4 | 1 | 1,000 | 0 | 1,000 |
| 155 | 4 | 8 | 3 | 0,800 | 0 | 0,800 |
| 97 | 2 | 11 | 3 | 0,571 | 0 | 0,571 |
| 187 | 4 | 14 | 5 | 0,000 | 0 | 0,000 |
| 174 | 1 | 17 | 6 | 0,222 | 0 | 0,222 |
| 208 | 0 | 17 | 6 | 1,000 | 0 | 1,000 |

Tất cả đều có không quá 17 mẫu train.

Class 187 có best F1 bằng 0 và forgetting bằng 0. Điều này nghĩa là class này chưa từng được học tốt, chứ không phải đã học tốt rồi mới bị quên.

Ngược lại, class 104 từng có F1 bằng 1 nhưng cuối cùng bằng 0. Tuy nhiên, class này chỉ có một mẫu test. Chỉ cần prediction duy nhất thay đổi từ đúng sang sai là F1 chuyển từ 1 xuống 0.

Hai trường hợp trên cho thấy cần phân biệt:

- **class chưa từng học được**
- **class đã học được nhưng bị quên**

---

### 12. Rare class bị dự đoán quá nhiều dưới replay

Gộp 64 class có không quá 50 mẫu train trong seed 0:

| Phương pháp | Macro-Precision | Macro-Recall | Macro-F1 | Số mẫu test thật | Số prediction ước lượng |
| --- | ---: | ---: | ---: | ---: | ---: |
| `joint_seen` | 0,789 | 0,705 | 0,724 | 370 | 313 |
| `replay` | 0,148 | 0,875 | 0,233 | 370 | 5.509 |
| `replay_distill` | 0,188 | 0,834 | 0,270 | 370 | 5.608 |

Số prediction ước lượng được tái dựng từ class-level precision và recall:

$$
TP_c
=
\mathrm{Recall}_c\times N_c^{\mathrm{test}}
$$

$$
\widehat N_c^{\mathrm{pred}}
=
\frac{TP_c}{\mathrm{Precision}_c}
$$

Với replay, nhóm rare class chỉ có 370 mẫu thật nhưng được dự đoán khoảng 5.509 lần:

$$
\frac{5.509}{370}\approx14,9
$$

lần so với số mẫu thật.

#### Diễn giải

Rare class không bị bỏ qua theo nghĩa mô hình không dự đoán chúng. Ngược lại:

- recall của rare class rất cao
- precision rất thấp
- mô hình gán nhiều mẫu của class khác vào rare class

Vấn đề chính là over-prediction, không phải thiếu prediction.

---

### 13. Cơ chế memory có thể giải thích rare-class over-prediction

Memory seed 0 sử dụng quy tắc gần như:

$$
M_c=\min(n_c,50)
$$

Trong đó:

- $n_c$ là số mẫu train thật của class $c$
- $M_c$ là số exemplar của class đó được giữ trong memory

Ví dụ:

- class có 5 mẫu được giữ đủ 5 mẫu
- class có 20 mẫu được giữ đủ 20 mẫu
- class có 1.000 mẫu chỉ giữ 50 mẫu
- class có 73.634 mẫu cũng chỉ giữ 50 mẫu

Tổng memory tăng theo số class:

| Task | Số class đã thấy | Total memory |
| ---: | ---: | ---: |
| 0 | 38 | 1.253 |
| 1 | 58 | 2.113 |
| 2 | 78 | 2.805 |
| 3 | 98 | 3.694 |
| 4 | 118 | 4.307 |
| 5 | 138 | 5.095 |
| 6 | 158 | 5.971 |
| 7 | 178 | 6.801 |

#### Memory làm thay đổi class prior

Trong toàn bộ dữ liệu train seed 0:

- 64 class có không quá 50 mẫu chỉ có tổng cộng 1.101 mẫu
- chúng chiếm khoảng 0,21% toàn bộ dữ liệu train
- vì tất cả đều được giữ lại, chúng cũng chiếm 1.101 trên 6.801 mẫu memory
- tương đương khoảng 16,19% memory

Mức hiện diện tương đối của rare class trong memory được khuếch đại khoảng:

$$
\frac{16,19\%}{0,21\%}\approx76,6
$$

lần so với dữ liệu train gốc.

#### Cơ sở lý thuyết

Một classifier học không chỉ pattern $P(x\mid y=c)$ mà còn chịu ảnh hưởng của tần suất class trong dữ liệu train:

$$
P_{\mathrm{train}}(y=c\mid x)
\propto
P(x\mid y=c)P_{\mathrm{train}}(y=c)
$$

Khi rare class xuất hiện trong replay nhiều hơn rất nhiều so với tỷ lệ thật, mô hình học một prior ngầm rằng rare class phổ biến hơn thực tế.

Test set vẫn giữ phân bố long-tailed ban đầu. Vì vậy xuất hiện train–test prior mismatch:

$$
P_{\mathrm{replay}}(y)
\neq
P_{\mathrm{test}}(y)
$$

Kết quả là mô hình mở rộng vùng dự đoán của rare class, làm recall tăng nhưng false positive cũng tăng mạnh.

---

### 14. Tỷ lệ dữ liệu được giữ liên quan mạnh với recall và precision

Với mỗi class, định nghĩa retention ratio:

$$
r_c
=
\frac{M_c}{n_c}
$$

Trong replay seed 0, tương quan Spearman cho thấy:

- retention ratio và recall:

$$
\rho=0,679
$$

- retention ratio và precision:

$$
\rho=-0,543
$$

- retention ratio và F1 gần bằng 0:

$$
\rho\approx-0,007
$$

Diễn giải:

- class được giữ lại tỷ lệ dữ liệu càng lớn thì recall thường càng cao
- nhưng precision thường càng thấp
- lợi ích về recall bị bù lại bởi sự gia tăng false positive
- vì vậy F1 không cải thiện rõ ràng

Đây là bằng chứng class-level phù hợp với kết quả tổng thể: replay bảo vệ khả năng gọi tên class nhưng làm vùng dự đoán trở nên quá rộng.

---

### 15. Head class bị mất hiệu năng mạnh do memory compression

Nhóm 39 class có hơn 1.000 mẫu trong seed 0:

| Phương pháp | Macro-Precision | Macro-Recall | Macro-F1 |
| --- | ---: | ---: | ---: |
| `joint_seen` | 0,936 | 0,941 | 0,939 |
| `replay` | 0,492 | 0,302 | 0,216 |
| `replay_distill` | 0,469 | 0,306 | 0,211 |

Mức giảm từ `joint_seen` xuống `replay` là:

$$
0,939-0,216=0,723
$$

Trong khi nhóm class ≤50 mẫu giảm:

$$
0,724-0,233=0,491
$$

Như vậy, trong seed 0, head class chịu CIL degradation lớn hơn nhóm rare class.

#### Cơ sở lý thuyết

Head class có thể có phân bố rất rộng, gồm nhiều kiểu drug pair khác nhau. Khi hàng nghìn hoặc hàng chục nghìn mẫu bị nén xuống 50 exemplar, memory chỉ giữ được một phần nhỏ distribution.

Ví dụ, class lớn nhất có 73.634 mẫu nhưng chỉ giữ 50:

$$
\frac{50}{73.634}\approx0,068\%
$$

Nếu các exemplar không bao phủ đủ các vùng khác nhau trong latent space, khi đánh giá:

- một phần mẫu head class vẫn nằm gần exemplar và được dự đoán đúng
- nhiều mẫu khác nằm xa vùng memory đã giữ
- chúng bị gán sang class mới hoặc rare class
- recall của head class giảm mạnh

Điều này cho thấy hai khái niệm cần được phân biệt về mặt diễn giải:

- **rarity:** class có ít dữ liệu ban đầu
- **memory compression:** class bị mất bao nhiêu phần thông tin khi đưa vào memory

Rare class có rarity cao nhưng compression thấp. Head class có rarity thấp nhưng compression rất cao.

---

### 16. Replay-distillation thể hiện stability–plasticity trade-off

Kết quả trung bình 5 seeds:

| Chỉ số | Replay | Replay-distill | Thay đổi khi thêm distillation |
| --- | ---: | ---: | ---: |
| Accuracy | 0,1905 | 0,2235 | +0,0330 |
| Weighted-F1 | 0,1452 | 0,1636 | +0,0184 |
| Balanced Accuracy | 0,6383 | 0,6110 | −0,0273 |
| Macro-F1 | 0,2682 | 0,2611 | −0,0071 |
| Task-level forgetting | 0,2381 | 0,2008 | −0,0373 |
| Class-wise forgetting | 0,2161 | 0,1979 | −0,0182 |

Distillation giúp:

- giảm forgetting
- tăng Accuracy
- tăng Weighted-F1
- giảm bớt new-class bias trong seed 0

Nhưng đồng thời:

- Balanced Accuracy giảm
- Macro-F1 trung bình giảm nhẹ

#### Cơ sở lý thuyết

Knowledge distillation thêm một ràng buộc để prediction của model mới không thay đổi quá xa model cũ:

$$
\mathcal L
=
\mathcal L_{\mathrm{classification}}
+
\lambda_{KD}\mathcal L_{KD}
$$

Thành phần distillation tăng **stability**, tức khả năng giữ kiến thức cũ. Tuy nhiên, nếu ràng buộc quá mạnh, mô hình khó thay đổi để học class mới, làm giảm **plasticity**.

Đây là stability–plasticity trade-off:

- stability quá thấp → quên class cũ
- plasticity quá thấp → học class mới không tốt
- cần cân bằng hai mục tiêu

#### So sánh Macro-F1 theo từng seed

Chênh lệch `replay - replay_distill` trên 5 seeds có trung bình:

$$
\Delta=0,00716
$$

Paired t-test thăm dò cho:

$$
p\approx0,357
$$

Khoảng tin cậy 95% của chênh lệch xấp xỉ:

$$
[-0,0119,\,0,0263]
$$

Khoảng tin cậy chứa 0, nên S01 chưa cho thấy bằng chứng rõ rằng một trong hai phương pháp tốt hơn về final Macro-F1. Hai phương pháp có ưu thế ở các metric khác nhau.

---

### 17. Raw forgetting của ultra-rare class có thể rất nhiễu

Công thức forgetting dùng F1 tốt nhất trong quá khứ:

$$
\mathrm{Forgetting}_{c,t}
=
\max_{\tau\leq t}F1_{c,\tau}
-F1_{c,t}
$$

Với class có test set đủ lớn, metric này phản ánh khá tốt sự suy giảm qua thời gian. Nhưng với class chỉ có một vài mẫu test, F1 có thể thay đổi mạnh chỉ vì một prediction.

#### Ví dụ class 104

- train count: 4
- test count: 1
- best F1: 1
- final F1: 0
- forgetting: 1

Vì chỉ có một mẫu test:

- dự đoán đúng → F1 có thể bằng 1
- dự đoán sai → F1 bằng 0

Do đó, forgetting bằng 1 có thể xuất hiện chỉ vì một mẫu chuyển từ đúng sang sai. Nó chưa chắc phản ánh toàn bộ representation của class đã bị phá hủy.

#### Ý nghĩa

Forgetting của rare class chứa đồng thời:

1. forgetting thật của mô hình
2. độ nhiễu do số mẫu test quá nhỏ

Vì vậy, một class có forgetting cao và test count bằng 1 không có cùng độ tin cậy với một class có forgetting cao trên hàng nghìn mẫu test.

---

### 18. Joint-seen giúp phân biệt intrinsic difficulty và CIL degradation

`joint_seen` vẫn có trung bình khoảng 4,6 class F1 bằng 0, dù mô hình được truy cập toàn bộ dữ liệu đã thấy.

Điều này chứng tỏ một số class có thể vốn khó vì:

- quá ít dữ liệu
- đặc trưng hóa học không đủ phân biệt
- overlap với class khác
- label noise
- test support quá nhỏ

Do đó, cần phân biệt:

#### Intrinsic difficulty

Class có F1 thấp ngay cả trong joint training:

$$
F1_c^{\mathrm{joint}}\text{ thấp}
$$

#### CIL degradation

Class học tốt trong joint training nhưng giảm mạnh dưới CIL:

$$
\mathrm{CILGap}_c
=
F1_c^{\mathrm{joint}}
-
F1_c^{\mathrm{CIL}}
$$

Ví dụ:

- nếu joint F1 = 0 và replay F1 = 0, chưa có bằng chứng replay gây ra thất bại
- nếu joint F1 = 0,9 và replay F1 = 0,1, CIL gap bằng 0,8 và continual learning là nguyên nhân chính

Seed 0 cho thấy CIL gap của head class rất lớn. Điều này củng cố nhận định rằng memory compression và new-class bias là hai vấn đề quan trọng bên cạnh rarity.

---

### 19. Joint-seen vẫn có một mức forgetting nhỏ

Final class-wise forgetting trung bình của `joint_seen` là 0,0574.

Mặc dù joint-seen không bị giới hạn memory như replay, F1 của một class vẫn có thể giảm khi thêm class mới vì:

- bài toán phân loại trở nên khó hơn
- số class cạnh tranh tăng
- decision boundary phải được điều chỉnh
- kết quả training có dao động ngẫu nhiên

Vì vậy, raw forgetting không chỉ đo catastrophic forgetting do mất dữ liệu cũ. Nó còn chứa phần suy giảm tự nhiên khi bài toán mở rộng.

Có thể xem forgetting của joint-seen như một mức nền tham khảo:

| Phương pháp | Raw class-wise forgetting | Phần cao hơn joint-seen |
| --- | ---: | ---: |
| `replay` | 0,2161 | 0,1587 |
| `replay_distill` | 0,1979 | 0,1405 |
| `sequential` | 0,4531 | 0,3957 |

Hiệu số này không nhất thiết là một metric chính thức, nhưng giúp diễn giải rằng không phải toàn bộ raw forgetting đều do memory limitation.

---

### 20. Confidence cao chưa đồng nghĩa với dự đoán đáng tin cậy

Trong seed 0 của joint-seen có một số class F1 thấp nhưng mean confidence cao. Ví dụ:

| Class | Train count | Test count | F1 | Mean confidence | Mean entropy |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 155 | 8 | 3 | 0,000 | 0,938 | 0,268 |
| 12 | 24 | 8 | 0,167 | 0,889 | 0,358 |
| 19 | 79 | 26 | 0,263 | 0,880 | 0,380 |
| 136 | 13 | 4 | 0,333 | 0,902 | 0,341 |

Class 155 có F1 bằng 0 nhưng mean confidence khoảng 0,94. Đây là dấu hiệu mô hình có thể sai nhưng vẫn tự tin.

Tuy nhiên, confidence và entropy trong S01 chưa được calibration. Do đó, các số liệu trên chỉ là dấu hiệu chẩn đoán, chưa đủ để kết luận xác suất của mô hình không đáng tin cậy.

Về lý thuyết:

- confidence là xác suất lớn nhất mà softmax đưa ra
- calibration kiểm tra xem confidence có phù hợp với xác suất dự đoán đúng thật hay không
- một mô hình có thể có confidence 0,9 nhưng accuracy thực tế chỉ 0,5

S01 cho thấy cần cẩn thận khi dùng uncertainty hoặc confidence để giải thích forgetting.

---

## Phần III. Ý nghĩa đối với các câu hỏi nghiên cứu hiện tại

### 21. RQ1 — Multi-prototype

S01 chưa có latent feature nên chưa thể xác nhận một class có thật sự gồm nhiều cluster hay không.

Tuy nhiên, kết quả cho thấy một điểm quan trọng:

- rare class thường được giữ phần lớn hoặc toàn bộ mẫu
- head class bị nén từ hàng nghìn mẫu xuống 50 exemplar
- head class bị giảm F1 và recall rất mạnh

Điều này gợi ý rằng multi-prototype có thể hữu ích không chỉ cho rare class mà đặc biệt cho class:

- có phân bố rộng
- có nhiều nhóm dữ liệu khác nhau
- bị memory compression mạnh
- không thể được mô tả tốt bằng một số ít exemplar

S01 không chứng minh multi-prototype hiệu quả, nhưng cung cấp cơ sở rằng vấn đề representation coverage có thể tồn tại, đặc biệt ở head class.

---

### 22. RQ2 — Rarity-aware replay

Giả thuyết ban đầu là rare class có ít dữ liệu nên dễ bị quên và cần được replay nhiều hơn.

S01 hỗ trợ một phần đầu:

> Rare class vốn khó và kém ổn định hơn trong joint training.

Nhưng seed 0 không hỗ trợ rõ phần sau:

> Rare class đang thiếu replay.

Dưới replay hiện tại:

- rare-class recall rất cao
- rare class được giữ tỷ lệ dữ liệu lớn
- rare class bị over-predict
- precision của rare class rất thấp

Điều này cho thấy rarity và nhu cầu replay không phải lúc nào cũng giống nhau.

Một class có ít dữ liệu nhưng memory giữ được toàn bộ class có thể không thiếu thông tin replay. Ngược lại, một head class có rất nhiều dữ liệu nhưng bị nén xuống 50 exemplar có thể mất phần lớn distribution ban đầu.

Do đó, kết quả S01 làm yếu cách giải thích đơn giản:

$$
\text{ít mẫu}
\Rightarrow
\text{cần nhiều replay hơn}
$$

Vấn đề thực tế có thể phụ thuộc đồng thời vào:

- class frequency
- retention ratio
- memory compression
- class age
- latent complexity
- overlap với class mới
- precision và false-positive burden

---

### 23. RQ3 — Uncertainty dự báo future forgetting

S01 đã lưu mean confidence và entropy theo class, nhưng các giá trị chưa được calibration.

Ngoài ra, S01 cho thấy future degradation có nhiều dạng:

- recall giảm vì class cũ bị class mới chiếm vùng
- precision giảm vì class bị dự đoán quá rộng
- F1 thấp vì class vốn đã khó
- forgetting dao động vì test set quá nhỏ

Do đó, chỉ kiểm tra uncertainty với raw forgetting có thể chưa phản ánh đầy đủ vấn đề.

S01 chưa đủ để trả lời RQ3, nhưng chỉ ra rằng uncertainty cần được diễn giải cùng với:

- current F1
- precision
- recall
- class frequency
- class age
- memory compression
- intrinsic difficulty trong joint-seen

---

### 24. Ý nghĩa đối với main research question

Main research question hiện hướng đến ba mục tiêu:

1. giảm catastrophic forgetting
2. bảo vệ rare class
3. duy trì confidence calibration

S01 xác nhận mục tiêu thứ nhất là có ý nghĩa vì khoảng cách giữa sequential, replay và joint-seen rất lớn.

Tuy nhiên, S01 cho thấy vấn đề không chỉ là rare class bị bỏ quên. Bức tranh thực tế gồm nhiều thành phần:

- sequential quên gần như toàn bộ class cũ
- replay duy trì class coverage nhưng precision thấp
- model bị new-class bias mạnh
- rare class vốn khó nhưng lại bị over-predict dưới replay
- head class bị suy giảm mạnh vì memory compression
- distillation giảm forgetting nhưng tạo stability–plasticity trade-off
- raw forgetting của ultra-rare class chịu nhiễu lớn

Do đó, S01 không trực tiếp xác nhận toàn bộ phương pháp đề xuất, nhưng cung cấp bằng chứng rằng bài toán cần giải quyết đồng thời:

$$
\text{bảo tồn representation}
+
\text{kiểm soát class bias}
+
\text{phân bổ memory/replay hợp lý}
$$

---

## 25. Tổng kết

S01 cho thấy các kết luận chính sau:

### 25.1. Catastrophic forgetting là vấn đề rất nghiêm trọng

Sequential có Macro-F1 chỉ 0,0341 và trung bình 157,2 trên 178 class có F1 bằng 0. Mô hình gần như chỉ còn nhận diện được class của các task mới nhất.

### 25.2. Replay giúp class không biến mất nhưng chưa phân biệt class tốt

Replay giảm số class F1 bằng 0 xuống khoảng 1–2 class. Tuy nhiên, Macro-F1 chỉ khoảng 0,26–0,27, chứng tỏ decision boundary giữa các class vẫn yếu.

### 25.3. Vấn đề nổi bật của replay là precision thấp

Balanced Accuracy cao nhưng Macro-F1 thấp cho thấy recall tương đối tốt nhưng false positive nhiều. Mô hình dự đoán quá rộng vào một số class.

### 25.4. New-class bias rất mạnh

Trong seed 0, class task cuối chỉ chiếm 3,66% test set nhưng nhận 69,01% prediction của replay. Class mới có recall rất cao nhưng precision cực thấp.

### 25.5. Rare class vốn khó nhưng không bị bỏ qua dưới replay

Trong joint-seen, rare class có F1 thấp hơn head class, xác nhận intrinsic difficulty. Nhưng dưới replay, rare-class recall tăng cao và precision giảm mạnh. Rare class bị dự đoán quá nhiều thay vì bị bỏ quên hoàn toàn.

### 25.6. Memory thay đổi class prior

Các class ≤50 mẫu chỉ chiếm 0,21% dữ liệu train nhưng chiếm khoảng 16,19% memory. Mức hiện diện của chúng trong memory được khuếch đại khoảng 76,6 lần, tạo train–test prior mismatch.

### 25.7. Head class chịu memory compression rất mạnh

Class có hơn 1.000 mẫu giảm Macro-F1 từ 0,939 trong joint-seen xuống 0,216 trong replay. Chỉ giữ tối đa 50 exemplar có thể không bao phủ được phân bố phức tạp của head class.

### 25.8. Distillation giảm forgetting nhưng không thắng replay trên mọi metric

Replay-distill có forgetting và Accuracy tốt hơn, nhưng Balanced Accuracy và Macro-F1 trung bình thấp hơn nhẹ. Đây là biểu hiện của stability–plasticity trade-off.

### 25.9. Forgetting của ultra-rare class cần được diễn giải thận trọng

Khi class chỉ có 1–3 mẫu test, một prediction thay đổi có thể làm F1 và forgetting thay đổi cực lớn. Raw forgetting của các class này có độ nhiễu cao.

### 25.10. S01 chưa chứng minh rarity-aware hay multi-prototype là cần thiết

S01 chứng minh catastrophic forgetting, class bias và memory coverage là các vấn đề thật. Tuy nhiên, nó cũng cho thấy rarity không đồng nghĩa trực tiếp với thiếu replay. Multi-prototype và uncertainty-aware replay vẫn cần bằng chứng riêng từ latent representation, prediction-level data và calibration.

Tóm lại, kết quả S01 mô tả một bức tranh phức tạp hơn giả thuyết ban đầu:

> Mô hình không chỉ quên class cũ. Replay giúp duy trì recall nhưng đồng thời làm thay đổi class prior, tạo false positive, new-class bias và suy giảm mạnh ở các class bị nén thông tin. Rare class vốn khó, nhưng trong baseline hiện tại chúng thường bị dự đoán quá nhiều hơn là bị bỏ qua.
