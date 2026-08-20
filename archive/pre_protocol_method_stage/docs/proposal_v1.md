# DDI-CIL Research Proposal

## Background

**1.1. Bài toán T-DDI hiện tại:**

T-DDI là bài toán phân loại loại tương tác của một cặp thuốc dựa trên physicochemical descriptors. Bài báo T-DDI hiện tại mô tả 868.069 cặp thuốc thuộc 178 interaction types, sử dụng explicit chemical descriptors và một uncertainty-aware estimator. Bài báo cũng báo cáo sự khác biệt đáng kể giữa toàn bộ test set và tập dự đoán high-confidence, cho thấy confidence là một phần quan trọng để đưa ra dự đoán chính xác.

Hiện tại, theo như repo DDI-CIL bài toán được chuyển sang Class-Incremental Learning:

$$
\mathcal{T}_1,\mathcal{T}_2,\ldots,\mathcal{T}_8
$$

Trong đó:

- Task đầu có 38 classes
- mỗi task tiếp theo +20 classes mới
- sau mỗi task, mô hình phải nhận diện được tất cả các classes mới và cũ đã được học.

Repo hiện có sequential fine-tuning, exemplar replay, replay-distillation, EWC và joint-seen oracle. Kết quả hiện tại cho thấy replay và replay-distillation mạnh hơn rõ rệt so với sequential và EWC.

→ Nếu phát triển thêm, có thể lấy **replay-distillation làm strong baseline,** không nên chỉ so với các phương pháp naive fine-tuning.

**1.2. Imbalanced problem**

Train split hiện tại có:

- class nhỏ nhất: 3 samples
- class lớn nhất: 73.634 samples
- 9 classes có không quá 5 samples
- 20 classes có không quá 10 samples
- 45 classes có không quá 20 samples
- 64 classes có không quá 50 samples
- imbalance ratio lớn hơn 24.000 lần.

Điều này tạo ra hai tầng imbalance:

**Tầng 1:**

Ngay trong dữ liệu ban đầu, một số interaction types có hàng chục nghìn cặp thuốc, trong khi một số loại chỉ có vài cặp.

**Tầng 2:**

Khi học task mới:

- class mới có toàn bộ dữ liệu thật
- class cũ chỉ còn một memory nhỏ hoặc synthetic replay
- classifier nhận nhiều gradient từ class mới hơn class cũ.

Vì vậy một class cũ, đồng thời cũng là class hiếm (tạm gọi là old-tail class) sẽ chịu hai disadvantage cùng lúc khi training:

$$
\text{ít dữ liệu ban đầu} + \text{ít thông tin replay}
$$

Bài báo Long-tailed Class Incremental Learning cũng chỉ ra rằng hành vi của các phương pháp CIL thay đổi đáng kể khi giả định balanced tasks bị loại bỏ. Imbalance ban đầu và exemplar scarcity (sự thiếu hụt các mẫu đại diện được lưu lại từ những lớp cũ trong continual learning) cùng tạo ra sự bias trong incremental learning.

---

## Problem Modeling

### 1. Mô hình học các loại tương tác thuốc theo từng giai đoạn

Trong bài toán này, mô hình không học toàn bộ 178 loại tương tác thuốc cùng một lúc. Thay vào đó, các loại tương tác được đưa vào lần lượt qua nhiều giai đoạn:

$$
\mathcal{T}_1,\mathcal{T}_2,\ldots,\mathcal{T}_T
$$

Trong đó, $\mathcal{T}_t$ là giai đoạn học thứ $t$. Mỗi giai đoạn cung cấp cho mô hình một nhóm loại tương tác mới, ký hiệu là $\mathcal{C}_t$.

Sau khi hoàn thành giai đoạn $t$, mô hình đã học tập hợp các loại tương tác:

$$
\mathcal{C}_{1:t}
=
\bigcup_{i=1}^{t}\mathcal{C}_i
$$

Khi dự đoán, mô hình phải chọn một nhãn trong tất cả các loại tương tác đã học:

$$
\hat y\in\mathcal{C}_{1:t}
$$

Mô hình không được biết cặp thuốc đang thuộc giai đoạn nào. Vì vậy, sau mỗi giai đoạn, nó vừa phải học các loại tương tác mới, vừa phải tiếp tục nhận diện đúng các loại tương tác cũ.

Mỗi cặp thuốc được biểu diễn bằng một vector đặc trưng:

$$
x\in\mathbb{R}^{d}
$$

Trong repo DDI-CIL hiện tại:

$$
d=3780
$$

Có thể hiểu $x$ là một dãy 3.780 con số mô tả các đặc điểm hóa học của cặp thuốc.

Mô hình trước tiên biến vector này thành một biểu diễn ngắn gọn hơn:

$$
z=f_{\phi}(x)
$$

Trong đó:

- $f_{\phi}$ là phần mạng dùng để xử lý các đặc trưng đầu vào
- $z\in\mathbb{R}^{m}$ là biểu diễn nén của cặp thuốc
- $m$ là số chiều của biểu diễn nén

Từ $z$, bộ phân loại đưa ra loại tương tác được dự đoán:

$$
\hat y=g_{\theta}(z)
$$

Nói đơn giản, quá trình dự đoán gồm hai bước:

$$
\text{đặc trưng của cặp thuốc}
\rightarrow
\text{biểu diễn nén}
\rightarrow
\text{loại tương tác}
$$

---

### 2. Mục tiêu: mô hình không được chỉ học tốt các lớp mới hoặc lớp phổ biến

Sau giai đoạn $t$, mục tiêu là mô hình hoạt động tốt trên tất cả các loại tương tác đã học. Mỗi loại tương tác nên được xem là quan trọng như nhau, dù loại đó có rất nhiều hay rất ít mẫu.

Mục tiêu này có thể viết như sau:

$$
\mathcal{R}^{\mathrm{bal}}_t(\theta)
=
\frac{1}{|\mathcal{C}_{1:t}|}
\sum_{c\in\mathcal{C}_{1:t}}
\mathbb{E}_{z\sim p_c(z)}
\left[
\ell\bigl(g_{\theta}(z),c\bigr)
\right]
$$

Ý nghĩa của công thức:

- $c$ là một loại tương tác thuốc
- $p_c(z)$ mô tả các biểu diễn $z$ thường gặp của loại tương tác $c$
- $\ell$ đo mức độ dự đoán sai của mô hình
- phép lấy trung bình theo số lớp giúp mỗi lớp có mức quan trọng ngang nhau.

Điểm quan trọng là công thức này tính trung bình **theo lớp**, không chỉ theo tổng số mẫu.

Nếu chỉ tính lỗi trên tất cả mẫu rồi lấy trung bình, các lớp có hàng chục nghìn mẫu sẽ ảnh hưởng đến quá trình học nhiều hơn rất nhiều so với các lớp chỉ có vài mẫu. Khi đó, mô hình có thể đạt kết quả tổng thể cao nhưng vẫn gần như không nhận diện được các loại tương tác hiếm.

Vì vậy, mục tiêu phù hợp hơn là:

> Mô hình phải duy trì khả năng nhận diện cả lớp phổ biến lẫn lớp hiếm, thay vì đạt kết quả cao chủ yếu nhờ các lớp có nhiều dữ liệu.
>

---

### 3. Vấn đề xảy ra khi dữ liệu cũ không còn được giữ đầy đủ

Nếu được huấn luyện chung trên toàn bộ dữ liệu, mô hình có thể xem lại tất cả các mẫu của mọi loại tương tác đã học. Tuy nhiên, trong học tăng dần theo lớp, dữ liệu cũ thường không được giữ lại đầy đủ.

Mô hình chỉ có một bộ nhớ nhỏ để đại diện cho các lớp cũ. Vì vậy, với một lớp cũ $c$, phân bố thật:

$$
p_c(z)
$$

được thay bằng một phân bố gần đúng được tạo từ bộ nhớ:

$$
q_c(z)\approx p_c(z)
$$

Trong đó:

- $p_c(z)$ là thông tin đầy đủ về lớp $c$ nếu còn toàn bộ dữ liệu
- $q_c(z)$ là phần thông tin mà bộ nhớ cố gắng giữ lại về lớp đó.

Bộ nhớ có thể lưu:

- một số mẫu thật
- một điểm trung tâm của lớp
- nhiều điểm trung tâm
- hoặc các tham số mô tả cách dữ liệu của lớp phân bố trong không gian biểu diễn.

Khi học giai đoạn mới, mô hình được huấn luyện từ hai nguồn:

1. dữ liệu thật của các lớp mới;
2. dữ liệu được lấy hoặc tạo lại từ bộ nhớ của các lớp cũ.

Có thể biểu diễn mục tiêu huấn luyện này như sau:

$$
\widetilde{\mathcal{R}}_t(\theta)
=
\frac{1}{|\mathcal{C}_{1:t}|}
\left[
\sum_{c\in\mathcal{C}_{\mathrm{old}}}
\mathbb{E}_{z\sim q_c(z)}
\left[
\ell\bigl(g_{\theta}(z),c\bigr)
\right]
+
\sum_{c\in\mathcal{C}_t}
\mathbb{E}_{z\sim p_c(z)}
\left[
\ell\bigl(g_{\theta}(z),c\bigr)
\right]
\right]
$$

Trong đó:

$$
\mathcal{C}_{\mathrm{old}}
=
\mathcal{C}_{1:t-1}
$$

Sự khác biệt ở đây là:

- lớp mới được học từ dữ liệu thật và tương đối đầy đủ
- lớp cũ chỉ được nhắc lại bằng một lượng thông tin nhỏ trong bộ nhớ.

Nếu bộ nhớ không đại diện tốt cho lớp cũ, ranh giới phân loại sẽ dần nghiêng về các lớp mới. Đây là nguyên nhân quan trọng gây ra hiện tượng mô hình quên kiến thức cũ.

---

### 4. Ba câu hỏi chính cần giải quyết

#### 4.1. Bộ nhớ có đại diện đúng cho lớp cũ không?

Với lớp $c$, ta muốn:

$$
q_c(z)\approx p_c(z)
$$

Nói cách khác, phần thông tin được lưu trong bộ nhớ phải gần giống với dữ liệu thật của lớp đó.

Một loại tương tác thuốc có thể gồm nhiều nhóm cặp thuốc khác nhau. Ví dụ, các cặp thuốc cùng có một nhãn tương tác nhưng vẫn có thể khác nhau đáng kể về đặc trưng hóa học.

Giả sử lớp $c$ có ba nhóm dữ liệu:

$$
p_c(z)
=
\sum_{k=1}^{3}\pi_{c,k}p_{c,k}(z),
\qquad
\sum_{k=1}^{3}\pi_{c,k}=1
$$

Trong đó, $p_{c,k}(z)$ mô tả nhóm thứ $k$, còn $\pi_{c,k}$ thể hiện tỷ lệ của nhóm đó trong lớp.

Nếu chỉ lưu một điểm trung tâm chung:

$$
\mu_c
=
\mathbb{E}_{z\sim p_c(z)}[z]
$$

thì điểm trung tâm này có thể nằm ở giữa các nhóm nhưng lại không thật sự gần nhóm nào. Khi đó, một điểm đại diện duy nhất không mô tả tốt toàn bộ lớp.

Giải pháp được đề xuất là lưu nhiều điểm đại diện cho mỗi lớp:

$$
q_c(z)
=
\sum_{k=1}^{K_c}
\pi_{c,k}
\mathcal{N}
\bigl(z;\mu_{c,k},\Sigma_{c,k}\bigr),
\qquad
\sum_{k=1}^{K_c}\pi_{c,k}=1
$$

Trong đó:

- $K_c$ là số nhóm đại diện của lớp $c$;
- $\mu_{c,k}$ là tâm của nhóm thứ $k$;
- $\Sigma_{c,k}$ mô tả mức độ phân tán của nhóm;
- $\pi_{c,k}$ là tỷ lệ của nhóm.

Có thể hiểu đơn giản:

> Thay vì nhớ một gương mặt trung bình cho cả lớp, mô hình nhớ nhiều kiểu mẫu khác nhau của lớp đó.
>

Mục tiêu của việc dùng nhiều điểm đại diện là giúp bộ nhớ bao phủ được nhiều vùng dữ liệu khác nhau của lớp cũ.

---

#### 4.2. Mỗi lớp cũ nên được ôn lại bao nhiêu lần?

Giả sử tại giai đoạn $t$, tổng số mẫu cũ được phép dùng để ôn lại là:

$$
B_t
$$

Số mẫu ôn lại dành cho lớp $c$ được ký hiệu là:

$$
R_{c,t}
$$

Tổng số mẫu của tất cả lớp cũ phải bằng ngân sách đã đặt:

$$
\sum_{c\in\mathcal{C}_{\mathrm{old}}}R_{c,t}
=
B_t
$$

Cách đơn giản nhất là chia đều:

$$
R_{c,t}^{\mathrm{equal}}
=
\frac{B_t}{|\mathcal{C}_{\mathrm{old}}|}
$$

Tuy nhiên, chia đều chưa chắc hợp lý. Một lớp có 50.000 mẫu ban đầu và một lớp chỉ có 5 mẫu không có cùng mức độ dễ bị quên.

Các lớp hiếm thường gặp nhiều khó khăn hơn:

- mô hình chỉ có rất ít dữ liệu để học đặc điểm của lớp;
- điểm đại diện của lớp kém ổn định hơn;
- ranh giới giữa lớp đó với lớp khác dễ thay đổi;
- một vài mẫu trong bộ nhớ có thể bị lặp lại quá nhiều.

Vì vậy, có thể ưu tiên lớp hiếm bằng điểm số:

$$
r_c
=
\frac{1}{(n_c+\varepsilon)^{\alpha}}
$$

Trong đó:

- $n_c$ là số mẫu thật ban đầu của lớp $c$;
- $\alpha$ điều khiển mức độ ưu tiên lớp hiếm;
- $\varepsilon>0$ giúp tránh chia cho 0.

Khi $n_c$ nhỏ, $r_c$ lớn. Điều đó có nghĩa là lớp hiếm sẽ được cấp nhiều lượt ôn lại hơn.

Tuy nhiên, cần đặt giới hạn:

$$
R_{\min}
\leq
R_{c,t}
\leq
R_{\max}
$$

Giới hạn này giúp tránh trường hợp một vài lớp cực hiếm sử dụng gần như toàn bộ ngân sách ôn lại.

Ý tưởng chính là:

> Tổng số mẫu ôn lại không thay đổi, nhưng phần ngân sách dành cho từng lớp được phân chia theo nhu cầu bảo vệ của lớp đó.
>

---

#### 4.3. Làm thế nào biết lớp nào đang có nguy cơ bị quên?

Số lượng mẫu là một tín hiệu quan trọng nhưng chưa đủ.

Hai lớp có cùng 100 mẫu vẫn có thể rất khác nhau:

- một lớp nằm tách xa các lớp khác nên tương đối dễ nhận diện
- một lớp nằm gần nhiều lớp mới nên dễ bị nhầm
- một lớp có dữ liệu phân tán thành nhiều nhóm
- một lớp thường bị mô hình dự đoán sai nhưng mô hình vẫn tỏ ra rất tự tin.

Vì vậy, ngoài độ hiếm, cần đo mức độ không ổn định của từng lớp.

Gọi tập mẫu thuộc lớp $c$ là:

$$
\mathcal{I}_c=\{i:y_i=c\},
\qquad
N_c=|\mathcal{I}_c|
$$

Có thể sử dụng ba tín hiệu sau.

##### a. Entropy

$$
\bar{H}_{c,t}
=
\frac{1}{N_c}
\sum_{i\in\mathcal{I}_c}
\left[
-\sum_{j\in\mathcal{C}_{1:t}}
p_{ij}\log p_{ij}
\right]
$$

Trong đó $p_{ij}$ là xác suất mô hình cho rằng mẫu $i$ thuộc lớp $j$.

Giá trị này cao khi mô hình phân vân giữa nhiều lớp. Khi đó, lớp $c$ có thể cần được bảo vệ nhiều hơn.

##### b. Distance

$$
\bar{D}_{c,t}
=
\frac{1}{N_c}
\sum_{i\in\mathcal{I}_c}
\min_{1\leq k\leq K_c}
d\bigl(z_i,\mu_{c,k}\bigr)
$$

Nếu nhiều mẫu nằm xa tất cả các điểm đại diện của lớp, bộ nhớ hiện tại có thể chưa mô tả tốt lớp đó.

##### c. Mức giảm kết quả của lớp qua thời gian

$$
F_{c,t}
=
\max_{\tau\leq t}M_{c,\tau}
-
M_{c,t}
$$

Trong đó $M_{c,t}$ là kết quả của lớp $c$ sau giai đoạn $t$.

Ví dụ, nếu độ chính xác tốt nhất trước đây của lớp là 80% nhưng hiện tại chỉ còn 55%, mức quên của lớp là 25 điểm phần trăm.

Ba tín hiệu có thể được kết hợp:

$$
u_{c,t}
=
\lambda_H\bar{H}_{c,t}
+
\lambda_D\bar{D}_{c,t}
+
\lambda_F\bar{F}_{c,t}
$$

Trước khi cộng, các giá trị cần được đưa về cùng thang đo.

Điểm $u_{c,t}$ càng cao thì lớp $c$ càng có dấu hiệu khó học, được đại diện chưa tốt hoặc đã bị quên nhiều.

---

### 5. Kết hợp độ hiếm và nguy cơ bị quên để chia ngân sách replay

Mỗi lớp cũ nhận một điểm ưu tiên:

$$
s_{c,t}
=
\alpha_r\bar r_c
+
\alpha_u\bar u_{c,t}
$$

Trong đó:

- $\bar r_c$ thể hiện lớp đó hiếm đến mức nào
- $\bar u_{c,t}$ thể hiện lớp đó đang khó hoặc dễ bị quên đến mức nào
- $\alpha_r$ và $\alpha_u$ điều khiển mức ảnh hưởng của hai yếu tố.

Từ điểm ưu tiên này, số mẫu ôn lại sơ bộ được tính như sau:

$$
\widetilde{R}_{c,t}
=
R_{\min}
+
\frac{\exp(s_{c,t})}
{\sum_{j\in\mathcal{C}_{\mathrm{old}}}\exp(s_{j,t})}
\left(
B_t-|\mathcal{C}_{\mathrm{old}}|R_{\min}
\right)
$$

Công thức trên thực hiện hai việc:

1. mỗi lớp luôn nhận ít nhất $R_{\min}$ mẫu
2. phần ngân sách còn lại được chia theo điểm ưu tiên $s_{c,t}$.

Điều kiện để cách chia này thực hiện được là:

$$
B_t
\geq
|\mathcal{C}_{\mathrm{old}}|R_{\min}
$$

Sau đó, số mẫu được giới hạn trong khoảng cho phép:

$$
R_{c,t}
=
\operatorname{clip}
\left(
\widetilde{R}_{c,t},R_{\min},R_{\max}
\right)
$$

Cuối cùng, cần làm tròn và điều chỉnh lại để bảo đảm:

$$
\sum_{c\in\mathcal{C}_{\mathrm{old}}}R_{c,t}
=
B_t
$$

Cách chia này hướng đến năm mục tiêu:

1. không bỏ quên hoàn toàn bất kỳ lớp cũ nào
2. ưu tiên thêm cho các lớp hiếm
3. ưu tiên thêm cho các lớp đang khó hoặc đang bị quên
4. không để một lớp chiếm toàn bộ ngân sách
5. giữ nguyên tổng chi phí ôn lại để so sánh công bằng với các phương pháp khác.

---

### 6. Mô hình được huấn luyện như thế nào ở mỗi giai đoạn?

Tại giai đoạn $t$, dữ liệu huấn luyện gồm:

- mẫu thật của các lớp mới
- mẫu biểu diễn được tạo lại từ bộ nhớ của các lớp cũ.

Với một mẫu mới:

$$
z_i^{\mathrm{new}}
=
f_{\phi}\left(x_i^{\mathrm{new}}\right)
$$

Với lớp cũ $c$, mẫu dùng để ôn lại được lấy từ phần thông tin đã lưu:

$$
\widetilde z_i^{\mathrm{old}}
\sim
q_c(z)
$$

Hàm mất mát tổng thể là:

$$
\mathcal{L}_t
=
\mathcal{L}_{\mathrm{new}}
+
\lambda_{\mathrm{replay}}\mathcal{L}_{\mathrm{old}}
+
\lambda_{\mathrm{KD}}\mathcal{L}_{\mathrm{distill}}
$$

Ba thành phần có ý nghĩa như sau:

#### Học các lớp mới

$$
\mathcal{L}_{\mathrm{new}}
=
\frac{1}{N_{\mathrm{new}}}
\sum_{i=1}^{N_{\mathrm{new}}}
\ell
\left(
g_{\theta}\left(z_i^{\mathrm{new}}\right),
y_i^{\mathrm{new}}
\right)
$$

Thành phần này giúp mô hình học cách nhận diện các loại tương tác mới.

#### Ôn lại các lớp cũ

$$
\mathcal{L}_{\mathrm{old}}
=
\frac{1}{N_{\mathrm{replay}}}
\sum_{i=1}^{N_{\mathrm{replay}}}
\ell
\left(
g_{\theta}\left(\widetilde z_i^{\mathrm{old}}\right),
y_i^{\mathrm{old}}
\right)
$$

Thành phần này nhắc mô hình rằng các lớp cũ vẫn phải được nhận diện đúng.

#### Giữ cho mô hình mới không thay đổi quá xa mô hình trước

$$
\mathcal{L}_{\mathrm{distill}}
=
D_{\mathrm{KL}}
\left(
p_{\theta_{t-1}}(y\mid z)
\,\|\,
p_{\theta_t}(y\mid z)
\right)
$$

Thành phần này khuyến khích mô hình mới giữ cách dự đoán trên các lớp cũ gần với mô hình ở giai đoạn trước.

Có thể hiểu toàn bộ quá trình như một sự cân bằng giữa bốn mục tiêu:

- học được lớp mới
- không quên lớp cũ
- không bỏ rơi các lớp hiếm
- đưa ra mức độ tin cậy phù hợp với khả năng dự đoán thật.

---

### 7. Tóm tắt ba phần của phương pháp

| Câu hỏi | Ký hiệu chính | Điều gì có thể xảy ra? | Hướng giải quyết |
| --- | --- | --- | --- |
| Bộ nhớ có đại diện đúng lớp cũ không? | $q_c(z)$ | Bộ nhớ chỉ giữ được một phần của lớp | Lưu nhiều điểm đại diện cho mỗi lớp |
| Mỗi lớp cũ được ôn lại bao nhiêu lần? | $R_{c,t}$ | Chia đều có thể không đủ cho lớp hiếm | Ưu tiên theo độ hiếm |
| Lớp nào đang dễ bị quên? | $u_{c,t}$ | Số lượng mẫu không phản ánh đầy đủ độ khó | Dùng mức phân vân, khoảng cách và lịch sử bị quên |

---

### 8. Lập luận

Lập luận chính của nghiên cứu là:

> Mô hình quên các loại tương tác thuốc cũ không chỉ vì thiếu dữ liệu cũ. Vấn đề còn có thể đến từ việc bộ nhớ mô tả lớp cũ chưa đầy đủ, ngân sách ôn lại được chia chưa phù hợp và mô hình chưa nhận biết được lớp nào đang có nguy cơ bị quên.
>

Nghiên cứu sẽ kiểm tra liệu việc kết hợp ba yếu tố sau có giúp cải thiện kết quả hay không:

$$
\text{đại diện lớp cũ tốt hơn}
+
\text{chia ngân sách hợp lý hơn}
+
\text{phát hiện lớp dễ bị quên}
$$

Mục tiêu là xây dựng một mô hình:

- quên ít hơn
- bảo vệ các loại tương tác hiếm tốt hơn
- có độ tin cậy phù hợp hơn với chất lượng dự đoán
- sử dụng cùng một ngân sách bộ nhớ hiệu quả hơn phương pháp lưu mẫu thông thường.

Ba thành phần trên không được xem là chắc chắn hiệu quả. Mỗi thành phần tạo ra một câu hỏi cần kiểm nghiệm bằng thực nghiệm:

1. Một loại tương tác thuốc có thực sự cần nhiều điểm đại diện không?
2. Ưu tiên ôn lại lớp hiếm có giúp lớp hiếm tốt hơn hay chỉ làm mô hình học thuộc các mẫu ít ỏi?
3. Mức độ không chắc chắn có dự báo được lớp nào sẽ bị quên, ngoài thông tin về số lượng mẫu hay không?
4. Ba thành phần có hỗ trợ lẫn nhau khi tổng ngân sách bộ nhớ được giữ nguyên hay không?

---

## Main Research Question

> Trong điều kiện memory budget cố định, liệu uncertainty- and imbalance-aware multi-prototype latent replay có thể đồng thời giảm catastrophic forgetting, bảo vệ các DDI class hiếm và duy trì confidence calibration tốt hơn exemplar replay thông thường trong Class-Incremental T-DDI hay không?
>

Câu hỏi này có ba đầu ra cần chứng minh:

- mô hình nhớ lớp cũ tốt hơn
- lớp hiếm được bảo vệ tốt hơn
- confidence của mô hình đáng tin cậy hơn

---

## Research Questions and Hypotheses

### Main Research Question

> Trong điều kiện memory và replay budget được giữ cố định, liệu **Uncertainty- and Imbalance-Aware Multi-Prototype Replay** có thể giảm catastrophic forgetting, cải thiện khả năng nhận diện các DDI classes hiếm và duy trì độ tin cậy của dự đoán tốt hơn exemplar replay thông thường trong Class-Incremental T-DDI hay không?
>

**Main hypothesis:** Phương pháp đề xuất sẽ giảm forgetting và cải thiện Macro-F1 trên các tail classes so với replay-distillation, đồng thời không làm giảm đáng kể overall Macro-F1 hoặc làm xấu probability calibration.

---

### RQ1. Multi-prototype có mô tả DDI classes tốt hơn single-prototype không?

Một DDI interaction class có thể chứa nhiều nhóm drug pairs khác nhau trong latent space. Vì vậy, một prototype duy nhất có thể chỉ mô tả được vùng phổ biến nhất của class và bỏ sót các nhóm nhỏ hơn.

> **RQ1:** Các DDI classes có cấu trúc đa cụm trong latent space hay không, và việc sử dụng nhiều prototypes có bảo tồn class distribution và giảm forgetting tốt hơn một prototype duy nhất hay không?
>

**H1:** Các classes có latent distribution đa cụm sẽ được mô tả tốt hơn bằng nhiều prototypes. Multi-prototype replay sẽ giảm class-wise forgetting và cải thiện old-class performance, đặc biệt đối với các classes có cấu trúc latent phức tạp.

**Hypothesis không được hỗ trợ nếu:** phần lớn classes chỉ cần một prototype, các cụm không ổn định hoặc multi-prototype không cải thiện forgetting dưới cùng memory budget.

---

### RQ2. Rarity-aware replay có bảo vệ các DDI classes hiếm tốt hơn không?

Trong dữ liệu T-DDI, số lượng mẫu giữa các classes rất chênh lệch. Uniform replay dành lượng replay giống nhau cho mọi class, mặc dù tail classes thường có representation yếu và dễ bị quên hơn.

> **RQ2:** Phân bổ nhiều replay hơn cho các classes hiếm có giúp giảm tail-class forgetting và cải thiện hiệu năng trên các classes này so với uniform class-balanced replay hay không?
>

**H2:** Rarity-aware replay sẽ tăng Macro-F1 và Recall của các tail classes, giảm số classes có F1 bằng 0 và giảm tail-class forgetting.

**Hypothesis không được hỗ trợ nếu:** rarity-aware replay không tốt hơn uniform replay, gây overfitting trên các classes có quá ít mẫu hoặc làm giảm mạnh hiệu năng của head classes và toàn bộ mô hình.

---

### RQ3. Uncertainty có dự báo class nào sẽ bị quên không?

Class frequency không phản ánh đầy đủ độ khó của một class. Hai classes có cùng số mẫu vẫn có thể khác nhau về mức độ phân tán, khả năng bị nhầm và khoảng cách tới decision boundary.

> **RQ3:** Calibrated uncertainty tại một incremental task có dự báo được future forgetting của một class, sau khi đã xem xét ảnh hưởng của class frequency và thời điểm class xuất hiện hay không?
>

**H3:** Các classes có uncertainty cao sẽ có nguy cơ giảm F1 và bị quên nhiều hơn trong các task tiếp theo. Uncertainty sẽ cung cấp thông tin bổ sung ngoài class rarity.

**Hypothesis không được hỗ trợ nếu:** uncertainty không liên hệ với future forgetting, hoặc mối liên hệ biến mất sau khi đã xét class frequency và current class performance.

---

### RQ4. Uncertainty-guided replay có thực sự giảm forgetting không?

Ngay cả khi uncertainty có thể dự báo forgetting, điều đó chưa chứng minh rằng replay nhiều hơn cho các classes uncertainty cao sẽ giúp mô hình.

> **RQ4:** Phân bổ replay budget dựa trên calibrated uncertainty có giảm future forgetting tốt hơn uniform replay hoặc rarity-only replay hay không?
>

**H4:** Uncertainty-guided replay sẽ giảm forgetting ở các classes mà mô hình đang không chắc chắn, đồng thời giảm các dự đoán sai có confidence cao.

**Hypothesis không được hỗ trợ nếu:** uncertainty-guided replay không tốt hơn rarity-only replay, ưu tiên các classes quá khó nhưng không cải thiện chúng hoặc làm calibration xấu hơn.

---

### RQ5. Ba thành phần có bổ trợ cho nhau không?

Multi-prototype, rarity-aware replay và uncertainty-aware replay giải quyết ba vấn đề khác nhau:

- Multi-prototype cải thiện cách biểu diễn old classes.
- Rarity-aware replay bảo vệ classes thiếu dữ liệu.
- Uncertainty-aware replay ưu tiên classes đang khó hoặc dễ bị quên.

> **RQ5:** Việc kết hợp ba thành phần có tạo ra hiệu quả tốt hơn so với từng thành phần riêng lẻ hay không?
>

**H5:** Phương pháp đầy đủ sẽ giảm forgetting và tăng tail-class performance tốt hơn multi-prototype replay, rarity-only replay và uncertainty-only replay.

**Hypothesis không được hỗ trợ nếu:** full method không tốt hơn thành phần mạnh nhất, hoặc một phương pháp đơn giản hơn đạt kết quả tương đương với ít độ phức tạp hơn.

---

### Overall Hypothesis

> Dưới cùng memory và replay budget, việc mô tả old DDI classes bằng nhiều latent prototypes và phân bổ replay dựa trên cả class rarity lẫn calibrated uncertainty sẽ bảo tồn kiến thức cũ tốt hơn, bảo vệ các interaction classes hiếm hiệu quả hơn và giảm các lỗi dự đoán có confidence cao so với exemplar replay-distillation.
>
