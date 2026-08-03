# Milestone: Diagnostic Evaluation

## 1. Mục tiêu

Milestone hiện tại xây dựng nền tảng đánh giá để trả lời ba câu hỏi trước khi triển khai full method:

1. **Q1 — Rare-class forgetting:** Các DDI class hiếm có thực sự bị quên nhiều hơn không?
2. **Q2 — Multi-prototype:** Một DDI class có thực sự cần nhiều prototype trong latent space không?
3. **Q3 — Uncertainty:** Uncertainty hiện tại có dự báo được class nào sẽ bị quên trong các task tiếp theo không?

Kết quả của milestone sẽ quyết định có nên đưa ba thành phần sau vào phương pháp cuối cùng:

- multi-prototype replay;
- rarity-aware replay allocation;
- uncertainty-aware replay allocation.

## 2. Bối cảnh và khoảng trống hiện tại

Repo đã có các baseline quan trọng:

- Sequential fine-tuning;
- Exemplar replay;
- Replay-distillation;
- EWC;
- Joint-seen oracle.

Replay và replay-distillation hiện là các baseline mạnh. Tuy nhiên, pipeline chưa lưu đủ thông tin để kiểm chứng các giả thuyết trong research proposal, gồm:

- F1 của từng class sau từng task;
- class-wise forgetting;
- logits, probabilities và prediction confidence;
- entropy hoặc uncertainty của từng class;
- latent representation của từng mẫu;
- calibration metrics theo từng task;
- kết quả so sánh dưới cùng tổng memory budget và replay budget.

Vì vậy, ưu tiên hiện tại là hoàn thiện evaluation và fixed-budget baseline. Chưa triển khai full method cho đến khi có đủ bằng chứng từ các thí nghiệm chẩn đoán.

## 3. Nguyên tắc thực nghiệm

1. Mọi phương pháp phải được so sánh dưới cùng **total memory budget**.
2. Tổng số lượt replay phải được giữ cố định theo task hoặc epoch.
3. Temperature scaling chỉ được fit trên validation set.
4. Test set không được dùng để tính uncertainty phục vụ replay allocation.
5. Mỗi mức phát triển chỉ thêm một thay đổi để xác định chính xác đóng góp của thành phần đó.
6. Kết quả phải truy vết được theo `seed`, `method`, `train_task` và `class_id`.

## 4. Kế hoạch triển khai

### 4.1. S01 — Hoàn thiện class-wise evaluation

Sau mỗi task, đánh giá toàn bộ class đã học và lưu các trường sau:

- precision;
- recall;
- F1;
- số mẫu test;
- mean confidence;
- mean entropy;
- task mà class bắt đầu xuất hiện.

Calibration error và calibrated confidence được xử lý riêng trong S03.

Các bước nhỏ:

- [x] **S01.1** Mở rộng evaluator để tính metric theo từng class.
- [x] **S01.2** Gắn metadata về seed, method, train task, class ID và train count.
- [x] **S01.3** Ghi trajectory của mỗi class sau từng task vào `[O01]`.
- [x] **S01.4** Tính class-wise forgetting theo công thức:

  $$
  \text{forgetting}_{c,t}
  =
  \max_{\tau \le t} \operatorname{F1}_{c,\tau}
  -
  \operatorname{F1}_{c,t}
  $$

- [x] **S01.5** Tổng hợp forgetting theo class và task vào `[O02]`.

Full run đã hoàn thành cho 4 methods × 5 seeds. Xem [S01 full experiment results](results/s01_results.md).

Schema tối thiểu của `[O01]`:

```text
seed, method, train_task, class_id, first_task, train_count,
test_count, precision, recall, f1, confidence, entropy
```

### 4.2. S02 — Lưu prediction và latent representation

Hàm đánh giá cần trả thêm:

- logits;
- softmax probabilities;
- predictions;
- labels;
- latent features.

Luồng xử lý của MLP cần được tách rõ:

```text
Input descriptors
    -> Backbone / Encoder
    -> Latent representation
    -> Classifier
    -> DDI class
```

Các bước nhỏ:

- [ ] **S02.1** Tách encoder và classifier trong model interface.
- [ ] **S02.2** Cho evaluator trả về logits, probabilities, predictions và labels.
- [ ] **S02.3** Cho evaluator trả về latent feature của từng mẫu.
- [ ] **S02.4** Lưu prediction-level data vào `[O03]`.
- [ ] **S02.5** Lưu latent features và metadata vào `[O04]`.
- [ ] **S02.6** Kiểm tra sample ID và label khớp giữa `[O03]` và `[O04]`.

Các artifact này phục vụ cả uncertainty analysis và prototype replay.

### 4.3. S03 — Tích hợp calibration

Các metric tối thiểu:

- Expected Calibration Error (ECE);
- Brier score;
- Negative Log-Likelihood (NLL);
- mean confidence;
- high-confidence error rate.

Các bước nhỏ:

- [ ] **S03.1** Tính calibration metrics trước temperature scaling.
- [ ] **S03.2** Fit temperature scaling trên validation set.
- [ ] **S03.3** Tính lại calibration metrics sau scaling.
- [ ] **S03.4** Tổng hợp metric theo seed, method và task vào `[O05]`.
- [ ] **S03.5** Xác nhận replay allocation không sử dụng dữ liệu từ test set.

Uncertainty chỉ được dùng làm tín hiệu khi probabilities đã được kiểm tra về calibration.

### 4.4. S04 — Xây fixed-budget replay baseline

Baseline hiện tại lưu tối đa một số exemplar cố định cho mỗi class, khiến tổng memory tăng khi số class tăng. Baseline mới cần có tên:

```text
replay_distill_fixed_budget_uniform
```

Baseline vẫn dùng exemplar replay và distillation, nhưng phải bảo đảm:

- tổng memory không tăng theo số class;
- tổng số lượt replay được giữ cố định;
- memory được chia đều giữa các class đã học.

Các bước nhỏ:

- [ ] **S04.1** Định nghĩa total memory budget cố định.
- [ ] **S04.2** Định nghĩa replay budget cố định theo task hoặc epoch.
- [ ] **S04.3** Cài đặt uniform allocation giữa các class đã học.
- [ ] **S04.4** Ghi nhận budget thực tế sau mỗi task.
- [ ] **S04.5** Chạy baseline với cùng seed và protocol của các phương pháp đối chứng.
- [ ] **S04.6** Lưu budget audit và kết quả baseline vào `[O06]` và `[O07]`.

Đây là baseline chính để so sánh với prototype replay.

## 5. Thí nghiệm chẩn đoán

### 5.1. E01 — Rare classes có bị quên nhiều hơn không?

Nhóm so sánh:

- head classes;
- medium classes;
- tail classes;
- classes có không quá 20 mẫu;
- classes có không quá 5 mẫu.

Các bước nhỏ:

- [ ] **E01.1** Chốt quy tắc phân nhóm head, medium và tail.
- [ ] **E01.2** Tính Macro-F1 và Recall theo nhóm.
- [ ] **E01.3** Tính mean forgetting theo nhóm.
- [ ] **E01.4** Đếm số class có F1 bằng 0.
- [ ] **E01.5** Đếm và phân tích high-confidence errors.
- [ ] **E01.6** Kiểm soát task order và current F1 khi phân tích quan hệ giữa rarity và forgetting.
- [ ] **E01.7** Lưu bảng phân tích vào `[O08]`.

**Tiêu chí quyết định:** Nếu rare classes không bị quên nhiều hơn sau khi kiểm soát task order và current F1, rarity-aware replay có thể không cần thiết.

### 5.2. E02 — Các class có cấu trúc đa cụm không?

Với latent representation của từng class, thử:

$$
K \in \{1, 2, 3, 4\}
$$

Các metric đánh giá:

- Silhouette score;
- GMM BIC;
- khoảng cách tới prototype gần nhất;
- cluster stability.

Các bước nhỏ:

- [ ] **E02.1** Xác định số mẫu tối thiểu để một class được thử nhiều prototype.
- [ ] **E02.2** Fit mô hình với `K = 1, 2, 3, 4` trên từng class đủ điều kiện.
- [ ] **E02.3** Tính các metric về chất lượng và độ ổn định của cluster.
- [ ] **E02.4** So sánh single-prototype với multi-prototype dưới cùng memory budget.
- [ ] **E02.5** Lưu kết quả vào `[O09]`.

Không dùng nhiều prototype cho class có quá ít mẫu.

**Tiêu chí quyết định:** Nếu phần lớn class chỉ cần một prototype, ưu tiên số prototype thích nghi theo class thay vì mặc định nhiều prototype cho mọi class.

### 5.3. E03 — Uncertainty có dự báo future forgetting không?

Đo uncertainty của từng class tại task hiện tại, sau đó kiểm tra mức giảm F1 của class đó trong các task tiếp theo.

Các mô hình dự báo cần so sánh:

1. class frequency;
2. current F1;
3. class frequency + current F1;
4. class frequency + current F1 + uncertainty.

Các bước nhỏ:

- [ ] **E03.1** Xây biến mục tiêu future forgetting từ class trajectory.
- [ ] **E03.2** Chuẩn hoá class frequency, current F1 và uncertainty.
- [ ] **E03.3** Fit và đánh giá bốn mô hình dự báo trên cùng split/protocol.
- [ ] **E03.4** Kiểm tra phần thông tin bổ sung của uncertainty.
- [ ] **E03.5** Lưu kết quả vào `[O10]`.

**Tiêu chí quyết định:** Uncertainty chỉ có giá trị nếu cung cấp thêm thông tin ngoài class frequency và current performance.

## 6. Danh mục output files

| ID | File | Nội dung chính | Nguồn |
| --- | --- | --- | --- |
| **O01** | `class_trajectory.csv` | Metric của từng class sau từng task | S01 |
| **O02** | `class_forgetting.csv` | Best F1, current F1 và forgetting theo class/task | S01 |
| **O03** | `predictions.parquet` | Logits, probabilities, predictions, labels và sample metadata | S02 |
| **O04** | `latent_features.npz` | Latent features và khóa liên kết tới từng mẫu | S02 |
| **O05** | `calibration_by_task.csv` | ECE, Brier, NLL, confidence và high-confidence error rate | S03 |
| **O06** | `replay_budget_audit.csv` | Memory và replay budget thực tế theo task/class | S04 |
| **O07** | `fixed_budget_baseline.csv` | Kết quả của `replay_distill_fixed_budget_uniform` | S04 |
| **O08** | `rarity_forgetting.csv` | Phân tích forgetting theo nhóm frequency | E01 |
| **O09** | `prototype_diagnostics.csv` | Kết quả thử `K = 1, 2, 3, 4` prototypes | E02 |
| **O10** | `uncertainty_forgetting.csv` | Khả năng dự báo future forgetting | E03 |

Tên thư mục output cụ thể cần tuân theo cấu trúc experiment hiện có của repo. Mỗi file phải chứa hoặc liên kết được với `seed`, `method` và `train_task` để tái lập kết quả.

## 7. Thứ tự phát triển phương pháp

Chỉ bắt đầu chuỗi này sau khi hoàn thành các bước S01–S04 và thí nghiệm E01–E03.

| Mức | Phương pháp | Mục đích |
| --- | --- | --- |
| **M0** | Fixed-budget exemplar replay-distillation | Tạo baseline công bằng |
| **M1** | Single-prototype latent replay | Kiểm tra latent replay |
| **M2** | Multi-prototype latent replay | Kiểm tra lợi ích của nhiều prototype |
| **M3** | Multi-prototype + rarity-aware replay | Kiểm tra rarity allocation |
| **M4** | Multi-prototype + uncertainty-aware replay | Kiểm tra uncertainty allocation |
| **M5** | Full method | Kiểm tra khả năng bổ trợ giữa các thành phần |

## 8. Tiêu chí hoàn thành milestone

Milestone hoàn thành khi đáp ứng toàn bộ các điều kiện sau:

- [x] **C01** Có F1 trajectory của từng class qua tất cả các task.
- [x] **C02** Tính được class-wise forgetting.
- [ ] **C03** Có probabilities, confidence, entropy và latent representation.
- [ ] **C04** Có calibration metrics theo task.
- [ ] **C05** Có fixed-memory và fixed-replay baseline.
- [ ] **C06** Có kết luận sơ bộ về việc rare classes có bị quên nhiều hơn không.
- [ ] **C07** Có kết luận sơ bộ về việc multi-prototype có cần thiết không.
- [ ] **C08** Có kết luận sơ bộ về việc uncertainty có dự báo future forgetting không.

## 9. Kết quả quyết định hướng phát triển

Milestone này phải cung cấp đủ bằng chứng để quyết định:

1. có nên dùng multi-prototype;
2. có nên ưu tiên rare classes;
3. có nên dùng uncertainty để phân bổ replay budget.

Phương pháp cuối cùng chỉ nên giữ các thành phần được thực nghiệm hỗ trợ dưới cùng memory và replay budget.
