# Báo cáo kết quả và phân tích thực nghiệm hiện tại

## Phạm vi báo cáo

Báo cáo này tổng hợp trạng thái thực nghiệm hiện tại của dự án DDI2025-CIL. Ba nguồn kết quả chính được sử dụng là:

- 20 clean S02 runs: 4 phương pháp × 5 seeds × 8 tasks;
- kết quả calibration S03 trên các S02 runs;
- 5 S04 runs của phương pháp `replay_distill_fixed_budget_uniform`.

Các kết quả S02 được ưu tiên hơn bộ S01 legacy vì S02 có run provenance sạch, mỗi output directory tương ứng với đúng một execution và có prediction/latent artifacts đầy đủ. S01 vẫn có giá trị tham khảo, nhưng không được dùng làm nguồn số liệu chính trong báo cáo này.

Các phương pháp được phân tích:

- `joint_seen`: tại mỗi task được truy cập toàn bộ dữ liệu train của tất cả lớp đã thấy; đây là upper-bound thực nghiệm, không phải phương pháp continual learning khả thi;
- `sequential`: chỉ học dữ liệu của task hiện tại, không có cơ chế chống quên;
- `replay`: lưu exemplar theo lớp và dùng inverse-class-frequency sampling;
- `replay_distill`: replay kết hợp logit distillation và feature distillation;
- `replay_distill_fixed_budget_uniform`: replay-distillation với tổng memory cố định và số lượt replay cố định.

---

# 1. Kết quả thực nghiệm hiện tại

## 1.1. Kết quả audit dữ liệu

Dataset gồm ba split Parquet:

| Split | Số dòng | Số cột |
|---|---:|---:|
| Train | 520.841 | 3.787 |
| Validation | 173.614 | 3.787 |
| Test | 173.614 | 3.787 |
| **Tổng** | **868.069** | — |

Schema đã được khóa như sau:

- label: `class`;
- số interaction classes: 178;
- số descriptor features: 3.780;
- 6 meta columns bị loại khỏi model input: drug ID, drug name và SMILES của hai thuốc;
- class IDs được remap từ danh sách ID thực, không suy ra bằng `max(class) + 1`;
- scaler chỉ được fit trên train split;
- validation và test không tham gia fit preprocessing.

Audit hiện tại không phát hiện:

- label hoặc meta column lọt vào feature list;
- NaN, infinity hoặc null;
- duplicate pair trong từng split;
- ordered pair overlap giữa các split;
- reverse-pair overlap giữa các split.

Tuy nhiên, drug-level overlap giữa các split rất cao, khoảng 97%. Vì vậy benchmark đánh giá các drug pair mới được tạo phần lớn từ những thuốc đã xuất hiện ở train, không phải khả năng tổng quát hóa sang tập thuốc hoàn toàn mới.

Phân bố lớp rất mất cân bằng:

| Thống kê trên train | Giá trị |
|---|---:|
| Số lớp | 178 |
| Số mẫu nhỏ nhất của một lớp | 3 |
| Số mẫu lớn nhất của một lớp | 73.634 |
| Median | 147 |
| Số lớp có không quá 20 mẫu | 45 |
| Imbalance ratio | khoảng 24.545 lần |

Audit cũng ghi nhận 2.767/3.780 descriptor là constant trên dữ liệu đã quét. Đây là một cơ hội giảm chi phí tính toán cần được kiểm tra bằng ablation riêng; nó không thay đổi feature policy của các runs đã hoàn thành.

## 1.2. Thiết lập Class-Incremental Learning

Main benchmark dùng 8 tasks:

- task 0: 38 classes;
- task 1–7: mỗi task thêm 20 classes;
- tổng cộng: 178 classes;
- protocol chính: Random-CIL;
- seeds: 0, 1, 2, 3, 4;
- model: descriptor-only MLP-base;
- validation policy: early stopping trên toàn bộ seen classes;
- test policy: chỉ đánh giá các classes đã thấy tại task tương ứng.

Random task order không đại diện cho tiến trình thời gian thật. Kết quả chỉ nên được mô tả là class-incremental protocol được xây dựng từ DDI2025.

## 1.3. Hiệu năng cuối cùng của các baseline

Các số dưới đây là mean ± sample standard deviation trên 5 seeds. Kết quả của bốn baseline đầu lấy từ clean S02 runs. Kết quả fixed-budget lấy từ S04.

| Phương pháp | Accuracy | Macro-F1 | Weighted-F1 | Balanced Accuracy |
|---|---:|---:|---:|---:|
| `joint_seen` | 0,9287 ± 0,0023 | **0,8371 ± 0,0045** | **0,9283 ± 0,0024** | **0,8346 ± 0,0071** |
| `sequential` | 0,1308 ± 0,1047 | 0,0324 ± 0,0138 | 0,0550 ± 0,0484 | 0,0716 ± 0,0260 |
| `replay` | 0,1882 ± 0,0595 | 0,2717 ± 0,0251 | 0,1471 ± 0,0306 | 0,6411 ± 0,0234 |
| `replay_distill` | 0,2242 ± 0,0401 | 0,2632 ± 0,0257 | 0,1618 ± 0,0245 | 0,6166 ± 0,0170 |
| `replay_distill_fixed_budget_uniform` | 0,2180 ± 0,0596 | **0,3761 ± 0,0191** | 0,1481 ± 0,0425 | 0,4994 ± 0,0433 |

Trong nhóm các phương pháp continual learning thực tế:

- S04 có Macro-F1 cao nhất;
- replay có Balanced Accuracy cao nhất;
- replay-distill có Accuracy và Weighted-F1 nhỉnh hơn replay;
- không có một phương pháp thắng trên tất cả metric.

## 1.4. Forgetting và class coverage

| Phương pháp | Task-level forgetting | Final class forgetting | Số lớp F1 = 0 ở task cuối |
|---|---:|---:|---:|
| `joint_seen` | 0,0174 ± 0,0038 | 0,0566 ± 0,0033 | 5,0 ± 1,7 |
| `sequential` | 0,8246 ± 0,0639 | 0,4481 ± 0,0400 | 156,8 ± 2,9 |
| `replay` | 0,2466 ± 0,0151 | 0,2138 ± 0,0261 | **1,2 ± 0,4** |
| `replay_distill` | 0,2077 ± 0,0338 | 0,1994 ± 0,0320 | 1,8 ± 0,4 |
| `replay_distill_fixed_budget_uniform` | 0,3273 ± 0,0578 | 0,2375 ± 0,0248 | 6,2 ± 2,2 |

Sequential có khoảng 157/178 classes đạt F1 bằng 0 sau task cuối. Con số này gần bằng tổng số classes của các task cũ, cho thấy mô hình gần như chỉ còn nhận diện được nhóm classes vừa học gần đây.

Replay và replay-distill giảm số lớp F1 bằng 0 xuống còn khoảng 1–2. Tuy nhiên, Macro-F1 vẫn thấp, nghĩa là việc một class có F1 khác 0 không đồng nghĩa class đó được phân biệt tốt.

## 1.5. Precision, recall và F1 theo phương pháp

Trung bình class-level ở task cuối:

| Phương pháp | Macro Precision | Macro Recall | Macro-F1 |
|---|---:|---:|---:|
| `joint_seen` | 0,863 | 0,835 | 0,837 |
| `sequential` | 0,042 | 0,072 | 0,032 |
| `replay` | 0,290 | **0,641** | 0,272 |
| `replay_distill` | 0,275 | 0,617 | 0,263 |
| `replay_distill_fixed_budget_uniform` | **0,506** | 0,499 | **0,376** |

Replay có recall cao nhưng precision thấp. S04 giảm recall nhưng tăng precision mạnh, tạo ra Macro-F1 tốt hơn.

## 1.6. Kết quả theo độ phổ biến của lớp

Phân tích thăm dò chia classes thành bốn nhóm theo train count:

- `≤20`: cực hiếm;
- `21–50`: hiếm;
- `51–1000`: trung bình;
- `>1000`: phổ biến.

### Final Macro-F1

| Phương pháp | ≤20 | 21–50 | 51–1000 | >1000 |
|---|---:|---:|---:|---:|
| `joint_seen` | 0,697 | 0,810 | 0,875 | **0,939** |
| `sequential` | 0,025 | 0,021 | 0,033 | 0,046 |
| `replay` | 0,203 | 0,245 | **0,347** | 0,220 |
| `replay_distill` | 0,202 | 0,257 | 0,327 | 0,215 |
| `replay_distill_fixed_budget_uniform` | **0,447** | **0,469** | **0,413** | 0,178 |

Joint-seen cho thấy quan hệ tự nhiên: lớp có nhiều dữ liệu thường có F1 cao hơn. Trong replay, quan hệ này bị đảo một phần vì class-balanced replay làm tăng mạnh mức hiện diện tương đối của lớp hiếm, trong khi lớp lớn bị nén xuống số exemplar rất nhỏ.

### Final forgetting trên các lớp cũ

Chỉ tính classes có `first_task < 7`, vì class xuất hiện lần đầu ở task cuối chưa có thời gian để bị quên.

| Phương pháp | ≤20 | 21–50 | 51–1000 | >1000 |
|---|---:|---:|---:|---:|
| `joint_seen` | 0,137 | 0,084 | 0,041 | **0,014** |
| `sequential` | 0,440 | 0,536 | 0,515 | 0,547 |
| `replay` | 0,229 | 0,263 | 0,192 | **0,341** |
| `replay_distill` | 0,172 | 0,226 | 0,189 | **0,356** |
| `replay_distill_fixed_budget_uniform` | 0,174 | 0,202 | 0,236 | **0,468** |

Khi được truy cập toàn bộ dữ liệu, lớp cực hiếm kém ổn định hơn lớp phổ biến. Khi dữ liệu cũ bị nén vào replay memory, nhóm lớp phổ biến lại có forgetting cao nhất.

## 1.7. Kết quả calibration

S03 fit một scalar temperature riêng cho mỗi run/task bằng validation NLL. Cùng temperature được áp dụng lên test; test không tham gia fit.

Kết quả task cuối:

| Phương pháp | Raw ECE | Scaled ECE | Raw NLL | Scaled NLL | Final temperature |
|---|---:|---:|---:|---:|---:|
| `joint_seen` | 0,006 | 0,004 | 0,221 | 0,220 | 1,024 |
| `replay` | 0,376 | 0,109 | 8,693 | 4,395 | 4,861 |
| `replay_distill` | 0,256 | 0,125 | 6,076 | 4,212 | 3,238 |
| `sequential` | 0,628 | 0,099 | 14,267 | 4,961 | 11,188 |
| `replay_distill_fixed_budget_uniform` | 0,419 | 0,112 | 7,189 | 4,191 | 3,835 |

Joint-seen gần như đã calibrated trước temperature scaling. Các phương pháp continual learning bị overconfidence mạnh. Temperature scaling giảm ECE và NLL nhưng không thay đổi prediction hoặc accuracy.

Sau scaling, mean confidence ở task cuối giảm còn khoảng:

- replay: 0,080;
- replay-distill: 0,099;
- S04: 0,106;
- sequential: 0,033.

Ở S04 task cuối không còn sample đạt confidence từ 0,9 trở lên. Vì vậy high-confidence error rate không xác định và được để trống, không được diễn giải là error rate bằng 0.

## 1.8. Kết quả audit fixed-budget S04

S04 đạt các invariant đã đặt ra:

- 5/5 runs hoàn thành đủ 8 tasks;
- tổng memory sau mỗi task là 6.800 unique train exemplars;
- task 1–7 có đúng 6.800 old-sample replay draws mỗi epoch;
- memory chỉ lấy từ current-task train data;
- validation/test chỉ dùng cho early stopping, evaluation và export;
- không có NaN/Inf trong metric hoặc budget audit.

Tuy nhiên, số mẫu mới ở từng task thay đổi từ khoảng 7.900 đến 156.900. Vì replay draws được cố định ở 6.800, tỷ lệ old/new draws dao động khoảng 0,043–0,859. S04 cố định replay count tuyệt đối, nhưng không cố định tỷ lệ replay so với dữ liệu task mới.

---

# 2. Phân tích kết quả thực nghiệm

## 2.1. Catastrophic forgetting là vấn đề thật và rất lớn

Sequential có Macro-F1 chỉ 0,032 và task forgetting khoảng 0,825. Sau task cuối, trung bình 156,8/178 classes có F1 bằng 0.

Kết quả này chứng minh rằng việc mở rộng classifier head và tiếp tục fine-tune trên task mới không đủ để giữ kiến thức cũ. Khi loss chỉ được tính từ dữ liệu mới, gradient không chứa tín hiệu bảo vệ decision boundary của các classes trước đó.

Khoảng cách giữa sequential và joint-seen không phải do model thiếu capacity đơn thuần:

- cùng họ MLP đạt Macro-F1 khoảng 0,84 khi được truy cập dữ liệu cũ;
- hiệu năng giảm xuống khoảng 0,03 khi dữ liệu cũ bị loại khỏi training.

Do đó continual-learning constraint là nguyên nhân chính của suy giảm, dù class imbalance và intrinsic class difficulty vẫn đóng vai trò phụ.

## 2.2. Replay giải quyết class coverage nhưng chưa giải quyết class discrimination

Replay giảm số classes có F1 bằng 0 từ khoảng 157 xuống còn 1–2. Đây là cải thiện rất lớn về class coverage: gần như mọi class vẫn có một số prediction đúng.

Tuy nhiên, replay có:

- Macro Recall khoảng 0,64;
- Macro Precision chỉ khoảng 0,29;
- Macro-F1 chỉ khoảng 0,27.

Điều này cho thấy replay thường mở rộng vùng dự đoán của nhiều classes quá mức. Mô hình tìm đúng nhiều mẫu thật của một class, nhưng đồng thời gán nhiều mẫu thuộc class khác vào class đó.

Nói cách khác:

> Replay giúp mô hình nhớ rằng class vẫn tồn tại, nhưng chưa giúp mô hình phân biệt class đó chính xác với các class khác.

Vì thế số classes có F1 bằng 0 không nên được dùng riêng như bằng chứng rằng replay đã giải quyết forgetting.

## 2.3. Class-balanced replay làm thay đổi class prior

Dữ liệu gốc có long-tail rất mạnh. Một số lớp chỉ có vài mẫu, trong khi lớp lớn nhất có hơn 73.000 mẫu.

Trong legacy replay:

- lớp hiếm có thể được giữ toàn bộ mẫu;
- lớp phổ biến chỉ giữ tối đa khoảng 50 exemplar;
- inverse-class-frequency sampler làm các classes xuất hiện gần cân bằng trong training draws.

Điều này làm phân bố lớp mà model nhìn thấy trong replay rất khác phân bố test thật:

```text
P_replay(class) != P_test(class)
```

Hệ quả quan sát được:

- lớp hiếm có recall rất cao nhưng precision thấp;
- lớp phổ biến có recall giảm mạnh;
- Balanced Accuracy cao nhưng Accuracy, Weighted-F1 và Macro-F1 thấp;
- model có xu hướng dự đoán quá nhiều vào classes mới hoặc classes được oversample.

Đây là prior mismatch chứ không chỉ là catastrophic forgetting theo nghĩa hẹp.

## 2.4. Rare class khó học, nhưng không nhất thiết thiếu replay

Joint-seen cho thấy lớp hiếm vốn khó:

- nhóm `≤20` mẫu chỉ đạt F1 khoảng 0,70;
- nhóm `>1000` mẫu đạt F1 khoảng 0,94.

Nhưng dưới replay, nhóm cực hiếm có recall khoảng 0,84–0,86, trong khi precision chỉ khoảng 0,13. Lớp hiếm không bị bỏ qua; ngược lại, chúng thường bị dự đoán quá nhiều.

Final forgetting cũng cho thấy:

- replay: head forgetting 0,341, cao hơn rare forgetting 0,229;
- replay-distill: head 0,356, rare 0,172;
- S04: head 0,468, rare 0,174.

Do đó quan hệ đơn giản sau không được kết quả hiện tại hỗ trợ:

```text
ít mẫu hơn -> bị quên nhiều hơn -> cần replay nhiều hơn
```

Cần phân biệt ít nhất ba khái niệm:

1. **Intrinsic difficulty:** class khó ngay cả khi có toàn bộ dữ liệu;
2. **Memory compression:** bao nhiêu phần dữ liệu của class bị mất khi đưa vào memory;
3. **Continual forgetting:** class đã từng học tốt nhưng suy giảm sau các task sau.

Một class hiếm có thể khó học nhưng được giữ gần như toàn bộ trong memory. Một class phổ biến có thể dễ học trong joint-seen nhưng mất phần lớn distribution khi hàng chục nghìn mẫu bị nén xuống vài chục exemplar.

## 2.5. Head class là nhóm chịu memory compression mạnh nhất

Nhóm `>1000` mẫu đạt F1 khoảng 0,939 trong joint-seen nhưng chỉ còn:

- 0,220 trong replay;
- 0,215 trong replay-distill;
- 0,178 trong S04.

Đây là mức suy giảm lớn hơn nhiều so với nhóm rare.

Một head class có thể chứa nhiều kiểu drug pair hoặc nhiều vùng latent khác nhau. Herding theo khoảng cách tới class mean ưu tiên các exemplar điển hình, nhưng có thể bỏ sót các mode nhỏ hoặc vùng biên của class distribution.

Kết quả này tạo động lực hợp lý cho multi-prototype, nhưng chưa chứng minh multi-prototype hiệu quả. E02 vẫn phải kiểm tra:

- class có thực sự đa cụm hay không;
- cluster có ổn định giữa seeds/task hay không;
- nhiều prototype có giảm forgetting dưới cùng budget hay không;
- lợi ích có tập trung ở classes có compression cao hay không.

## 2.6. Distillation tạo stability–plasticity trade-off

So với replay, replay-distill:

- giảm task forgetting từ 0,247 xuống 0,208;
- giảm class forgetting từ 0,214 xuống 0,199;
- tăng Accuracy và Weighted-F1;
- giảm nhẹ Macro-F1 và Balanced Accuracy.

Distillation giúp model mới không thay đổi quá xa teacher, nên tăng stability đối với classes cũ. Nhưng teacher cũng chứa bias và lỗi từ task trước. Khi ràng buộc quá mạnh, model có thể khó thích nghi với classes mới hoặc giữ lại decision boundary chưa tốt.

Kết quả hiện tại không cho thấy replay-distill thắng replay trên mọi metric. Hai phương pháp đại diện cho hai điểm khác nhau trên trade-off giữa:

- stability: giữ kiến thức cũ;
- plasticity: học và phân biệt classes mới.

## 2.7. S04 tăng Macro-F1 bằng cách cân bằng lại precision và recall

S04 tăng Macro Precision từ khoảng 0,28–0,29 lên 0,51, nhưng giảm Macro Recall từ khoảng 0,62–0,64 xuống 0,50.

Điều này giải thích đồng thời:

- Macro-F1 tăng lên 0,376;
- Balanced Accuracy giảm;
- forgetting tăng;
- số classes F1 bằng 0 tăng nhẹ.

S04 không phải là phiên bản tốt hơn tuyệt đối của legacy replay. Nó giảm over-prediction và cải thiện discrimination, nhưng đổi lại giữ old-class recall kém hơn.

Ngoài ra, so sánh S04 với legacy replay không cô lập một causal factor duy nhất. Hai protocol khác nhau đồng thời ở:

- cách cấp tổng memory theo task;
- số replay draws;
- cách sampler xử lý current/replay data;
- replay-to-current ratio;
- số optimizer steps mỗi epoch.

Vì vậy chỉ được kết luận S04 tạo ra outcome khác, không được kết luận riêng fixed storage hoặc fixed exposure là nguyên nhân của thay đổi.

## 2.8. Fixed count không có nghĩa fixed training pressure

S04 luôn replay 6.800 old samples mỗi epoch, nhưng current dataset size phụ thuộc các classes xuất hiện trong task. Do long-tail, task có thể chứa dưới 10.000 hoặc hơn 150.000 mẫu mới.

Vì vậy:

- cùng 6.800 replay draws có thể rất mạnh ở task nhỏ;
- cùng 6.800 replay draws có thể rất yếu ở task lớn;
- task order và seed ảnh hưởng trực tiếp tới stability–plasticity balance.

Các phương pháp tương lai vẫn có thể được so sánh công bằng nếu dùng cùng seed, task order và budget. Tuy nhiên, phân tích nên lưu và kiểm soát thêm:

- current dataset size;
- old/new draw ratio;
- optimizer steps;
- class composition của task;
- tổng train count của old và new classes.

## 2.9. Calibration drift là hệ quả nghiêm trọng của continual learning

Task 0 của các phương pháp có calibration tương đối tốt. Khi số task tăng:

- replay và sequential ngày càng overconfident;
- ECE và NLL tăng mạnh;
- confidence cao không còn phản ánh xác suất dự đoán đúng.

Temperature scaling giảm calibration error nhưng không thay đổi argmax prediction. Vì vậy nó không sửa:

- class bias;
- false positives;
- catastrophic forgetting;
- head-class compression;
- new-class bias.

Temperature scaling chỉ làm probability dễ diễn giải hơn. Đây là điều kiện cần trước khi dùng uncertainty, nhưng không phải giải pháp cho forgetting.

RQ3 hiện chưa được trả lời. Cần kiểm tra calibrated validation uncertainty tại task hiện tại có dự báo future F1 drop hay không sau khi kiểm soát:

- log train count;
- current F1, precision và recall;
- first task/class age;
- memory count và compression ratio;
- current task size;
- seed và task order.

Test uncertainty chỉ được dùng để reporting, không được dùng làm replay decision.

## 2.10. Ultra-rare metrics có độ nhiễu cao

Một số classes chỉ có 1–3 test samples. Với các class này:

- một prediction đúng có thể làm F1 tăng rất mạnh;
- một prediction sai có thể làm F1 giảm về 0;
- forgetting có thể thay đổi lớn dù representation chỉ thay đổi nhỏ.

Năm training seeds không giải quyết vấn đề này vì cả năm runs dùng cùng một test split. Mean ± standard deviation trên seeds chủ yếu phản ánh training randomness, không phản ánh uncertainty do test sampling.

Rare-class conclusions nên:

- ưu tiên group-level metrics;
- luôn báo test support;
- tách các ngưỡng `≤5`, `≤10`, `≤20`;
- dùng bootstrap hoặc confidence interval khi phù hợp;
- không overclaim từ một class riêng lẻ.

## 2.11. Ý nghĩa đối với main research question

Main research question yêu cầu phương pháp đề xuất đồng thời:

1. giảm catastrophic forgetting;
2. bảo vệ rare classes;
3. duy trì calibration;
4. sử dụng cùng memory và replay budget;
5. tốt hơn exemplar replay thông thường.

Kết quả hiện tại mới hoàn thành nền tảng để kiểm tra câu hỏi này:

- catastrophic forgetting đã được chứng minh;
- replay và fixed-budget baseline đã có;
- class trajectories, latent features và calibrated probabilities đã có;
- chưa có multi-prototype method;
- chưa có rarity-aware allocation experiment;
- chưa có uncertainty-forgetting prediction experiment;
- chưa có uncertainty-guided replay;
- chưa có full method hoặc component ablation.

Do đó chưa thể chấp nhận main hypothesis hoặc các H1–H5.

---

# 3. Kết luận từ kết quả thực nghiệm

## 3.1. Các kết luận đã được dữ liệu hỗ trợ

### Kết luận 1: Catastrophic forgetting là vấn đề cốt lõi

Sequential gần như chỉ giữ được các classes mới nhất. Đây là bằng chứng rõ rằng DDI2025-CIL cần cơ chế bảo tồn kiến thức cũ.

### Kết luận 2: Replay là cần thiết nhưng chưa đủ

Replay giúp gần như mọi class còn có prediction đúng, nhưng precision thấp và false positive nhiều. Replay hiện tại giải quyết class coverage tốt hơn class discrimination.

### Kết luận 3: Class-balanced replay tạo prior mismatch

Việc cho các classes xuất hiện gần cân bằng trong replay training không khớp phân bố long-tail của test. Điều này làm recall của lớp hiếm tăng nhưng precision giảm, đồng thời làm recall của head classes suy giảm.

### Kết luận 4: Rarity không đồng nghĩa với nhu cầu replay

Rare classes vốn khó trong joint-seen, nhưng trong replay chúng thường được giữ tỷ lệ dữ liệu cao và bị over-predict. Head classes mới là nhóm có forgetting cao nhất do memory compression.

### Kết luận 5: Distillation giảm forgetting nhưng không cải thiện mọi metric

Distillation tăng stability nhưng tạo trade-off với plasticity. Nó không phải cải tiến tuyệt đối so với replay.

### Kết luận 6: S04 là baseline công bằng cần thiết nhưng chưa phải lời giải cuối

S04 kiểm soát memory và replay count, đồng thời cải thiện Macro-F1 thông qua precision tốt hơn. Tuy nhiên forgetting tăng và head-class performance giảm. S04 nên được dùng làm M0 để so sánh với các phương pháp tiếp theo, không nên được xem là phương pháp thắng toàn diện.

### Kết luận 7: Raw confidence không đủ tin cậy

Các phương pháp continual learning bị overconfidence mạnh sau nhiều tasks. Mọi uncertainty signal phục vụ replay phải được calibration bằng validation data trước.

### Kết luận 8: Benchmark không đo unseen-drug hoặc temporal generalization

Benchmark hiện tại là constructed class-incremental setting trên mostly-seen drugs. Không được diễn giải kết quả như khả năng dự đoán thuốc mới hoặc continual learning theo thời gian thật.

## 3.2. Trạng thái các giả thuyết nghiên cứu

| Research question | Kết luận hiện tại |
|---|---|
| RQ1: Multi-prototype có cần thiết không? | Chưa trả lời. Kết quả head-class compression tạo động lực để chạy E02. |
| RQ2: Rarity-aware replay có tốt hơn không? | Chưa được hỗ trợ. Kết quả hiện tại cho thấy rare class không đơn giản là nhóm thiếu replay. |
| RQ3: Uncertainty có dự báo forgetting không? | Chưa trả lời. Calibration đã hoàn thành nhưng predictive analysis chưa chạy. |
| RQ4: Uncertainty-guided replay có giảm forgetting không? | Chưa triển khai. |
| RQ5: Các thành phần có bổ trợ nhau không? | Chưa triển khai full method và component ablation. |

## 3.3. Điều chỉnh hướng nghiên cứu được đề xuất từ kết quả

Kết quả hiện tại cho thấy phương pháp cuối không nên chỉ phân bổ replay theo class rarity. Tín hiệu allocation nên xem xét đồng thời:

- memory compression ratio;
- latent complexity và số mode ổn định;
- calibrated uncertainty;
- current precision, recall và F1;
- observed performance drift;
- class age;
- test/validation support khi diễn giải metric.

Rarity nên được dùng như một safeguard để tránh bỏ đói lớp ít dữ liệu, không nên mặc định là trọng số ưu tiên chính.

Multi-prototype cũng nên được cấp thích nghi:

- không cluster classes có quá ít mẫu;
- ưu tiên classes có latent distribution đa cụm;
- ưu tiên classes có compression cao;
- luôn giữ tổng prototype/memory budget cố định.

## 3.4. Thứ tự thực nghiệm tiếp theo

### Bước 1: Hoàn thành E01 — rarity và forgetting

Cần tạo O08 chính thức, tách rõ:

- intrinsic difficulty;
- initial learnability;
- future forgetting;
- CIL gap so với joint-seen;
- ảnh hưởng của test support và class age.

### Bước 2: Chạy E02 — prototype diagnostics

Dùng S02 latent features để kiểm tra `K = 1, 2, 3, 4`, cluster stability và khoảng cách tới prototype. Phân tích nên tập trung thêm vào classes có train count lớn và compression cao.

### Bước 3: Chạy E03 — uncertainty dự báo forgetting

Dùng calibrated validation probabilities, so sánh tối thiểu bốn mô hình:

1. frequency;
2. current F1;
3. frequency + current F1;
4. frequency + current F1 + calibrated uncertainty.

Chỉ giữ uncertainty trong phương pháp cuối nếu mô hình 4 cung cấp thông tin bổ sung ổn định.

### Bước 4: Triển khai component methods theo từng mức

Thứ tự nên là:

```text
M0: fixed-budget exemplar replay-distillation
M1: single-prototype latent replay
M2: adaptive multi-prototype replay
M3: prototype + rarity/compression-aware allocation
M4: prototype + calibrated uncertainty-aware allocation
M5: full method
```

Mỗi mức chỉ thêm một thay đổi và phải dùng cùng:

- task files;
- seeds;
- model backbone;
- total memory budget;
- replay draw budget;
- validation policy;
- evaluation artifacts.

## 3.5. Kết luận cuối cùng

Kết quả hiện tại xác nhận DDI2025-CIL là một bài toán continual learning khó và có ý nghĩa. Mô hình không chỉ bị quên lớp cũ; replay còn làm thay đổi class prior, gây false positive, làm lớp mới hoặc lớp được oversample chiếm vùng quyết định, và nén quá mạnh distribution của head classes.

Insight trung tâm là:

> Lớp cần được bảo vệ không nhất thiết là lớp hiếm nhất. Đó có thể là lớp bị mất nhiều thông tin nhất khi nén vào memory, có latent distribution phức tạp, hoặc đang thể hiện dấu hiệu suy giảm đã được calibration xác nhận.

Vì vậy hướng nghiên cứu phù hợp hơn là **risk- and representation-aware replay dưới fixed budget**, trong đó rarity chỉ là một thành phần. Multi-prototype và uncertainty-aware allocation vẫn là các giả thuyết hợp lý, nhưng phải được xác nhận độc lập qua E01–E03 trước khi kết hợp thành full method.

## Tài liệu và artifacts liên quan

- `outputs/audit/audit_summary.md`
- `outputs/class_distribution/class_distribution_summary.md`
- `outputs/leakage/leakage_summary.md`
- `outputs/runs_s02/*`
- `outputs/s03/calibration_by_task.csv`
- `outputs/runs_s04/*`
- `outputs/s04/fixed_budget_baseline.csv`
- `outputs/s04/calibration/calibration_by_task.csv`
- `docs/results/s01_results.md`
- `docs/results/s03_results.md`
- `docs/results/s04_results.md`
- `docs/diagnostics.md`
- `docs/proposal_v1.md`
