# Kế hoạch T-DDI Ensemble ×3 với EWC cho DDI-CIL

## 1. Mục tiêu và phạm vi

Mục tiêu là bổ sung một backbone/study mới cho bài toán class-incremental learning (CIL):

```text
method          = ewc
backbone        = tddi_ensemble3
protocol        = P4
experiment_seed = 0
members         = 3
```

Mỗi member là một T-DDI numerical-only độc lập:

```text
QSAR 3.780
  -> LayerNorm(3.780)
  -> Linear(3.780, 7.560)
  -> activation/dropout
  -> Linear(7.560, 7.560)
  -> activation/dropout
  -> expandable head C_t
```

`C_t` là tổng số lớp đã thấy đến task `t`; không dùng head 178 lớp ngay từ task đầu. Ba member được train lần lượt qua toàn bộ chuỗi task, mỗi member dùng toàn bộ dữ liệu task và EWC state riêng. Ensemble chỉ diễn ra khi evaluation bằng cách lấy trung bình posterior.

Study này là study backbone mới. Không được sửa, ghi đè hoặc trộn kết quả của bảng protocol P0-P8 đang khóa trong `README.md`.

Không nằm trong phạm vi đầu tiên:

- Replay và replay-distillation.
- GEM/A-GEM.
- Joint ensemble loss hoặc train ba member đồng thời.
- Thay đổi task split, class order hoặc protocol P4.
- Chọn threshold dựa trên test set.

## 2. Các quyết định thiết kế đã chốt

1. Train ba member lần lượt để giảm peak VRAM.
2. Mỗi member nhận toàn bộ dữ liệu của mỗi task; không dùng 3-fold data exposure trong bản CIL đầu tiên.
3. Ba member dùng chung experiment seed, task order, data split và class order.
4. Randomness của member được tách khỏi experiment seed.
5. Loss giữ theo T-DDI gốc: mỗi member được train độc lập bằng Focal Loss; không có ensemble loss.
6. Head của cả ba trajectory mở rộng theo cùng một `seen_class_map`.
7. EWC state gồm Fisher và `theta_star` riêng cho từng member.
8. Ensemble inference dùng trung bình xác suất, không trung bình raw logits.
9. UE dùng entropy, variance và mutual information; entropy/MI được chuẩn hóa theo `log(C_t)` để so sánh qua các task.
10. Threshold confidence được đặt thành một grid định trước, chọn trên validation rồi đóng băng trước khi test.
11. Checkpoint tại task boundary chỉ cần lưu model state và Fisher; `theta_star` có thể khôi phục bằng cách clone model state khi resume.
12. Khi sau này thêm replay, ba member phải dùng chung exemplar identities và một memory budget toàn hệ thống.

## 3. Khó khăn và giải pháp

### 3.1. Peak VRAM

Một member đầy đủ có khoảng 87,45 triệu tham số; ba member có khoảng 262 triệu tham số. Con số GPU idle như `911 MiB / 24.570 MiB` không phản ánh peak VRAM khi có activation, gradient, AdamW moments, Fisher và `theta_star`.

Giải pháp:

- Train từng member lần lượt, không giữ ba optimizer trên GPU.
- Smoke test task 0 với batch size 64 hoặc 128.
- Ghi lại `torch.cuda.max_memory_allocated()`.
- Dùng gradient accumulation để tăng effective batch size.
- Chỉ bật AMP sau khi bản FP32 chạy đúng và có kiểm tra numerical stability.
- Chuyển member đã hoàn thành sang CPU/disk.

### 3.2. Thời gian huấn luyện

Ba member làm thời gian train tăng ít nhất ba lần. EWC còn cần thêm một vòng forward/backward sau mỗi task để tính Fisher.

Giải pháp:

- Chạy smoke test 2-3 epoch trên task 0 trước.
- Tiếp theo chạy hai task để kiểm tra head expansion và EWC penalty.
- Chỉ chạy đủ tám task khi các invariant đã đúng.
- Cache task indices, feature metadata và class map.
- Cho phép giới hạn số sample/batch dùng tính Fisher bằng một option rõ ràng; mặc định vẫn dùng toàn bộ loader để giữ baseline.

### 3.3. Expandable head

Nếu dùng head 178 lớp từ đầu, semantics CIL bị thay đổi. Khi head tăng kích thước, trọng số lớp cũ, Fisher và `theta_star` phải được ánh xạ đúng theo raw class ID.

Giải pháp:

- Tiếp tục dùng `current_seen_map` làm nguồn sự thật duy nhất.
- Copy hàng weight/bias của lớp cũ theo raw class ID.
- Fisher của hàng head mới bằng 0.
- `theta_star` của hàng mới có thể nhận initial weight vì Fisher tương ứng bằng 0.
- Thêm unit test xác nhận logits của lớp cũ không đổi ngay sau head expansion.

### 3.4. Class-order alignment

Nếu ba member có thứ tự cột logits khác nhau, trung bình posterior sẽ sai nhưng có thể không phát sinh exception.

Giải pháp:

- Dùng chung artifact `seen_class_map_task_<t>.json`.
- Prediction artifact phải lưu `raw_class_ids` theo đúng thứ tự cột.
- Trước ensemble phải assert class order, task ID và sample IDs giống nhau.
- Aggregate theo raw class ID hoặc chỉ aggregate theo cột sau khi assertion thành công.

### 3.5. Tách experiment seed và member seed

Ba member dùng cùng toàn bộ RNG state có thể học gần như giống nhau, làm UE thấp giả tạo. Ngược lại, dùng ba `--seed` khác nhau sẽ trộn experiment seed với ensemble diversity.

Giải pháp:

```text
experiment_seed = 0
member_seed      = deterministic_function(experiment_seed, member_id)
```

Experiment seed quyết định protocol, split, task order và các artifact dùng chung. Member seed chỉ quyết định initialization, dropout và DataLoader shuffle của member. Manifest phải lưu cả hai seed và hàm/offset derivation.

### 3.6. Khác biệt với 3-fold ensemble của paper

T-DDI gốc dùng stratified three-fold ensemble. Nếu áp dụng trực tiếp trong CIL, mỗi member chỉ thấy khoảng hai phần ba dữ liệu task, gây bất lợi cho lớp hiếm và làm phức tạp replay/memory fairness.

Giải pháp:

- Bản CIL đầu tiên dùng ba full-data member với independent member seed.
- Gọi đúng tên là deep ensemble CIL hoặc `tddi_ensemble3`, không tuyên bố là tái hiện nguyên vẹn 3-fold training protocol.
- Nếu cần faithful 3-fold reproduction, mở study riêng sau này.

### 3.7. Định nghĩa loss

Joint ensemble loss có thể khiến các member che lỗi cho nhau và không còn giống training của T-DDI gốc.

Giải pháp:

- Train từng member bằng Focal Loss độc lập.
- Giữ `gamma` và các hyperparameter theo config của study.
- Chỉ ensemble khi evaluation:

```text
p_bar = (softmax(z_0) + softmax(z_1) + softmax(z_2)) / 3
```

### 3.8. Replay-memory fairness

Không áp dụng cho EWC thuần túy vì EWC hiện không dùng replay. Khi thêm replay sau này, ba buffer riêng sẽ vô tình tăng memory budget gấp ba.

Giải pháp sau này:

- Một exemplar set logic chung cho toàn ensemble.
- Ba member dùng cùng sample IDs nhưng có thể shuffle khác nhau.
- Báo cáo memory budget theo toàn hệ thống, không theo member.

### 3.9. EWC state và stability-plasticity

Với mỗi member, EWC dùng:

```text
L_total = L_focal + lambda * sum_i(F_i * (theta_i - theta_star_i)^2)
```

`lambda` quá nhỏ không ngăn forgetting; quá lớn làm model không học được lớp mới. Repo hiện tích lũy Fisher không decay:

```text
F_total <- F_total + F_new
```

Giải pháp:

- Fisher và `theta_star` phải riêng cho từng member.
- Baseline đầu giữ accumulation hiện tại và empirical Fisher dựa trên cross-entropy.
- Log riêng classification loss, raw EWC penalty, scaled EWC penalty và total loss.
- Sweep ngắn `ewc_lambda` trên validation, ví dụ `{1, 10, 100, 1000}`.
- Theo dõi đồng thời old-class và new-class Macro-F1.
- Chỉ thêm Fisher decay `gamma` hoặc class-balanced Fisher trong study biến thể sau, không trộn với baseline.

### 3.10. Fisher bị chi phối bởi lớp phổ biến

Dữ liệu DDI long-tail khiến empirical Fisher trên natural loader có thể bảo vệ lớp phổ biến tốt hơn lớp hiếm.

Giải pháp:

- Bản baseline giữ natural loader để tránh đổi định nghĩa method.
- Báo cáo forgetting theo task/lớp và theo frequency bin.
- Nếu có bằng chứng lớp hiếm bị under-protected, mở ablation `balanced_fisher` riêng.

### 3.11. UE qua các task có số lớp khác nhau

Raw entropy của task có 38 lớp không so trực tiếp được với task có 178 lớp.

Giải pháp:

```text
predictive_entropy = H(p_bar)
expected_entropy   = mean_m H(p_m)
mutual_information = H(p_bar) - mean_m H(p_m)
normalized_entropy = H(p_bar) / log(C_t)
normalized_mi      = mutual_information / log(C_t)
confidence         = 1 - normalized_entropy
```

Ngoài ra lưu mean variance giữa member probabilities và pairwise disagreement để kiểm tra ensemble có diversity thật hay không.

### 3.12. Threshold và test leakage

Threshold đặt tay vẫn có thể gây test leakage nếu điều chỉnh sau khi xem test.

Giải pháp:

- Đặt trước candidate grid, ví dụ `{0.70, 0.75, 0.80, 0.85, 0.90}`.
- Chọn threshold chỉ bằng validation theo trade-off accuracy/coverage đã định nghĩa.
- Lưu artifact gồm grid, selection rule, validation metrics và selected threshold.
- Đóng băng artifact trước khi chạy test.
- Báo cáo cả full-set metrics và high-confidence metrics kèm coverage.

### 3.13. Checkpoint và dung lượng ổ đĩa

Nếu lưu model, Fisher, `theta_star`, optimizer và tất cả task cho ba member, dung lượng tăng nhanh.

Giải pháp:

- Checkpoint resume tại task boundary lưu `model_state`, `fisher_total`, class map, seed/RNG state, task ID và config.
- Không lưu riêng `theta_star` tại task boundary; tái tạo bằng `clone(model_state)` khi resume.
- Chỉ giữ `latest_ewc_state.pt` cho resume và `final_model.pt` cho inference của mỗi member.
- Metrics/predictions từng task vẫn được giữ để tính forgetting, nhưng không cần giữ Fisher của mọi task.
- Không lưu optimizer state cho mọi task; chỉ lưu nếu cần resume giữa task/epoch.

### 3.14. Ensemble thiếu diversity

Ba member có thể mắc cùng lỗi dù seed khác nhau, khiến mutual information gần 0 và UE không hữu ích.

Giải pháp:

- Independent initialization, dropout và batch order là yêu cầu tối thiểu.
- Đo pairwise disagreement, pairwise KL, error overlap và mean MI.
- Chưa dùng bootstrap trong baseline; nếu diversity không đủ thì mở ablation riêng.

## 4. Tiêu chí nghiệm thu toàn hệ thống

1. Study P0-P8 cũ và checkpoint cũ vẫn hoạt động, không bị ghi đè.
2. Có variant/config mới, tên rõ ràng và output directory riêng.
3. Experiment seed 0 và ba member seed được lưu trong manifest.
4. Ba member có cùng sample order và raw-class column order ở mỗi task.
5. Head mở rộng đúng và Fisher của các hàng lớp mới bằng 0 trước khi học task mới.
6. Mỗi member dùng Focal Loss độc lập và EWC state riêng.
7. Ensemble bằng mean probabilities; không mean raw logits.
8. Có normalized entropy, normalized MI, variance và confidence.
9. Threshold chỉ được chọn từ validation artifact.
10. Resume từ `latest_ewc_state.pt` tạo cùng class map và tiếp tục đúng task.
11. Smoke test task 0 và task 1 chạy xong trên GPU, có peak VRAM và runtime audit.
12. Test hiện có của repo vẫn pass.

## 5. Prompt bao hàm cho Codex

Dùng prompt này nếu muốn Codex thực hiện toàn bộ trong một task lớn:

```text
Bạn đang làm việc trên repository DDI-CIL. Hãy đọc README.md, CIL.md,
docs/TDDI_ENSEMBLE3_EWC_PLAN.md và các implementation hiện tại trong
src/models/mlp.py, src/methods/ewc.py, src/training/train_cil.py,
src/eval/cil_evaluation.py, src/eval/calibration_metrics.py trước khi sửa code.

Mục tiêu: bổ sung study mới method=EWC × backbone=tddi_ensemble3 × protocol=P4
× experiment_seed=0. Không sửa hoặc trộn kết quả study P0-P8 đang khóa.

Yêu cầu kiến trúc và training:
- Tạo T-DDI numerical-only member: input LayerNorm(3780), MLP
  3780->7560->7560, activation/dropout theo config, expandable head C_t.
- Có ba member độc lập, nhưng train lần lượt ba CIL trajectory; không giữ ba
  optimizer/model trên GPU đồng thời.
- Mỗi member dùng toàn bộ dữ liệu mỗi task, cùng protocol/split/task order và
  cùng raw-class order.
- Tách experiment_seed khỏi member_seed. Experiment seed 0 quyết định protocol
  và artifact dùng chung; member seed chỉ quyết định model initialization,
  dropout và DataLoader shuffle. Derivation phải deterministic và được lưu manifest.
- Mỗi member train độc lập bằng Focal Loss giống T-DDI; không dùng joint ensemble
  loss. Ensemble chỉ ở evaluation bằng trung bình softmax probabilities.
- EWC Fisher và theta_star phải riêng cho từng member. Giữ empirical diagonal
  Fisher dựa trên cross-entropy và Fisher accumulation baseline hiện tại.
- Expand Fisher/theta/head theo raw class ID; Fisher của head rows mới bằng 0.
- Log classification loss, raw EWC penalty, scaled penalty và total loss.
- EWC checkpoint tại task boundary lưu model_state, fisher_total, class map,
  task/member/seed/config/RNG metadata. Không lưu theta_star trùng lặp; khôi phục
  theta_star từ model_state khi resume. Chỉ giữ latest resume state và final model
  cho mỗi member, không xóa artifact của user.
- Export per-member probabilities/logits, sample IDs, task ID và raw class column
  order. Trước ensemble phải assert alignment tuyệt đối.
- Tính ensemble predictive entropy, expected member entropy, mutual information,
  normalized entropy, normalized MI, variance, confidence và disagreement.
  Chuẩn hóa bằng log(C_t), xử lý an toàn C_t <= 1.
- Threshold confidence dùng candidate grid đặt trong config, chọn chỉ trên
  validation, lưu artifact rồi đóng băng trước test. Báo cáo accuracy/Macro-F1,
  ECE/NLL/Brier và accuracy/coverage của high-confidence subset.
- Không triển khai replay, replay-distillation, GEM, A-GEM, bootstrap hoặc
  balanced Fisher trong task này.
- Thêm unit/integration tests cho seed separation, head/Fisher expansion,
  probability aggregation, class/sample alignment, normalized UE, checkpoint
  round-trip và backward compatibility.
- Thêm smoke-test config task 0/task 1 với batch nhỏ, ít epoch, peak VRAM/runtime
  audit; không tự chạy full experiment tốn nhiều giờ.

Trước khi sửa, kiểm tra git status và các thay đổi sẵn có của user; không ghi đè
thay đổi không liên quan. Triển khai theo các module nhỏ, chạy test sau mỗi nhóm
thay đổi. Cuối cùng báo cáo file đã sửa, command smoke test, test results, giới hạn
còn lại và xác nhận rằng study cũ không bị thay đổi.
```

## 6. Bộ prompt chia nhỏ theo từng task

Nên dùng các prompt dưới đây theo đúng thứ tự. Mỗi task phải hoàn thành code, test và tài liệu của chính nó trước khi chuyển sang task tiếp theo.

### Task 0 — Audit và thiết kế interface, không sửa behavior

```text
Đọc docs/TDDI_ENSEMBLE3_EWC_PLAN.md và audit repository DDI-CIL cho study mới
EWC × tddi_ensemble3 × P4 × experiment seed 0. Chưa triển khai model.

Hãy xác định chính xác:
1. CLI/config nào tạo model, seed, optimizer và output directory.
2. Luồng expandable head và class map.
3. Luồng EWC Fisher/theta_star và checkpoint hiện tại.
4. Evaluation/export hiện có và những invariant cần giữ.
5. Test nào cần thêm ở từng giai đoạn.

Tạo một implementation checklist ngắn trong docs, chỉ rõ file/function dự kiến
sửa và acceptance criteria. Không thay đổi behavior hoặc result study P0-P8.
Chạy test hiện có để ghi baseline và báo cáo các failure có sẵn.
```

### Task 1 — Tách experiment seed và member seed

```text
Triển khai riêng phần seed theo docs/TDDI_ENSEMBLE3_EWC_PLAN.md. Chưa tạo ensemble
và chưa đổi backbone.

Thêm khái niệm experiment_seed và member_id/member_seed với derivation deterministic.
Experiment seed tiếp tục điều khiển protocol, split, task order và artifact dùng chung;
member seed chỉ điều khiển initialization, dropout và DataLoader shuffle. Giữ backward
compatibility: các command cũ chỉ có --seed phải cho kết quả/behavior cũ.

Lưu cả hai seed và derivation metadata vào manifest/audit. Thêm unit tests chứng minh:
- cùng experiment seed + member ID tạo cùng member seed;
- member ID khác tạo model RNG khác;
- class map/protocol metadata không đổi giữa member;
- CLI cũ vẫn dùng được.

Chỉ sửa các file cần thiết cho seed. Chạy test liên quan và báo cáo kết quả.
```

### Task 2 — Checkpoint EWC gọn và resume an toàn

```text
Chỉ triển khai checkpoint/resume cho EWC theo
docs/TDDI_ENSEMBLE3_EWC_PLAN.md; chưa tạo backbone hoặc ensemble mới.

Tại task boundary, checkpoint tối thiểu phải lưu model_state, fisher_total,
seen_class_map, completed_task_id, experiment/member seed metadata, config và RNG
state cần thiết. Không lưu theta_star trùng lặp; khi resume tạo theta_star bằng clone
model_state trước khi train task tiếp theo. Không lưu/xóa optimizer checkpoint ngoài
phạm vi cần thiết và không xóa artifact hiện có của user.

Thêm version/schema cho checkpoint, validation rõ ràng khi metadata/class map không
khớp, atomic write nếu repo đã có convention phù hợp, và tests cho round-trip:
continuous run so với save/resume phải có cùng head shape, Fisher, theta_star và next
task. Giữ command cũ hoạt động nếu không bật resume mới.
```

### Task 3 — Tạo một T-DDI paper-size member

```text
Thêm backbone variant mới cho một T-DDI numerical-only member theo
docs/TDDI_ENSEMBLE3_EWC_PLAN.md:
LayerNorm input 3780 -> Linear 7560 -> activation/dropout -> Linear 7560 ->
activation/dropout -> expandable head C_t.

Không đổi variant tddi=3780->1024->512 hiện tại và không sửa kết quả P0-P8. Đặt tên
variant mới rõ ràng, ví dụ tddi_paper_member. Hidden dimensions và dropout/activation
phải được ghi trong config/manifest. Model phải hỗ trợ forward, encode,
forward_with_latent và head expansion theo interface hiện có.

Thêm tests cho parameter shapes/count, input LayerNorm placement, forward shape,
latent shape, head expansion và copy old-class rows. Chỉ smoke forward/backward bằng
batch rất nhỏ; không train dataset đầy đủ.
```

### Task 4 — Bảo đảm EWC đúng với paper-size member

```text
Tích hợp và kiểm chứng method EWC với variant tddi_paper_member đã có. Chưa chạy ba
member và chưa ensemble.

Giữ member training loss là Focal Loss; empirical diagonal Fisher tiếp tục dựa trên
cross-entropy như baseline hiện tại. Xác nhận Fisher/theta_star mở rộng đúng khi head
tăng: old rows ánh xạ theo raw class ID, new Fisher rows bằng 0. Log riêng
classification_loss, raw_ewc_penalty, scaled_ewc_penalty và total_loss.

Thêm tests cho penalty bằng 0 tại theta_star, penalty dương khi weight quan trọng bị
dịch chuyển, new head rows không bị phạt trước khi học, và two-task synthetic smoke
test. Không thêm Fisher decay hoặc balanced Fisher.
```

### Task 5 — Export prediction của từng member

```text
Mở rộng evaluation/export để mỗi run/member có thể lưu artifact phục vụ offline
ensemble. Mỗi artifact phải chứa sample IDs ổn định, labels, task ID, member ID,
experiment/member seed, logits hoặc probabilities, và raw_class_ids đúng thứ tự cột.

Không ensemble trong task này. Thêm validation không cho duplicate sample ID, mismatch
row count, mismatch probability width hoặc thiếu class-order metadata. Giữ API/output
cũ tương thích. Thêm tests cho export/load round-trip và deterministic sample order.
```

### Task 6 — Offline ensemble và normalized UE

```text
Triển khai offline ensemble cho đúng ba prediction artifact đã tạo ở Task 5.
Không train model trong task này.

Trước khi aggregate, assert ba artifact có cùng protocol, experiment seed, task ID,
sample IDs, labels và raw_class_ids; member IDs phải khác nhau. Aggregate bằng mean
probabilities, tuyệt đối không mean raw logits.

Tính và export predictive entropy, expected member entropy, mutual information,
normalized entropy, normalized MI, mean probability variance, confidence và pairwise
disagreement. Chuẩn hóa theo log(C_t), xử lý C_t<=1. Thêm numerical tests bằng tensor
nhỏ có kết quả tính tay, test permutation/misalignment phải fail rõ ràng, và test ba
member giống hệt nhau cho MI gần 0.
```

### Task 7 — Threshold validation và calibration report

```text
Thêm pipeline chọn confidence threshold cho offline ensemble. Candidate grid phải đến
từ config và được đặt trước. Chỉ validation được dùng để chọn threshold; test chỉ được
đánh giá sau khi load frozen threshold artifact.

Artifact phải lưu grid, selection rule, selected threshold, validation accuracy,
Macro-F1, coverage, timestamp/config hash và source split. Báo cáo full-set metrics,
high-confidence accuracy/Macro-F1/coverage, ECE, NLL và Brier. Thêm guard chống chọn
threshold từ test split và tests cho selection/freeze/load. Không hard-code 0.88.
```

### Task 8 — Orchestrator ba member chạy lần lượt

```text
Tạo entrypoint/config study tddi_ensemble3 để điều phối ba CIL trajectory EWC lần lượt.
Không load/train đồng thời cả ba member trên GPU. Dùng experiment seed 0 và ba member
seed deterministic đã triển khai. Mỗi member dùng toàn bộ task data, cùng protocol P4
và cùng class map, nhưng model RNG/DataLoader RNG riêng.

Output của mỗi member phải nằm trong namespace riêng; sau khi cả ba hoàn tất task cần
có command gọi offline ensemble. Có resume theo member/task, không ghi đè run hoàn tất,
và manifest tổng hợp liên kết ba member artifacts. Không tự động chạy full 8-task job.

Thêm dry-run test và synthetic integration test chứng minh thứ tự chạy tuần tự, class
alignment và ensemble invocation đúng.
```

### Task 9 — Chuẩn bị smoke config cho máy GPU riêng

```text
Chỉ chuẩn bị config và runbook để người dùng chuyển sang máy GPU riêng chạy smoke test
cho study EWC × tddi_ensemble3 × P4 × seed 0. Không chạy training hoặc smoke test trên
máy hiện tại.

Tạo smoke config bắt đầu bằng member_id=0, chỉ task 0, 2-3 epoch, batch 64 hoặc 128.
Chuẩn bị thêm config/lệnh task 1 nhưng chỉ được chạy trên máy GPU riêng sau khi task 0
đạt tiêu chí go. Config phải ghi rõ microbatch, effective batch, gradient accumulation,
đường dẫn dữ liệu/output và không được tự động khởi chạy full experiment.

Cập nhật docs/RUNBOOK.md với command chính xác để setup môi trường, kiểm tra GPU, chạy
smoke config và thu thập peak allocated/reserved VRAM, runtime/epoch, checkpoint size,
classification/EWC loss components, head sizes, Fisher statistics và validation metrics.
Nếu OOM, runbook hướng dẫn giảm microbatch và tăng gradient accumulation nhưng không
âm thầm đổi effective batch hoặc hyperparameter. Nêu rõ file log/artifact người dùng
cần gửi lại để đánh giá go/no-go trước full run.
```

## 7. Thứ tự chạy khuyến nghị

```text
Task 0 audit
  -> Task 1 seed
  -> Task 2 checkpoint
  -> Task 3 paper-size member
  -> Task 4 EWC verification
  -> Task 5 prediction export
  -> Task 6 offline ensemble + UE
  -> Task 7 threshold/calibration
  -> Task 8 sequential orchestrator
  -> Task 9 chuẩn bị config/runbook cho máy GPU riêng
  -> chỉ sau đó mới chạy full 8-task experiment
```

Sau mỗi task nên commit riêng. Nếu một task làm thay đổi invariant hoặc interface đã chốt, cập nhật tài liệu này và test liên quan trước khi chuyển sang task tiếp theo.
