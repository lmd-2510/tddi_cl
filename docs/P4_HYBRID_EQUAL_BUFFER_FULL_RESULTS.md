# P4 Hybrid equal-buffer — full run 3 members

Tài liệu này tổng hợp bundle
`p4_hybrid_equal_buffer_full8_e30_mem4_review_20260930T081319Z.zip`.
Thứ tự trình bày được cố định là **config trước, kết quả sau** để có thể dùng
làm mẫu đối chiếu cho các run tiếp theo.

## 1. Định danh và tính toàn vẹn

| Thành phần | Giá trị |
|---|---|
| Loại run | Full run, `member_0`, `member_1`, `member_2` và offline ensemble |
| Config | `configs/p4_hybrid_equal_buffer_full8_e30_mem4.json` |
| Protocol | P4 `constrained_mass_balanced` |
| Experiment / fold seed | `0 / 42` |
| Member seeds | `409845317`, `215626784`, `3041879697` |
| Trạng thái train + offline ensemble | Hoàn tất (`exit=0`) |
| Final report | Hoàn tất (`exit=0`) |
| Visualization | Hoàn tất (`exit=0`) |
| Script phân tích riêng task 6 → 7 | Thất bại (`exit=1`); số liệu được dựng lại từ các CSV diagnostics đã tạo thành công |
| Run timestamp | `20260930T081319Z` |
| ZIP SHA256 kiểm tra lại | `ced27b250c83838ba4a668e141cd1e4f18b868453de8653cf7a7ce62d7a16a20` |
| SHA256 khai báo | `ced27b250c83838ba4a668e141cd1e4f18b868453de8653cf7a7ce62d7a16a20` |
| Config SHA256 | `7573040454f53acafe5e8ccfdc8f05eeb383b15a64b828b43ae0919bbde8e262` |
| Task-file SHA256 | `9d22af8618fe03c40b92e2aabbab8b68a89de83bc0de4f51e24932951cf298f4` |
| Fold-assignment SHA256 | `c84dc034431ed27907cdfd7217fa3460e0a9480bbf8343e5c91e68927de79a43` |
| Git commit ghi trong run | `e55a43e137c6909c4a84ea96bd35e7a126c3509f` |
| Git dirty | `true` |

Hai SHA256 của ZIP khớp hoàn toàn, nên file tải về không bị hỏng hoặc thay đổi
sau khi đóng gói. Tuy nhiên, `git_dirty=true` có nghĩa source tree tại thời điểm
chạy có thay đổi chưa commit. Kết quả vẫn hợp lệ như một quan sát thực nghiệm,
nhưng không nên coi commit nói trên là một mốc tái lập độc lập đầy đủ nếu không
lưu kèm chính xác diff của working tree.

Preprocessing artifact được đóng băng riêng cho từng fold/member:

| Member | Validation fold | Preprocessing SHA256 | Run-config SHA256 |
|---:|---:|---|---|
| 0 | 0 | `1664dff205f2d505b539173e9858683060233255f86050b6eef63b3673d13cf0` | `15dd792e1262b6a0ee5d3a37a8aab6178d6b0af179f68756e2364ad8e751b342` |
| 1 | 1 | `dcbf4d52cdf59090a3f81df579b2367aff02a9ec5d8834b87ea8bb9abb4b5be2` | `3aa003c751b7dfa30fb9ec3fe3e3413f44a10c19a32a9507dc05e23125f1b6a6` |
| 2 | 2 | `e5a53a3794adafe99f880f84b59fdba67d54c47d81e407d3029e5127623e5525` | `f9fbbaec21f17217d322fb614140e9438739bbd2ebfb80eabd2323c7030384c8` |

## 2. Config của thí nghiệm

### 2.1. Dữ liệu và protocol

| Thành phần | Giá trị |
|---|---|
| Protocol | P4 `constrained_mass_balanced` |
| Task layout | `[38, 20, 20, 20, 20, 20, 20, 20]` |
| Tổng số class | `178` |
| Số member | `3` |
| Preprocessing | `task0_standard_frozen` |
| Train / validation / test rows toàn bộ | `520,841 / 173,614 / 173,614` |
| Train-mass CV giữa task | `0.096595` |
| Tỷ lệ task train lớn nhất / nhỏ nhất | `1.284690` |

| Task | Classes | Train samples | Validation samples | Test samples |
|---:|---:|---:|---:|---:|
| 0 | 38 | 60,841 | 20,282 | 20,280 |
| 1 | 20 | 60,799 | 20,265 | 20,269 |
| 2 | 20 | 78,104 | 26,032 | 26,035 |
| 3 | 20 | 66,269 | 22,088 | 22,089 |
| 4 | 20 | 72,433 | 24,146 | 24,144 |
| 5 | 20 | 60,800 | 20,270 | 20,268 |
| 6 | 20 | 60,799 | 20,266 | 20,265 |
| 7 | 20 | 60,796 | 20,265 | 20,264 |

**Task 7 không có nhiều dữ liệu hơn các task trước.** Nó có `60,796` train
samples, gần như bằng task 1, 5 và 6, đồng thời nhỏ hơn task 2, 3 và 4. Vì vậy
không thể giải thích mức giảm cuối run chỉ bằng giả thuyết “task 7 quá lớn”.

### 2.2. Model

| Thành phần | Giá trị |
|---|---|
| Variant | `tddi_paper_member` |
| Input dimension | `3780` |
| Hidden layers | `[7560, 7560]` |
| Activation | GELU |
| Normalization | LayerNorm |
| Dropout | `0.2` |
| Classifier head | 178 raw classes, future-class mask theo boundary |

### 2.3. Training và loss thực thi

| Thành phần | Giá trị |
|---|---|
| Loss variant | `hybrid` |
| Current samples | Focal loss, `gamma=1.0` |
| Replay samples | Cross-entropy |
| Optimizer | AdamW |
| Learning rate | `0.001` |
| Weight decay | `0.0001` |
| Batch size | `64` |
| Effective batch size | `1024` |
| Gradient accumulation | `16` steps |
| Max epochs / patience | `30 / 5` |
| Weight alignment | `none` |

Tên method/training policy trong artifact vẫn là
`replay_distill_fixed_budget_uniform` / `frozen_fold_replay_distill_v1`, và
config dùng chung vẫn chứa `distill_alpha=1.0`, `temperature=2.0` cùng
`feature_distill_weight=0.5`. Tuy nhiên audit của run ghi:

```text
logit_distillation_loss          = 0
feature_distillation_loss        = 0
scaled_logit_distillation_loss   = 0
scaled_feature_distillation_loss = 0
scaled_ewc_penalty               = 0
```

Do đó cách gọi khoa học đúng cho run này là **Hybrid: Focal trên current + CE
trên replay, không distillation, không EWC, không weight alignment**. Không nên
đọc tên legacy `replay_distill...` rồi kết luận rằng distillation đã được dùng.

### 2.4. Buffer và replay

| Thành phần | Giá trị |
|---|---|
| Budget policy | `per_member_4_percent` |
| Buffer mỗi member | `27,778` slots |
| Tổng physical slots của 3 members | `83,334` |
| Buffer policy | `fold_equal_class_capacity_v1` |
| Cách chia | Equal-class có giới hạn bởi số mẫu khả dụng; phần dư được water-fill lại |
| Ranking | `raw_sample_normalized_class_mean_control_v1` |
| Replay fraction | `0.125` |
| Repeat cap | `3` |
| Replay bắt đầu | Task 1 |
| Sampler audit thực tế | `fold_fraction_capped_rotating_v1` |

“Equal-class” ở đây không có nghĩa mọi class cuối cùng luôn chứa đúng cùng một
số exemplar. Class hiếm được giữ tối đa số mẫu hiện có; slot thừa được phân bổ
lại cho các class còn sức chứa. Đây là water-filled equal-class allocation,
không phải nhân bản class hiếm cho đầy quota.

Tại đầu task 7, buffer của member 0 chứa 158 class cũ:

| Origin task | Classes | Slots trước task 7 | Slots/class trung bình | Min–max |
|---:|---:|---:|---:|---:|
| 0 | 38 | 6,679 | 175.76 | 7–370 |
| 1 | 20 | 4,087 | 204.35 | 14–370 |
| 2 | 20 | 2,275 | 113.75 | 3–369 |
| 3 | 20 | 3,041 | 152.05 | 4–370 |
| 4 | 20 | 2,987 | 149.35 | 4–370 |
| 5 | 20 | 4,384 | 219.20 | 12–370 |
| 6 | 20 | 4,325 | 216.25 | 11–370 |

Task 6 không bị thiếu quota: nó có `4,325` slots, cao thứ hai trong các nhóm
20-class. Vì vậy mức quên lớn của task 6 không được giải thích bởi việc nhóm này
nhận quá ít chỗ trong buffer.

Audit epoch đầu task 7 của member 0 cho thấy:

| Thuộc tính | Giá trị |
|---|---:|
| Current draws | 54,043 |
| Replay draws | 7,720 |
| Tổng draws | 61,763 |
| Replay fraction thực tế | `0.124994` |
| Old classes có trong replay | `158 / 158` |
| Unique replay exemplars trong epoch | 6,439 |
| Tỷ lệ unique exemplar của buffer trong epoch | `0.231802` |
| Max repeat / repeat cap | `3 / 3` |

Như vậy class coverage là đầy đủ, nhưng tỷ lệ update vẫn khoảng **7 current : 1
replay**. Sampler không bỏ sót task 6: trong epoch này task 6 nhận `1,025`
draws, tương đương task 5 (`1,032`) và gần task 1 (`1,065`). Điểm yếu hợp lý
hơn nằm ở cân bằng gradient current/replay và classifier bias, không phải ở việc
task 6 biến mất khỏi buffer.

## 3. Kết quả offline ensemble

### 3.1. Full-set metrics theo boundary

Đây là kết quả chính, tính trên **toàn bộ test samples của tất cả class đã thấy**
tại mỗi boundary.

| Task | Samples | Accuracy | Balanced Acc. | Macro-F1 | Weighted-F1 |
|---:|---:|---:|---:|---:|---:|
| 0 | 20,280 | 0.956213 | 0.935983 | 0.946139 | 0.956243 |
| 1 | 40,549 | 0.791142 | 0.881537 | 0.812324 | 0.765777 |
| 2 | 66,584 | 0.686636 | 0.735006 | 0.612625 | 0.659879 |
| 3 | 88,673 | 0.416880 | 0.733530 | 0.601885 | 0.351716 |
| 4 | 112,817 | 0.333824 | 0.719029 | 0.559922 | 0.251068 |
| 5 | 133,085 | 0.256032 | 0.706588 | 0.517996 | 0.195806 |
| 6 | 153,350 | 0.245804 | 0.720674 | 0.510158 | 0.192535 |
| 7 | 173,614 | **0.222286** | **0.695909** | **0.440707** | **0.166964** |
| Mean 8 tasks | — | 0.488602 | 0.766032 | 0.625220 | 0.442499 |

Kết quả cuối cùng cần trích dẫn cho run là **Macro-F1 = 0.440707 trên full test
set tại task 7**. Giá trị `0.625220` chỉ là trung bình tám boundary, không phải
kết quả cuối. Balanced Accuracy bằng macro recall; chênh lệch lớn giữa Balanced
Accuracy (`0.695909`) và Macro-F1 (`0.440707`) báo hiệu precision theo class yếu,
không chỉ thiếu recall.

### 3.2. Selective metrics tại task 7

| Threshold | Selected / total | Coverage | Accuracy | Balanced Acc. | Macro-F1 |
|---:|---:|---:|---:|---:|---:|
| `0.83` | 70,649 / 173,614 | `0.406931` | 0.417713 | 0.758565 | 0.607123 |

`0.607123` là Macro-F1 chỉ trên khoảng **40.69%** mẫu vượt confidence
threshold. Nó hữu ích cho selective prediction nhưng không được thay thế
`0.440707` khi báo cáo full-set CL performance.

### 3.3. Member cuối run và ensemble gain

| Model | Accuracy | Macro recall / Balanced Acc. | Macro-F1 |
|---|---:|---:|---:|
| Member 0 | 0.215824 | 0.640646 | 0.405011 |
| Member 1 | 0.216647 | 0.680155 | 0.411920 |
| Member 2 | 0.208716 | 0.634617 | 0.387392 |
| Member mean ± population std | — | — | `0.401441 ± 0.010327` |
| Offline ensemble | **0.222286** | **0.695909** | **0.440707** |

Offline ensemble tăng `0.039266` Macro-F1, tương đương khoảng `+9.78%` so với
member mean. Ensemble có giá trị rõ ràng, nhưng không đủ bù mức suy giảm ở
boundary cuối.

Tại task 7, mean pairwise disagreement là `0.325020`, cao nhất trong cả run;
mean member normalized MI là `0.146181`. Ba member tạo diversity đủ để ensemble
tăng điểm, nhưng mức bất đồng cao cũng cho thấy boundary cuối thiếu ổn định.

## 4. Phân tích task 6 → task 7

### 4.1. Mức giảm tổng thể

| Metric | Task 6 | Task 7 | Chênh lệch |
|---|---:|---:|---:|
| Accuracy | 0.245804 | 0.222286 | -0.023518 |
| Macro precision | 0.492552 | 0.432174 | -0.060378 |
| Balanced Acc. / macro recall | 0.720674 | 0.695909 | -0.024764 |
| Macro-F1 | 0.510158 | 0.440707 | **-0.069451** |
| Weighted-F1 | 0.192535 | 0.166964 | -0.025572 |

Macro-F1 giảm `6.945` điểm phần trăm, tương đương khoảng `13.61%` tương đối.
Macro precision giảm mạnh hơn macro recall, nên vấn đề chính không phải model
“không đoán ra class mới”; nó là **ranh giới classifier bị lệch và tạo nhiều
false positive**.

### 4.2. Old/current nhìn từ confusion đầy đủ

Bảng dưới lấy trung bình classwise metric từ confusion matrix đầy đủ tại từng
boundary. Cách tính này giữ cả false positive từ các origin-task khác nên phù
hợp để chẩn đoán cạnh tranh giữa class cũ và class mới.

| Boundary / nhóm | Classes | Macro precision | Macro recall | Macro-F1 |
|---|---:|---:|---:|---:|
| Task 6 — old (task 0–5) | 138 | 0.515190 | 0.693675 | 0.518890 |
| Task 6 — current (task 6) | 20 | 0.336350 | 0.906961 | 0.449909 |
| Task 7 — old (task 0–6) | 158 | 0.461592 | 0.683191 | 0.462910 |
| Task 7 — current (task 7) | 20 | **0.199768** | **0.796384** | **0.265306** |

Nhóm task 7 có recall cao (`0.796`) nhưng precision rất thấp (`0.200`). Model
đang dự đoán nhãn task 7 cho quá nhiều mẫu cũ. Đồng thời old-class Macro-F1 giảm
từ `0.519` xuống `0.463`. Vì thế boundary cuối gặp **cả forgetting ở old class
và bias sang new class**.

### 4.3. Nhóm origin task bị giảm ở boundary cuối

| Origin task | Classes | Mean class-F1 Δ (T7−T6) | Precision Δ | Recall Δ | Số class giảm F1 |
|---:|---:|---:|---:|---:|---:|
| 0 | 38 | -0.059456 | -0.051329 | +0.000428 | 29 |
| 1 | 20 | -0.066502 | -0.085081 | +0.064622 | 17 |
| 2 | 20 | -0.019316 | -0.019443 | -0.027114 | 9 |
| 3 | 20 | -0.042535 | -0.037085 | -0.011120 | 11 |
| 4 | 20 | -0.057150 | -0.058602 | -0.018063 | 14 |
| 5 | 20 | +0.000845 | +0.017714 | -0.001234 | 11 |
| 6 | 20 | **-0.075635** | +0.035439 | **-0.304014** | 11 |

Task 6 là nhóm giảm class-F1 trung bình lớn nhất, chủ yếu vì recall giảm rất
mạnh (`-0.304`). Điều này đáng chú ý vì buffer và replay audit cho thấy task 6
không bị thiếu slot hoặc thiếu draws. Do đó nguyên nhân gần hơn là interference
khi học task 7 và dịch chuyển decision boundary.

Final report còn cho task-group forgetting trung bình trên old tasks là
`0.180361`; task 6 lớn nhất với `0.269635`, tiếp theo là task 1 (`0.230237`) và
task 0 (`0.220930`). Task-group metric này được tính trên từng task slice, nên
không đồng nhất với class-F1 từ confusion đầy đủ ở bảng trên; hai số phục vụ hai
góc nhìn khác nhau.

## 5. Class và confusion bị ảnh hưởng

### 5.1. Class có support đáng kể nhưng quên mạnh

Các dòng dưới chỉ lấy class có ít nhất 100 test samples để tránh xếp hạng bị
chi phối hoàn toàn bởi class chỉ có vài mẫu.

| Class label | Support | Best previous F1 | Final F1 | Forgetting drop |
|---|---:|---:|---:|---:|
| `t0_balanced:c10` | 7,684 | 0.955551 | 0.111658 | 0.843893 |
| `t0_balanced:c9` | 2,541 | 0.935044 | 0.157744 | 0.777300 |
| `t2_balanced:c137` | 24,545 | 0.786166 | 0.029711 | 0.756455 |
| `t1_balanced:c32` | 5,777 | 0.946103 | 0.201351 | 0.744752 |
| `t0_balanced:c27` | 1,718 | 0.972799 | 0.246028 | 0.726771 |
| `t0_balanced:c26` | 2,387 | 0.955425 | 0.251877 | 0.703548 |
| `t1_balanced:c58` | 3,361 | 0.875200 | 0.193146 | 0.682054 |
| `t1_balanced:c30` | 6,536 | 0.832619 | 0.159280 | 0.673338 |
| `t0_balanced:c124` | 1,445 | 0.937458 | 0.267677 | 0.669781 |
| `t0_balanced:c152` | 922 | 0.990239 | 0.409500 | 0.580738 |

Ensemble có 8 class final F1 bằng 0: `c104`, `c132`, `c136`, `c142`, `c155`,
`c174`, `c187`, `c207`. Tất cả chỉ có 1–6 test samples, nên đây là vấn đề quan
trọng đối với Macro-F1 nhưng không nên dùng riêng chúng để suy luận cơ chế lỗi.

### 5.2. Confusion lớn nhất tại task 7

| True class | Predicted class | Count | Tỷ lệ trong true row |
|---|---|---:|---:|
| `t4_balanced:c6` | `t7_balanced:c7` | 10,788 | 0.495522 |
| `t2_balanced:c137` | `t7_balanced:c28` | 8,536 | 0.347769 |
| `t3_balanced:c15` | `t7_balanced:c7` | 6,290 | 0.320771 |
| `t2_balanced:c137` | `t7_balanced:c50` | 6,200 | 0.252597 |
| `t2_balanced:c137` | `t7_balanced:c7` | 5,668 | 0.230923 |
| `t3_balanced:c15` | `t7_balanced:c28` | 4,483 | 0.228620 |
| `t0_balanced:c10` | `t7_balanced:c7` | 4,226 | 0.549974 |
| `t6_balanced:c25` | `t7_balanced:c7` | 4,072 | 0.298950 |
| `t6_balanced:c25` | `t7_balanced:c28` | 4,062 | 0.298216 |
| `t4_balanced:c6` | `t7_balanced:c28` | 3,844 | 0.176565 |

Mười confusion lớn nhất đều đẩy class cũ sang class mới của task 7, đặc biệt
vào `c7`, `c28` và `c50`. Đây là bằng chứng trực tiếp cho **new-class attraction
/ classifier boundary shift**, không phải chỉ là nhiễu ngẫu nhiên rải đều.

## 6. Loss audit và diễn giải nguyên nhân

Những điều bundle hỗ trợ kết luận:

1. **Task 7 không lớn bất thường.** Số train samples gần bằng task 6.
2. **Task 6 không thiếu buffer/replay class coverage.** Nó có nhiều slot và
   draws hơn nhiều origin task khác, nhưng vẫn có recall collapse lớn nhất.
3. **Equal-buffer hoạt động đúng mục tiêu lưu trữ**, nhưng không tự động cân
   bằng gradient. Ở task 7, current:replay vẫn xấp xỉ `7:1`.
4. **Run không có loss chống trôi logits/features.** Các trường distillation và
   EWC đều bằng 0.
5. **Không có weight alignment.** Trong khi đó confusion cho thấy classifier
   nghiêng mạnh về một vài class mới.
6. **Task 7 chọn best epoch rất sớm.** Cả ba member đều có best epoch bằng 1 và
   dừng sau 6 epoch (patience 5), cho thấy thêm epoch không sửa được bias này.

Không thể từ một run duy nhất chứng minh Hybrid loss hoặc equal-buffer là
nguyên nhân gây sụt giảm. Muốn kết luận nhân quả phải có ablation P4 cùng seed,
cùng folds và chỉ đổi đúng một thành phần.

## 7. Kết luận và hướng tiếp theo

### Kết luận chính

- Full run và offline ensemble đã hoàn tất thành công.
- Kết quả cuối chính thức là **Macro-F1 `0.440707`**, Accuracy `0.222286`,
  Balanced Accuracy `0.695909` trên 173,614 test samples.
- Ensemble cải thiện đáng kể so với member mean (`+0.039266` Macro-F1), nhưng
  không giải quyết được bias ở task 7.
- Macro-F1 giảm `0.069451` từ task 6 sang task 7. Dữ liệu cho thấy nguyên nhân
  gần nhất là kết hợp của catastrophic interference và classifier bias sang
  new classes, không phải task 7 nhiều samples hay task 6 thiếu slot buffer.
- Equal-buffer giải quyết **storage balance**, nhưng chưa giải quyết hoàn toàn
  **training balance** và **old/new logit calibration**.

### Thí nghiệm ưu tiên

Nên chạy từng thay đổi một trên member 0 trước, dùng validation để chọn config:

1. **Ưu tiên old/new bias correction hoặc weight alignment** sau mỗi boundary.
   Đây là thay đổi khớp nhất với tín hiệu “new recall cao, new precision thấp”.
   Không fit hệ số trên test set.
2. Nếu chưa đủ, giữ buffer như cũ và thay đổi **cân bằng loss/replay theo nhánh**
   để current branch không áp đảo replay branch. Không nên đồng thời đổi cả
   buffer, sampler và loss vì sẽ mất khả năng quy nguyên nhân.
3. Chỉ quay lại logit distillation sau một matched P4 pilot rõ ràng; các thử
   nghiệm P3 trước đó chưa cho bằng chứng rằng chỉ thêm distillation sẽ tốt hơn.

Các chỉ số bắt buộc theo dõi cho pilot kế tiếp:

- full-set task-7 Macro-F1;
- old-class và current-class classwise precision/recall/F1 trong confusion đầy
  đủ;
- recall của origin task 6;
- số lỗi old → `t7:c7/c28/c50`;
- ensemble gain và coverage nếu báo selective metric.

## 8. Phạm vi diễn giải

- Đây là một full run ba member trên P4 và có offline probability ensemble.
- Selective metric luôn phải đi kèm coverage.
- Class có support cực nhỏ làm Macro-F1 nhạy; vì vậy báo cáo dùng cả bảng
  high-support forgetting và zero-F1 classes.
- Script phân tích chuyên biệt task 6 → 7 trả `exit=1`, nhưng các visualization
  và CSV nguồn đã hoàn tất. Các bảng task 6 → 7 trong tài liệu này được tính lại
  trực tiếp từ `classwise_metrics`, `forgetting_by_class`, `top_confusions`,
  buffer allocation và epoch audit có trong bundle.
- Do source tree dirty, cần dùng config, run-config hashes và artifact hashes
  kèm bundle nếu muốn tái lập chính xác.

## 9. Nguồn số liệu trong bundle

- `config/p4_hybrid_equal_buffer_full8_e30_mem4.json`
- `run/full_manifest.json`
- `run/final_results/ensemble3_p4_final_report.md`
- `run/final_results/ensemble3_p4_paper_table.csv`
- `run/final_results/ensemble3_p4_task_matrix.csv`
- `run/final_results/ensemble3_p4_forgetting_summary.csv`
- `run/final_results/ensemble3_p4_member_forgetting_summary.csv`
- `run/final_results/ensemble3_p4_diversity_summary.csv`
- `run/member_{0,1,2}/run_config.json`
- `run/member_{0,1,2}/run_summary.json`
- `run/member_{0,1,2}/metrics.csv`
- `run/member_{0,1,2}/training_audit.csv`
- `run/member_0/task_7/epoch_1_audit.json`
- `run/diagnostics/offline_ensemble/task_{6,7}_test/classwise_metrics_*.csv`
- `run/diagnostics/offline_ensemble/task_{6,7}_test/forgetting_by_class_*.csv`
- `run/diagnostics/offline_ensemble/task_7_test/top_confusions_task7_ensemble.csv`
- `run/diagnostics/offline_ensemble/task_{6,7}_test/buffer_allocation_*.csv`
- `run/diagnostics/offline_ensemble/task_{6,7}_test/task_sample_counts.csv`
- `experiment_status.txt`
