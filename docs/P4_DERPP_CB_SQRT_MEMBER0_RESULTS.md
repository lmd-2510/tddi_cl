# P4 DER++ equal-buffer + square-root replay — member 0 pilot

Tài liệu này tổng hợp bundle
`p4_derpp_cb_sqrt_member0_20260929T080727Z_review.zip`. Thứ tự trình bày được
cố định là **config trước, kết quả sau** để có thể dùng làm mẫu cho các run tiếp
theo.

## 1. Định danh và tính toàn vẹn

| Thành phần | Giá trị |
|---|---|
| Loại run | Pilot, chỉ `member_0` |
| Config | `configs/p4_derpp_cb_sqrt_replay_pilot.json` |
| Method label | `DERPP_CB_SQRT_REPLAY_P4_PILOT` |
| Protocol | P4 `constrained_mass_balanced` |
| Experiment / fold seed | `0 / 42` |
| Trạng thái | Hoàn tất (`pilot_exit=0`) |
| Offline ensemble | Không có |
| Run timestamp | `20260929T080727Z` |
| ZIP SHA256 kiểm tra lại | `5e728c0d98e2c6d791615b7df1944080eaa5d514a8a2eb917a271037e25e4fc3` |
| SHA256 khai báo | `5e728c0d98e2c6d791615b7df1944080eaa5d514a8a2eb917a271037e25e4fc3` |
| Config SHA256 | `ba65861fb093ca8c02385eaf3ab25144ce0b1f27f5e5b4908f321ecdf8e94b92` |
| Task-file SHA256 | `9d22af8618fe03c40b92e2aabbab8b68a89de83bc0de4f51e24932951cf298f4` |
| Fold-assignment SHA256 | `c84dc034431ed27907cdfd7217fa3460e0a9480bbf8343e5c91e68927de79a43` |
| Preprocessing SHA256 | `3011ab205f5d3e963bf464cc3b4a5eaabdb2465d654e9b549938c15bdc64769b` |

Hai SHA256 của ZIP khớp hoàn toàn, nên bundle tải về không bị hỏng hoặc thay
đổi sau khi đóng gói.

## 2. Config của thí nghiệm

### 2.1. Dữ liệu và protocol

| Thành phần | Giá trị |
|---|---|
| Protocol | P4 `constrained_mass_balanced` |
| Task layout | `[38, 20, 20, 20, 20, 20, 20, 20]` |
| Tổng số class | `178` |
| Member | Chỉ `member_0`, validation fold `0` |
| Preprocessing | `task0_standard_frozen` |
| Future-class mask | Bật |
| Head class order | 178 raw class cố định từ task 0 |

P4 cân bằng **khối lượng dữ liệu theo task**, không làm mọi class có cùng số
mẫu. Train-mass CV giữa tám task là `0.096595`; tỷ lệ task lớn nhất/nhỏ nhất là
`1.284690`. Task 7 không phải task lớn nhất.

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

`train_rows` thực sự đi vào member 0 nhỏ hơn tổng train samples do frozen-fold
split; xem bảng kết quả theo task ở phần 3.

### 2.2. Model

| Thành phần | Giá trị |
|---|---|
| Variant | `tddi_paper_member` |
| Input | `3780` |
| Hidden layers | `[7560, 7560]` |
| Activation | GELU |
| Normalization | LayerNorm |
| Dropout | `0.2` |
| Classifier head | `fixed_178_from_task0` |

### 2.3. DER++ loss và optimizer

Loss thực thi có dạng:

```text
L = CE(current) + 0.3 * MSE(replay logits, stored logits)
                + 0.5 * CE(replay labels)
```

| Thành phần | Giá trị |
|---|---|
| Current classification | Cross-entropy |
| Logit replay | MSE trên stored full-head logits |
| Replay-label loss | Cross-entropy |
| `alpha / beta` | `0.3 / 0.5` |
| Batch current / replay | `64 / 64` |
| Epoch mỗi task | `1` |
| Optimizer | AdamW |
| Learning rate | `0.001` |
| Weight decay | `0.0001` |
| Logit refresh | Tắt |

Lưu ý: pilot này chỉ train **một epoch/task** theo config DER++ đang thử. Vì
vậy không được so trực tiếp như một đối chứng chỉ-thay-sampler với các run
Hybrid 30 epoch/task.

### 2.4. Buffer và replay sampler

| Thành phần | Giá trị |
|---|---|
| Buffer policy | `online_equal_class_reservoir_sqrt_replay_v1` |
| Storage | Water-filled equal seen-class quota, không nhân bản vật lý |
| Sampling | `P(class) ∝ retained_count^0.5`, rồi uniform exemplar trong class |
| Sampling exponent | `0.5` |
| Budget scope | Per-member |
| Budget member 0 | `27,778` slots |
| Tổng physical slots của pilot | `27,778` |
| Replay draws/step | `2` |

Đây là **equal-class storage + square-root replay**, không phải strict
class-uniform replay. Căn bậc hai làm mềm chênh lệch: class nhiều exemplar vẫn
được lấy nhiều hơn, nhưng ít cực đoan hơn sampling tỷ lệ thẳng theo số exemplar.

## 3. Kết quả theo task

### 3.1. Macro-F1 validation và test

| Task | Seen classes | Member train rows | Validation Macro-F1 | Test Macro-F1 | Buffer cuối task |
|---:|---:|---:|---:|---:|---:|
| 0 | 38 | 54,082 | 0.826703 | 0.847310 | 27,778 |
| 1 | 58 | 54,042 | 0.761530 | 0.773173 | 27,778 |
| 2 | 78 | 69,422 | 0.688046 | 0.676814 | 27,778 |
| 3 | 98 | 58,904 | 0.620343 | 0.620460 | 27,778 |
| 4 | 118 | 64,384 | 0.578831 | 0.575887 | 27,778 |
| 5 | 138 | 54,048 | 0.511466 | 0.512604 | 27,778 |
| 6 | 158 | 54,045 | 0.490580 | 0.495350 | 27,778 |
| 7 | 178 | 54,043 | 0.451617 | 0.459732 | 27,778 |

Test Macro-F1 giảm liên tục từ `0.847310` ở task 0 xuống `0.459732` ở task 7:
giảm tuyệt đối `0.387578` (khoảng `45.7%` tương đối so với task 0). Tổng thời
gian tám task là khoảng `7,022 giây` (`1.95 giờ`).

### 3.2. Test metrics chi tiết

| Task | Accuracy | Macro Precision | Macro Recall | Macro-F1 | Weighted-F1 | Balanced Acc. |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.878402 | 0.833760 | 0.901168 | 0.847310 | 0.878695 | 0.901168 |
| 1 | 0.787541 | 0.760501 | 0.840703 | 0.773173 | 0.774652 | 0.840703 |
| 2 | 0.786375 | 0.624315 | 0.799136 | 0.676814 | 0.776031 | 0.799136 |
| 3 | 0.599517 | 0.581835 | 0.773295 | 0.620460 | 0.590921 | 0.773295 |
| 4 | 0.455969 | 0.551692 | 0.738685 | 0.575887 | 0.423835 | 0.738685 |
| 5 | 0.383101 | 0.453574 | 0.759987 | 0.512604 | 0.359891 | 0.759987 |
| 6 | 0.342693 | 0.445487 | 0.745903 | 0.495350 | 0.309147 | 0.745903 |
| 7 | 0.361526 | 0.394390 | 0.764650 | 0.459732 | 0.331306 | 0.764650 |

Validation và test task 7 gần nhau (`0.451617` so với `0.459732` Macro-F1),
nên không thấy dấu hiệu test gap bất thường ở mức metric tổng.

## 4. Phân tích task 6 → task 7

| Test metric | Task 6 | Task 7 | Thay đổi |
|---|---:|---:|---:|
| Accuracy | 0.342693 | 0.361526 | **+0.018833** |
| Macro Precision | 0.445487 | 0.394390 | **-0.051097** |
| Macro Recall / Balanced Accuracy | 0.745903 | 0.764650 | **+0.018747** |
| Macro-F1 | 0.495350 | 0.459732 | **-0.035618** |
| Weighted-F1 | 0.309147 | 0.331306 | **+0.022159** |

Macro-F1 giảm `0.035618`, tương đương khoảng `7.19%` tương đối. Tuy nhiên
Accuracy, Macro Recall, Balanced Accuracy và Weighted-F1 đều tăng. Thành phần
kéo Macro-F1 xuống là **Macro Precision giảm mạnh**, không phải recall giảm.

Điều này chỉ ra model mở rộng vùng dự đoán của nhiều class, đặc biệt class task
7, khiến số false positive tăng. Đây là dấu hiệu classifier-boundary/logit bias
rõ hơn là “model không học được task 7”.

### 4.1. Seen-all, old và current ở task 7

| Test group | Samples | Accuracy | Macro Precision | Macro Recall | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|---:|---:|---:|
| Seen all | 173,614 | 0.361526 | 0.394390 | 0.764650 | 0.459732 | 0.331306 |
| Old, task 0–6 | 153,350 | 0.282843 | 0.427992 | 0.752384 | 0.487466 | 0.328223 |
| Current, task 7 | 20,264 | 0.956968 | 0.913354 | 0.861554 | 0.883444 | 0.966767 |

`current` được tính trên các hàng có nhãn thật thuộc task 7. Nó cho thấy model
nhận ra mẫu task 7 rất tốt. Nhưng khi tính precision của từng class task 7 trên
**toàn bộ test set**, Macro-F1 trung bình của 20 class task 7 chỉ khoảng
`0.279042`, do rất nhiều mẫu old bị hút vào các nhãn mới.

Từ bảng classwise có khoảng `94,025` dự đoán đi vào 20 nhãn task 7, trong khi
test chỉ có `20,264` mẫu task 7 và `19,392` true positive task 7. Như vậy có
khoảng `74,633` false positive đi vào nhóm nhãn task 7; tối thiểu khoảng
`73,761` trong số đó phải đến từ mẫu old. Đây là bằng chứng định lượng mạnh cho
classifier shift old → new.

## 5. Buffer và replay exposure tại task 7

| Source task | Classes | Buffer slots | Buffer % | Replay draws | Replay % | Draws/slot |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 38 | 6,057 | 21.81% | 23,778 | 21.98% | 3.93 |
| 1 | 20 | 3,460 | 12.46% | 13,422 | 12.41% | 3.88 |
| 2 | 20 | 2,014 | 7.25% | 9,450 | 8.74% | 4.69 |
| 3 | 20 | 2,780 | 10.01% | 11,593 | 10.72% | 4.17 |
| 4 | 20 | 2,660 | 9.58% | 11,075 | 10.24% | 4.16 |
| 5 | 20 | 3,668 | 13.20% | 13,740 | 12.70% | 3.75 |
| 6 | 20 | 3,617 | 13.02% | 13,664 | 12.63% | 3.78 |
| 7 | 20 | 3,522 | 12.68% | 11,438 | 10.58% | 3.25 |

- Buffer luôn đầy `27,778` slots và **không task nào có class mất hoàn toàn
  exemplar**.
- Ở task 7, buffer sau rebalance old classes còn `24,256` slots; 3,522 slots
  được bổ sung cho task 7.
- Exposure theo class nằm trong khoảng `83–1,083`, median `661`, CV `0.582`.
- Tương quan tuyến tính giữa replay exposure và final class F1 chỉ `0.185`;
  giữa exposure và forgetting drop gần như bằng không (`-0.013`).

Do đó bundle không hỗ trợ kết luận “class quên vì hoàn toàn không được replay”.
Mọi class đều có exemplar và được replay. Vấn đề hợp lý hơn là **chất lượng/cách
dùng replay và độ tương thích của logits**, không chỉ số lượt replay thô.

## 6. Các class bị ảnh hưởng nặng

### 6.1. Forgetting có tác động lớn theo support

| Raw class | Origin task | Test support | Replay exposure | Final F1 | Forgetting drop |
|---:|---:|---:|---:|---:|---:|
| 137 | 2 | 24,545 | 1,023 | 0.227783 | 0.649713 |
| 6 | 4 | 21,771 | 971 | 0.087596 | 0.375554 |
| 15 | 3 | 19,609 | 1,004 | 0.267914 | 0.349098 |
| 10 | 0 | 7,684 | 1,021 | 0.302068 | 0.590971 |
| 30 | 1 | 6,536 | 1,039 | 0.434922 | 0.408640 |
| 32 | 1 | 5,777 | 952 | 0.506751 | 0.376699 |
| 9 | 0 | 2,541 | 1,027 | 0.394119 | 0.480833 |
| 58 | 1 | 3,361 | 982 | 0.467265 | 0.326524 |
| 14 | 1 | 3,182 | 986 | 0.510752 | 0.307059 |
| 13 | 5 | 11,210 | 961 | 0.244396 | 0.085707 |

Các class `137`, `6`, `15` và `10` vừa có support rất lớn vừa giảm mạnh. Chúng
quan trọng hơn các drop cực lớn nhưng chỉ có vài mẫu, ví dụ class `204` giảm
`0.824561` nhưng test support chỉ là `6`.

### 6.2. Confusion nổi bật ở task 7

| True → predicted | Count | Row rate |
|---|---:|---:|
| `6 → 7` | 7,586 | 34.84% |
| `137 → 28` | 5,933 | 24.17% |
| `137 → 50` | 5,431 | 22.13% |
| `15 → 28` | 3,549 | 18.10% |
| `137 → 7` | 3,270 | 13.32% |
| `15 → 7` | 3,229 | 16.47% |
| `25 → 28` | 3,088 | 22.67% |
| `6 → 28` | 2,989 | 13.73% |
| `10 → 7` | 2,629 | 34.21% |
| `13 → 28` | 2,400 | 21.41% |

Các predicted classes `7`, `28` và `50` đều thuộc task 7. Confusion không phân
tán ngẫu nhiên mà tập trung vào một số nhãn mới cụ thể.

### 6.3. Class task 7 có global F1 thấp

| Raw class | Test support | Global F1 |
|---:|---:|---:|
| 174 | 6 | 0.000000 |
| 138 | 7 | 0.099291 |
| 185 | 30 | 0.119658 |
| 173 | 6 | 0.130435 |
| 207 | 6 | 0.137931 |
| 45 | 33 | 0.190476 |
| 189 | 28 | 0.210526 |
| 43 | 257 | 0.246888 |
| 50 | 2,032 | 0.256400 |
| 141 | 225 | 0.278497 |

Nhiều class có recall cao nhưng precision thấp. Vì vậy model có thể đạt metric
tốt khi chỉ xét true-current rows nhưng vẫn có global class F1 thấp.

## 7. Loss audit

| Task | Current CE | Replay-label CE | Replay-logit MSE | Total loss |
|---:|---:|---:|---:|---:|
| 0 | 0.7271 | 0.4799 | 0.9152 | 1.2416 |
| 1 | 0.5752 | 0.6093 | 1.2754 | 1.2625 |
| 2 | 0.2069 | 0.5760 | 1.2203 | 0.8610 |
| 3 | 0.3515 | 0.6471 | 1.3963 | 1.0939 |
| 4 | 0.3674 | 0.7012 | 1.4788 | 1.1616 |
| 5 | 0.5528 | 0.6680 | 1.6347 | 1.3772 |
| 6 | 0.5802 | 0.7083 | 1.7251 | 1.4518 |
| 7 | 0.5340 | 0.6785 | 1.7919 | 1.4109 |

Replay-logit MSE tăng từ `0.9152` lên `1.7919` theo tiến trình. Đây là dấu hiệu
student ngày càng khó khớp stored logits cũ. Vì `logit_refresh=false`, stored
logits cũng không được cập nhật khi representation/head thay đổi. Số liệu này
phù hợp với logit drift nhưng chưa tự nó chứng minh quan hệ nhân quả.

## 8. Nhận xét và quyết định

### Kết luận chính

1. Square-root replay vẫn chưa giữ được hiệu năng dài hạn: final test Macro-F1
   chỉ `0.459732` ở member 0.
2. Task 6→7 giảm Macro-F1 không phải vì task 7 có nhiều mẫu hơn: task 7 có khối
   lượng gần bằng task 0/1/5/6 và nhỏ hơn task 2/3/4.
3. Task 7 được học tốt trên chính dữ liệu task 7, nhưng một số logits task 7 hút
   lượng lớn mẫu old. Precision giảm trong khi recall tăng là biểu hiện điển
   hình của classifier-boundary shift.
4. Buffer không rỗng và không mất class. Tăng exposure đồng loạt chưa chắc xử
   lý được vấn đề; exposure hầu như không tương quan với forgetting drop trong
   run này.
5. Đây chỉ là một seed/member và không có ensemble, nên kết quả đủ để bác bỏ
   kỳ vọng “sqrt replay tự nó giải quyết vấn đề”, nhưng chưa đủ để ước lượng
   mean/variance hay chốt kết quả full method.

### Hướng thí nghiệm hợp lý tiếp theo

- Không nên chạy full ba member của đúng config này chỉ để hy vọng ensemble kéo
  từ `0.4597` lên mức cạnh tranh; lỗi systematic old→new quá lớn.
- Giữ P4, buffer và seed cố định; ablation tiếp theo nên tác động trực tiếp vào
  calibration/decision boundary, ví dụ post-task balanced classifier tuning hoặc
  logit adjustment trên validation, thay vì chỉ tăng replay exposure.
- Nếu tiếp tục DER++, nên kiểm tra riêng `alpha/beta`, logit refresh hoặc cách
  lưu logits bằng pilot member 0; mỗi lần chỉ thay một yếu tố.
- Báo cáo run sau phải tiếp tục giữ: global class precision/F1, old→new
  confusion, task-origin retention và loss audit. Chỉ nhìn current-task F1 sẽ
  che giấu false positive trên old classes.

## 9. Phạm vi diễn giải

- Đây là pilot `member_0`, không phải full three-member run.
- Không có offline probability ensemble trong bundle.
- Không dùng test để chọn hyperparameter; các chẩn đoán test ở đây chỉ mô tả
  run đã hoàn tất. Quyết định tuning phải ưu tiên validation.
- Bundle không chứa NPZ/checkpoint theo chủ ý; các kết luận dựa trên metric,
  classwise CSV, confusion, buffer/replay audit và loss audit đã đóng gói.

## 10. Nguồn số liệu trong bundle

- Config và provenance: `config/p4_derpp_cb_sqrt_replay_pilot.json`,
  `run/member_0/run_config.json`.
- Metric theo task: `run/member_0/metrics.csv` và
  `run/member_0/task_*/metrics.csv`.
- Old/current split: `run/pilot_results/task7_old_current_metrics.csv`.
- Buffer và replay: `run/pilot_results/buffer_replay_by_source_task.csv`,
  `run/member_0/task_*/buffer_audit.json`, `replay_exposure.csv`.
- Classwise/forgetting/confusion: các CSV trong
  `run/visualizations_member0/task_6_*` và `task_7_*`.
- Loss audit: `run/member_0/task_*/training_audit.json`.
