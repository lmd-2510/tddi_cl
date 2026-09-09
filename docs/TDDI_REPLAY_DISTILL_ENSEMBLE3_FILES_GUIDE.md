# Bản đồ code đầy đủ — T-DDI Ensemble3 Replay-Distill P3

Tài liệu này trả lời bốn câu hỏi:

1. Study hiện tại thực sự dùng những file nào?
2. Mỗi file giữ trách nhiệm gì và gọi file nào khác?
3. Những file có tên EWC, GEM, TabM, S02… có tham gia study này không?
4. Dữ liệu đi qua pipeline như thế nào và từng metric có ý nghĩa gì?

## 1. Study đang được khóa như thế nào?

```text
method          = replay_distill_fixed_budget_uniform
backbone        = tddi_paper_member
input           = 3.780 QSAR descriptors
architecture    = LayerNorm(3780) -> 7560 -> 7560 -> expandable head C_t
protocol        = P3 / tail_to_head
experiment seed = 0
member IDs      = 0, 1, 2
tasks           = 8
class layout    = [38, 20, 20, 20, 20, 20, 20, 20]
total classes   = 178
memory budget   = 6.800 unique exemplars
replay exposure = 6.800 draws/epoch từ task 1
```

Tên đầy đủ của thí nghiệm có thể đọc là:

> Ba trajectory CIL độc lập dùng cùng protocol P3 và cùng replay memory identity,
> nhưng có initialization/dropout/sampler order riêng; sau đó trung bình xác suất
> của ba member để dự đoán và tính uncertainty estimation (UE).

`tddi_paper_member` trong repo là **paper-size numerical CIL member** đã được thiết kế
cho study này. Nó không tuyên bố tái tạo toàn bộ training recipe nguyên bản của paper
T-DDI. Các baseline MLP `small/base/large` không phải backbone của study này.

## 2. Chú giải mức độ liên quan

| Nhãn | Ý nghĩa |
|---|---|
| **CORE** | Được gọi trực tiếp khi train hoặc ensemble study hiện tại. |
| **SHARED** | Study dùng một phần chức năng của file; file cũng phục vụ method khác. |
| **OFFLINE** | Chỉ chạy sau training, không cập nhật weight model. |
| **PREP** | Chỉ dùng để tạo/kiểm tra dữ liệu và frozen asset. |
| **TEST** | Test synthetic/unit; không train dataset thật. |
| **OTHER** | Code tương thích cho study/method khác, không tham gia kết quả P3 hiện tại. |

Một file được giữ trong repo không có nghĩa là nó đã được dùng trong run. Nguồn sự
thật của một run là `run_config.json`, `study_manifest.json`, command trong log và các
hash được lưu cùng artifact.

## 3. Sơ đồ phụ thuộc ngắn gọn

```text
config study + frozen assets + parquet data
                    |
                    v
       tddi_ensemble3_study.py
           |       |       |
           v       v       v        chạy tuần tự, không đồng thời
        member_0 member_1 member_2
           \       |       /
            \      |      /
             train_cil.py
               |-- ddi_dataset.py + class_mapping.py
               |-- tddi_paper_member.py
               |-- fixed_budget_replay.py
               |-- replay_checkpoint.py
               |-- cil_evaluation.py + classification/classwise/continual metrics
               `-- member_predictions.py
                         |
                         v
                 offline_ensemble.py
                    /      |       \
                   v       v        v
             UE audit   threshold  temperature calibration (optional)
                   \       |        /
                    v      v       v
                   build_final_report.py
```

## 4. Config, asset và tài liệu ở root

### 4.1 Config điều khiển study

#### `configs/tddi_ensemble3_replay_distill_p3_seed0.json` — **CORE**

Đây là contract chính của cả thí nghiệm. Nó khóa:

- method `replay_distill_fixed_budget_uniform`;
- backbone `tddi_paper_member`;
- protocol P3 và task-file tail-to-head;
- ba member 0, 1, 2 và experiment seed 0;
- đường dẫn train/validation/test, feature list và scaler;
- microbatch 64, effective batch 1024, gradient accumulation 16;
- tối đa 20 epoch/task, patience 5;
- AdamW, learning rate `0.001`, weight decay `0.0001`;
- Focal gamma `1.0`;
- output distillation alpha `1.0`, temperature `2.0`;
- feature distillation weight `0.5`;
- memory 6.800 và replay draws 6.800/epoch;
- export prediction ở validation và test;
- task-boundary resume;
- output root riêng cho full P3.

Orchestrator đọc file này và chuyển các trường thành CLI cho `train_cil.py`. Sửa
hyperparameter ở đây mới là cách đúng để thay đổi toàn bộ study; không nên sửa default
rải rác trong mã nguồn.

#### `configs/tddi_ensemble_confidence_threshold_full_p3_primary.json` — **OFFLINE**

Config threshold chính:

- score: `entropy_confidence`;
- probability source: `raw`;
- grid đặt trước: `0.0, 0.1, ..., 0.9, 0.95`;
- chỉ nhận candidate có coverage ít nhất `0.5`;
- chọn Macro-F1 cao nhất;
- hòa thì ưu tiên accuracy, coverage, rồi threshold thấp hơn;
- ECE dùng 15 bin.

Điểm quan trọng: `entropy_confidence >= 0.9` **không** có nghĩa là softmax probability
`>= 0.9`. Hai đại lượng này khác nhau.

#### `configs/tddi_ensemble_confidence_threshold_full_p3_low_coverage_sensitivity.json` — **OFFLINE**

Giống config primary nhưng cho coverage tối thiểu `0.1`. Đây chỉ là sensitivity
analysis để xem đổi coverage constraint ảnh hưởng ra sao. Không dùng làm kết quả chính
nếu protocol báo cáo đã chọn primary coverage `0.5`.

#### `configs/tddi_ensemble_temperature_calibration_full_p3.json` — **OFFLINE, tùy chọn**

Đặt quy tắc fit một scalar temperature trên validation bằng NLL. Nhiệt độ bị chặn
trong `[0.05, 100]`, tối đa 50 vòng tối ưu.

Primary threshold config hiện ghi `probability_source = raw`, vì vậy calibration là
một nhánh phân tích riêng. Nó không âm thầm thay xác suất hay metric primary.

### 4.2 Frozen study assets

#### `study_assets/task_protocols/tail_to_head_tasks.json` — **CORE**

Định nghĩa P3: tám task, class nào xuất hiện ở task nào và thứ tự tail-to-head. Đây là
nguồn quyết định `current classes`, `old classes`, `seen classes`, head size và class
map ở mỗi task. Orchestrator kiểm tra đủ 178 class và layout
`[38,20,20,20,20,20,20,20]`, đồng thời lưu SHA256 để chống dùng nhầm task-file.

#### `study_assets/data_schema/feature_columns.json` — **CORE**

Danh sách chính xác và có thứ tự của 3.780 cột QSAR. Thứ tự này phải khớp scaler và
input model; đổi thứ tự feature mà không fit lại scaler sẽ làm input sai nghĩa.

#### `study_assets/preprocessing/scaler.pkl` — **CORE**

Scaler đã fit trên train. `ddi_dataset.py` load artifact này để impute/scale feature.
Validation và test chỉ được transform bằng scaler train, không được fit lại.

#### Các file provenance/audit — **SHARED/PREP**

| File | Tác dụng |
|---|---|
| `study_assets/data_schema/global_class_map.json` | Class map toàn cục đã audit; không thay cho expandable seen-class map lúc train. |
| `study_assets/data_schema/schema_summary.json` | Tóm tắt schema dữ liệu nguồn. |
| `study_assets/data_schema/meta_columns_check.json` | Kết quả kiểm tra các cột metadata. |
| `study_assets/data_schema/audit_summary.md` | Diễn giải audit schema cho người đọc. |
| `study_assets/preprocessing/feature_columns.json` | Bản feature list đi cùng preprocessing artifact để đối chiếu. |
| `study_assets/preprocessing/meta_columns.json` | Metadata columns không đưa vào numerical input. |
| `study_assets/preprocessing/scaler_config.json` | Cấu hình tạo scaler. |
| `study_assets/preprocessing/preprocessing_report.md` | Báo cáo fit preprocessing. |

Ba input lớn `train_extracted.parquet`, `validation_extracted.parquet` và
`test_extracted.parquet` nằm tại root trên máy train nhưng thường không commit vào Git.

### 4.3 Tài liệu

| File | Vai trò |
|---|---|
| `README.md` | Điểm vào của repo, mô tả study và các command chính. |
| `docs/TDDI_PAPER_REPLAY_DISTILL_P3_8TASK_GPU_RUNBOOK.md` | Runbook thực thi trên máy GPU: preflight, test, dry-run, nohup, PID/log, resume, ensemble và post-processing. |
| `docs/TDDI_ENSEMBLE3_REPLAY_DISTILL_P3_RESULTS.md` | Kết quả đã thu được và cách đọc bảng. |
| `docs/TDDI_REPLAY_DISTILL_ENSEMBLE3_FILES_GUIDE.md` | Chính tài liệu kiến trúc/pipeline hiện tại. |

## 5. Scripts chuẩn bị dữ liệu

Các script dưới đây là **PREP**. Chúng không được tự gọi trong một run full khi frozen
asset đã có.

| File | Dùng khi nào | Kết quả chính |
|---|---|---|
| `scripts/convert_csv_to_parquet.py` | Có CSV rất lớn và muốn đọc hiệu quả hơn. | Ba file Parquet. |
| `scripts/inspect_splits.py` | Audit cột, kiểu dữ liệu và đồng nhất schema giữa split. | Feature/schema audit. |
| `scripts/preprocess_features.py` | Cần tái tạo preprocessing từ train. | Scaler và preprocessing metadata. |
| `scripts/analyze_class_distribution.py` | Cần thống kê số mẫu của 178 class. | Class-count/distribution artifacts. |
| `scripts/build_cil_tasks.py` | Cần tái tạo protocol/task schedule. | Task JSON, bao gồm tail-to-head. |
| `scripts/check_leakage.py` | Cần kiểm tra drug-pair overlap giữa train/validation/test. | Leakage report. |

Nếu chỉ pull code lên server đã có đúng Parquet và `study_assets`, không cần chạy lại
sáu script này.

## 6. Các file model, data và seed trực tiếp tham gia training

### 6.1 `src/models/tddi_paper_member.py` — **CORE**

Định nghĩa đúng backbone study:

```text
x [B, 3780]
  -> LayerNorm(3780)
  -> Linear(3780, 7560)
  -> GELU
  -> Dropout(0.2)
  -> Linear(7560, 7560)
  -> GELU
  -> Dropout(0.2)
  -> latent [B, 7560]
  -> Linear(7560, C_t)
  -> logits [B, C_t]
```

Các interface:

- `encode(x)`: trả latent 7.560 chiều;
- `classify(latent)`: latent sang logits;
- `forward_with_latent(x)`: trả cả logits và latent, cần cho feature distillation và export;
- `forward(x)`: forward thông thường;
- `manifest()`: ghi kiến trúc vào run config/manifest mà không cần cấp phát model lớn.

`C_t` là số class đã thấy tới task `t`, không cố định 178 từ đầu.

### 6.2 `src/data/ddi_dataset.py` — **CORE**

Đọc dữ liệu theo từng split và class-filter. Nó chịu trách nhiệm:

- load ordered feature columns và scaler payload;
- đọc Parquet theo class của task hoặc all-seen evaluation;
- áp dụng imputation/scaling đã fit từ train;
- trả `float32` feature và `int64` raw labels;
- tùy chọn trả drug-pair metadata để tạo stable sample ID;
- đếm số train samples theo class.

Numerical member chỉ sử dụng 3.780 feature. Drug name, SMILES và ID là metadata, không
được ghép vào tensor đầu vào.

### 6.3 `src/data/class_mapping.py` — **CORE**

Raw class ID không được giả định là `0..C-1`. File này:

- tạo `raw_class_id -> dense head row` cho các class đã thấy;
- remap label raw sang dense index để tính loss;
- invert map để chuyển prediction về raw ID;
- validate map dày đặc và không trùng;
- lưu `seen_class_map_task_N.json`.

Khi head mở rộng, code copy row weight/bias theo **raw class ID**, không theo vị trí cũ
một cách mù quáng.

### 6.4 `src/utils/seed.py` — **CORE**

Tách hai loại seed:

- `experiment_seed`: protocol, task/class metadata, preprocessing và artifact dùng chung;
- `member_seed`: initialization, dropout và sampler/DataLoader order riêng.

`member_seed` được suy ra deterministic bằng NumPy `SeedSequence` từ
`experiment_seed + member_id + namespace constants`. Nếu command cũ không có
`--member-id`, `member_seed = experiment_seed`, giữ backward compatibility.

### 6.5 `src/data/fixed_budget_replay.py` — **CORE**

File này chứa hai khái niệm khác nhau cần phân biệt.

#### `FixedBudgetReplayBuffer`: mẫu nào được giữ

- Tổng số unique exemplar tối đa là 6.800.
- Budget được chia max-min gần đều giữa toàn bộ class đã thấy, nhưng không cấp quá số
  mẫu class đang có.
- Với class mới, mẫu được xếp theo khoảng cách tới class mean trong feature space;
  mẫu gần mean nhất được giữ, tie-break theo source row order.
- Với class cũ, khi số class tăng thì quota giảm và buffer cắt bớt phần cuối; không
  bốc lại identity ngẫu nhiên.
- Vì ba member dùng cùng data/task/preprocessing và cùng deterministic ranking, chúng
  có cùng exemplar identity/counts.

`random_seed=experiment_seed` được lưu như provenance/validation của buffer. Việc rank
hiện tại là deterministic class-mean ranking, không phải random sampling.

#### `FixedReplaySampler`: exemplar được trình bày theo thứ tự nào

Mỗi epoch:

1. mọi current-task sample được draw đúng một lần;
2. từ task 1, draw đúng 6.800 replay exposures;
3. replay draw được chia gần đều theo class cũ;
4. nếu quota draw lớn hơn unique exemplar của class, sampler đi theo nhiều chu kỳ
   without-replacement rồi mới lặp;
5. current và replay indices được trộn lại;
6. RNG là `SeedSequence([member_seed, task_id, epoch])`.

Do đó `memory_count` là số mẫu độc nhất đang lưu, còn `replay_draws_per_epoch` là số
lần trình bày mẫu cho optimizer. Hai số này không có cùng nghĩa.

### 6.6 `src/methods/replay.py` — **SHARED**

Chỉ cung cấp helper nối current arrays với replay arrays. Fixed-budget method dùng
helper này để tạo một `TensorDataset`, sau đó `FixedReplaySampler` quyết định index nào
được draw. Đây là code dùng chung, không biến current method thành method `replay` cũ.

### 6.7 `src/utils/logging.py` — **SHARED**

Tạo output directory và các đường dẫn chuẩn, ghi log, event và đường dẫn cho
`run_config.json`, `metrics.csv`, `training_audit.csv`, `run_summary.md`.

## 7. Training engine và loss

### 7.1 `src/training/train_cil.py` — **CORE trung tâm**

Đây là file quan trọng nhất cho một member. Method fixed-budget replay-distill được
triển khai trực tiếp qua training loop tại đây; loss, teacher và task loop đều nằm
trong `train_cil.py`.

Các nhóm chức năng quan trọng:

- `parse_args()`: data, method, backbone, seed, replay, distillation và resume;
- `expand_model_for_seen_classes()`: dựng model/head mới;
- `copy_previous_state_to_expanded_model()`: copy backbone và old-class head rows;
- `train_one_epoch()`: Focal + output distillation + feature distillation;
- `export_member_evaluation()`: ghi member prediction artifact;
- `build_replay_progress()/restore_replay_progress()`: state giữa task;
- `write_run_summary()`: tóm tắt cuối member;
- `main()`: toàn bộ vòng lặp 8 task.

#### Loss tại task 0

```text
L_total = L_focal
L_focal = mean[-(1 - p_y)^gamma * log(p_y)]
gamma   = 1.0
```

#### Loss từ task 1 trở đi

Teacher là model tốt nhất của task trước, được freeze. Student có head lớn hơn:

```text
L_total = L_focal
        + alpha * T^2 * KL(
              softmax(teacher_old_logits / T),
              softmax(student_old_logits / T)
          )
        + beta * MSE(student_latent, teacher_latent)

alpha = 1.0
T     = 2.0
beta  = 0.5
```

Chỉ các cột old class được dùng cho output distillation; mapping dựa trên raw class
ID. Feature distillation so sánh latent 7.560 chiều. Focal loss học nhãn thật của cả
current và replay sample.

#### Optimizer, validation và early stopping

- AdamW; microbatch 64; accumulation 16; effective batch 1024.
- Nhóm accumulation cuối được scale theo số sample thật.
- Sau mỗi epoch, validation trên **toàn bộ class đã thấy**.
- Best checkpoint là epoch có validation Macro-F1 cao nhất.
- `patience=5`; model load lại best state trước export/test.

### 7.2 `src/training/replay_checkpoint.py` — **CORE**

Checkpoint ở ranh giới task lưu schema/version, completed/next task, best model,
class map/order, fixed buffer, progress/metrics/audit, seed/RNG, task-file hash và
training contract. Nó không lưu teacher trùng lặp; resume dựng teacher từ best model
state. Save atomic và fail khi method/seed/hash/map/backbone/budget không khớp.

### 7.3 `src/training/ewc_checkpoint.py` — **SHARED một phần**

`replay_checkpoint.py` tái sử dụng helper RNG tổng quát từ file này. Phần
Fisher/theta-star/EWC checkpoint không dùng trong replay-distill study. Không nên kết
luận study dùng EWC chỉ vì file này xuất hiện trong import graph.

### 7.4 `src/training/tddi_ensemble3_study.py` — **CORE điều phối**

Orchestrator:

1. đọc/validate config;
2. tính member seed;
3. xác định member `fresh`, `resume` hay `complete`;
4. sinh command `train_cil.py`;
5. chạy tuần tự 0 → 1 → 2, không giữ hai model trên GPU;
6. skip complete, resume checkpoint hợp lệ;
7. fail nếu output dở mà không có checkpoint;
8. chỉ ensemble khi đủ ba member;
9. ghi `study_manifest.json` và hash liên kết artifact.

Không có `--execute` thì chỉ dry-run, không train.

## 8. Expandable head hoạt động ra sao?

Ở task `t`:

1. lấy union class từ task 0 tới task `t`;
2. sort raw class IDs và tạo dense map `0..C_t-1`;
3. tạo model mới với head `Linear(7560, C_t)`;
4. copy toàn bộ backbone state từ model trước;
5. copy old-class weight/bias theo raw ID;
6. new-class rows giữ initialization mới;
7. train student trên current + replay.

Ví dụ old raw classes `[10, 50]`, task mới thêm `3`. New order là `[3, 10, 50]`.
Class `10` chuyển row 0 → 1 nhưng vẫn giữ đúng weight. Vì vậy prediction artifact luôn
phải mang `raw_class_ids` đúng thứ tự cột.

## 9. Evaluation trong từng member

### 9.1 `src/eval/cil_evaluation.py` — **CORE**

Chạy eval/no-grad, tính Focal evaluation loss, softmax logits, aggregate metrics,
per-class metrics và tùy chọn thu logits/probabilities/latent. Nó bảo toàn Torch RNG
state quanh evaluation để evaluation không làm lệch trajectory training.

### 9.2 `src/eval/classification_metrics.py` — **CORE/SHARED**

Tính Accuracy, Macro-F1, Weighted-F1 và Balanced Accuracy trên explicit class set.

### 9.3 `src/eval/classwise_metrics.py` — **CORE**

Tính precision/recall/F1, mean max-probability confidence và entropy cho từng class sau
mỗi task; lưu class trajectory và class forgetting.

### 9.4 `src/eval/continual_metrics.py` — **CORE**

Quản lý result matrix `R[i,j]`, trong đó `i` là model sau task `i`, `j` là test group
của task `j`, giá trị là Macro-F1. File này tính task-group forgetting.

### 9.5 `src/eval/member_predictions.py` — **CORE**

Mỗi member/task/split xuất `.npz` chứa schema, provenance, task/split/member/seeds,
stable sample IDs, labels, logits, probabilities và raw class-column order. Loader
chặn duplicate ID, shape/width sai, non-finite, probability không tổng bằng 1 hoặc
không khớp `softmax(logits)`. Ghi atomic và không ghi đè.

### 9.6 `src/eval/s02_artifacts.py` — **SHARED một phần**

Current member export chỉ tái sử dụng tên cột drug ID và
`build_stable_sample_ids()` tạo ID `drugA|drugB`. Phần S02 export cũ chỉ chạy nếu bật
`--export-s02`; config P3 hiện không bật.

## 10. Offline ensemble và UE

### 10.1 `src/eval/offline_ensemble.py` — **CORE sau training**

Input là đúng ba prediction artifact cùng task/split. Nó bắt buộc cùng
method/protocol/experiment seed/task/split/sample IDs/labels/raw class order và member
IDs khác nhau. Sau đó:

```text
p_bar(y|x) = (p_0(y|x) + p_1(y|x) + p_2(y|x)) / 3
prediction = raw_class_ids[argmax(p_bar)]
```

Repo **mean probabilities**, không mean raw logits.

Gọi `M=3`, `C=C_t`, `p_m` là member probability:

| Field | Công thức/ý nghĩa | Cách đọc |
|---|---|---|
| `predictive_entropy` | `H(p_bar) = -Σ p_bar log p_bar` | Cao = tổng bất định cao. |
| `expected_member_entropy` | `(1/M)Σ H(p_m)` | Cao = từng member cũng mơ hồ. |
| `mutual_information` | `H(p_bar) - mean H(p_m)` | Cao = member bất đồng phân phối hơn. |
| `normalized_entropy` | `H(p_bar)/log(C)`; `C<=1` trả 0 | 0 chắc hơn, 1 gần uniform. |
| `normalized_mi` | `MI/log(C)` | Legacy normalization theo số class. |
| `member_normalized_mi` | `MI/log(M)`; `M<=1` trả 0 | Chuẩn hóa theo ensemble size. |
| `mean_probability_variance` | Mean theo class của variance giữa member | Cao = probability vector khác hơn. |
| `total_probability_variance` | Tổng variance theo class | Bằng `C × mean variance` với cùng C. |
| `pairwise_disagreement` | Tỷ lệ ba cặp member có argmax khác nhau | 0 = cùng dự đoán; 1 = mọi cặp khác. |
| `max_probability` | `max(p_bar)` | Max softmax probability thật. |
| `entropy_confidence` | `1 - normalized_entropy` | Cao = entropy thấp. |
| `confidence` | Alias cũ của `entropy_confidence` | **Không phải** max probability. |

MI cao thường phản ánh epistemic disagreement, nhưng không tự động chứng minh model
đúng. Diversity/UE phải được kiểm tra bằng khả năng phát hiện prediction sai.

### 10.2 `src/eval/offline_ue_audit.py` — **OFFLINE**

Với từng uncertainty score, báo cáo mean/std/quantile, mean trên đúng/sai, AUROC/AUPRC
phát hiện lỗi, risk-coverage, AURC và metric tại coverage 25/50/75/90/100%. Nó còn chia:

- old classes và current-task classes;
- `ultra_tail <=20`, `tail 21..100`, `medium 101..1000`, `head >1000`, dựa chỉ trên
  train counts.

Nếu split toàn đúng hoặc toàn sai, AUROC/AUPRC được ghi undefined. Test chỉ dùng để
report, không chọn score/hyperparameter.

## 11. Threshold và calibration

### 11.1 `src/eval/confidence_threshold.py` — **OFFLINE**

Luồng đúng:

1. validation ensemble → thử toàn bộ grid → freeze threshold;
2. test ensemble + frozen threshold → report, không chọn lại.

Primary dùng `entropy_confidence`, coverage tối thiểu 0.5, tối đa Macro-F1 rồi tie-break
accuracy/coverage/lower threshold. Frozen artifact lưu grid đầy đủ, candidate results,
metrics, hashes, class order, seed, score name và probability source.

`coverage = selected_count / total_count`. Selective metric phải luôn đi cùng coverage.

### 11.2 `src/eval/calibration_metrics.py` — **SHARED/OFFLINE**

Cung cấp ECE theo max-probability bins, multiclass Brier, NLL, softmax/temperature và
raw-label mapping. Threshold report và temperature calibration đều dùng file này.

### 11.3 `src/eval/offline_temperature_calibration.py` — **OFFLINE, tùy chọn**

Fit trên validation:

```text
calibrated_p = softmax(log(mean_probability) / T)
```

Không mean logits, không sửa raw artifact, test chỉ load frozen T. Argmax/accuracy phải
bất biến; raw và calibrated ECE/NLL/Brier được báo cáo riêng. Primary config hiện dùng
`raw`, nên calibration không tự động đi vào primary threshold.

## 12. Final report

### `src/eval/build_final_report.py` — **OFFLINE**

Không train/inference. Nó đọc ensemble test task 0..7, threshold test reports, member
forgetting artifacts và task-file; sau đó tạo:

- `ensemble3_p3_task_summary.csv`;
- `ensemble3_p3_paper_table.csv`;
- `ensemble3_p3_task_matrix.csv`;
- `ensemble3_p3_forgetting_summary.csv`;
- `ensemble3_p3_member_forgetting_summary.csv`;
- `ensemble3_p3_diversity_summary.csv`;
- `ensemble3_p3_final_report.md`.

`Mean across tasks` là mean tám stage, mỗi stage trọng số bằng nhau. Đây là summary phụ
về trajectory, không thay thế kết quả cuối `task 7 / seen_all`.

## 13. Pipeline theo thứ tự thời gian

### A — Chuẩn bị đầu vào

CSV → Parquet; audit split/schema/leakage; fit scaler chỉ trên train; đóng băng 3.780
feature columns và P3 task schedule.

### B — Dry-run

Orchestrator kiểm tra config/hash/output và in command. Không có `--execute` nên chưa
train.

### C — Train member 0

Task 0 dùng Focal/current data. Task 1..7 mở head thêm 20 class, dùng current + fixed
replay + frozen teacher distillation. Mỗi task chọn best validation all-seen Macro-F1,
export validation/test, update memory, test per-group/seen-all và save checkpoint.

### D — Train member 1 và 2

Lặp cùng protocol/data/class map/exemplar identity, nhưng member seed khác nên model,
dropout và sampler order khác.

### E — Offline ensemble

Mỗi task/split: load ba `.npz` → validate alignment → mean probabilities → prediction
+ UE → ensemble `.npz` → manifest.

### F — Threshold primary

Validation chọn/freeze threshold; test chỉ load frozen artifact. Không đổi threshold
sau khi nhìn test.

### G — UE audit

Đo error-detection AUROC/AUPRC và AURC để xác nhận uncertainty ranking có ích.

### H — Calibration tùy chọn

Validation fit T, test apply frozen T. Đây là nhánh riêng nếu primary vẫn dùng raw.

### I — Final report

Tập hợp classification, calibration, threshold, forgetting và diversity thành CSV/MD.

## 14. Cây output và artifact

```text
outputs/full/tddi_ensemble3_replay_distill_p3_seed0_8tasks_v1/
|-- member_0/
|   |-- run_config.json
|   |-- stdout.log, train.log, events.csv
|   |-- checkpoints/task_N_model.pt
|   |-- checkpoints/latest_replay_distill_state.pt
|   |-- memory/memory_summary.csv
|   |-- memory/memory_after_task_N.parquet
|   |-- member_predictions/task_N/{validation,test}.npz
|   |-- seen_class_map_task_N.json
|   |-- training_audit.csv, replay_budget_audit.csv
|   |-- metrics.csv, task_matrix.csv, forgetting.csv
|   |-- class_trajectory.csv, class_forgetting.csv
|   `-- run_summary.md
|-- member_1/...
|-- member_2/...
|-- offline_ensemble/task_N/{validation,test}.npz
|-- study_manifest.json
`-- final_results/...
```

| Artifact | Tác dụng |
|---|---|
| `run_config.json` | Chứng minh command/config/seed/backbone thực tế. |
| `stdout.log`, `train.log` | Theo dõi tiến độ/lỗi. |
| `events.csv` | Event có cấu trúc. |
| `task_N_model.pt` | Best model riêng task N. |
| `latest_replay_distill_state.pt` | Resume toàn trajectory ở task boundary. |
| `memory_summary.csv` | Exemplar count theo class. |
| `memory_after_task_N.parquet` | Snapshot feature/label buffer; file lớn. |
| `member_predictions/*.npz` | Nguồn bắt buộc cho ensemble. |
| `seen_class_map_task_N.json` | Raw class ↔ head row. |
| `training_audit.csv` | Data/replay/sampler/epoch/optimizer/seed audit. |
| `replay_budget_audit.csv` | Budget, draws và unique replay theo class/task. |
| `metrics.csv` | Per-group và seen-all metrics. |
| `task_matrix.csv`, `forgetting.csv` | Macro-F1 trajectory và task forgetting. |
| `class_trajectory.csv`, `class_forgetting.csv` | Per-class trajectory/forgetting. |
| `offline_ensemble/*.npz` | Mean probability, ensemble prediction và UE. |
| `study_manifest.json` | Run IDs, member seeds, hashes và links. |

## 15. Classification metrics

### Accuracy

`Accuracy = số dự đoán đúng / tổng mẫu`. Dễ hiểu nhưng bị class đông chi phối.

### Precision, Recall và F1 từng class

```text
Precision_c = TP_c / (TP_c + FP_c)
Recall_c    = TP_c / (TP_c + FN_c)
F1_c        = 2 * Precision_c * Recall_c / (Precision_c + Recall_c)
```

Precision hỏi “những gì model gọi là class c có đúng không”; recall hỏi “model tìm
được bao nhiêu mẫu thật của class c”; F1 cân bằng hai mặt.

### Macro-F1

`Macro-F1 = mean_c(F1_c)`. Mỗi class ngang nhau; phù hợp dữ liệu 178 class mất cân bằng
và là metric early stopping hiện tại.

### Weighted-F1

`Weighted-F1 = Σ support_c × F1_c / N`. Class đông có trọng số lớn, nên có thể che lỗi
tail class.

### Balanced Accuracy

Trong repo: `mean_c(Recall_c)`, tức macro recall. Mỗi class ngang nhau; cao là model
bao phủ đều class tốt hơn.

### Evaluation loss

Là mean Focal Loss trên split. Dùng theo dõi optimization, không thay Macro-F1. Loss
giữa task có head size khác nhau không nên so đơn giản như cùng một bài toán.

## 16. Continual-learning metrics

### Result matrix và per-task result

`R[i,j]` là Macro-F1 trên test group task `j` sau khi học task `i`. Đây là per-task CIL
trajectory; chỉ các ô `j <= i` có nghĩa.

### Seen-all

Sau task `i`, ghép test samples của mọi class đã thấy `0..i`. `task 7 / seen_all` là
model cuối đánh giá đủ 178 class và là kết quả cuối cần nhấn mạnh.

### Task forgetting

```text
Forgetting_j = max_{i>=j} R[i,j] - R[T,j]
```

Thấp hơn tốt hơn. `mean_old_tasks` mean task 0..6 ở stage cuối, không tính task 7 vừa
học; nó không phải mean Accuracy của tám task.

### Class forgetting

`best historical class F1 - current class F1`, giúp tìm class cụ thể bị mất.

### Mean across tasks/stages

Mean tám stage là trajectory summary phụ. Nó hợp lệ nếu ghi nhãn đúng, nhưng không được
trình bày thay cho final task 7/seen-all.

## 17. Calibration, threshold và UE quality metrics

### ECE

Weighted mean theo max-probability bin của
`|accuracy_in_bin - mean_confidence_in_bin|`. Thấp hơn tốt hơn. Confidence calibration
ở đây là max probability, không phải entropy confidence.

### NLL

`mean[-log(p_true)]`; phạt mạnh dự đoán sai nhưng quá tự tin. Thấp hơn tốt hơn.

### Multiclass Brier

`mean Σ_c (p_c - one_hot(y)_c)^2`; đo toàn probability vector. Thấp hơn tốt hơn.

### Coverage và selective risk

`Coverage = selected / total`; `Risk = error rate trên selected set`. Selective metric
cao với coverage thấp không đồng nghĩa model tốt trên toàn bộ dữ liệu.

### AURC

Area Under Risk-Coverage curve. Giữ sample uncertainty thấp trước; AURC thấp hơn tốt
hơn và đánh giá toàn ranking thay vì một threshold.

### Error-detection AUROC

Prediction sai là positive. `0.5` gần random, càng gần `1` thì uncertainty càng xếp
mẫu sai cao hơn mẫu đúng.

### Error-detection AUPRC

Cũng xem error là positive nhưng phụ thuộc error prevalence. Baseline gần error rate,
không cố định 0.5; phải đọc cùng error/correct count.

## 18. File của method/eval khác

### 18.1 Method khác — **OTHER**

| File | Method | Có tham gia P3? |
|---|---|---|
| `src/methods/ewc.py` | EWC Fisher + penalty | Không; chỉ import vì CLI dùng chung. |
| `src/methods/gem.py` | GEM memory/projection | Không. |
| `src/methods/agem.py` | A-GEM projection | Không. |

`src/methods/replay.py` là ngoại lệ SHARED như đã giải thích.

Các method legacy `sequential`, `joint_seen` và `replay_distill` đã được gỡ khỏi CLI
và xóa module để repo chỉ còn các lựa chọn còn được duy trì. Việc này không xóa hay đổi
method chính `replay_distill_fixed_budget_uniform`.

### 18.2 Data/model khác — **OTHER**

| File | Vai trò khác | Có tham gia P3 numerical study? |
|---|---|---|
| `src/models/mlp.py` | Baseline `small/base/large` | Không. |
| `src/models/tabm_classifier.py` | TabM | Không. |
| `src/models/ddi_gcn.py` | Graph backbone | Không. |
| `src/data/backbone_inputs.py` | Graph/pair-index input | Không. |
| `src/data/molecular_graphs.py` | Graph bank/cache | Không. |
| `src/data/replay_buffer.py` | Replay buffer per-class cũ | Không. |
| `src/training/train_static.py` | Static baseline training | Không. |

Import không đồng nghĩa thực thi; `--method` và `--variant` quyết định nhánh runtime.

### 18.3 Evaluation khác

| File | Vai trò | Quan hệ current study |
|---|---|---|
| `src/eval/run_s03_calibration.py` | Calibration pipeline S02 cũ | Không dùng; current dùng offline temperature module. |
| `src/eval/run_s04_aggregation.py` | Aggregate run/seed study cũ | Không tạo Ensemble3 P3 report. |
| `src/eval/rare_class_metrics.py` | Rare-class helper generic | Không dùng; UE audit có grouping riêng. |
| `src/eval/s02_artifacts.py` | S02 export cũ | Chỉ dùng stable-ID helper. |
| `src/eval/calibration_metrics.py` | Công thức calibration dùng chung | Có dùng. |

Các `src/**/__init__.py` đánh dấu Python package; chúng không tự chạy model/metric.

## 19. Tests bảo vệ phần nào?

### Current study tests — **TEST**

| Test | Bảo vệ |
|---|---|
| `tests/test_tddi_paper_member.py` | Model shape/LayerNorm/latent/head expansion. |
| `tests/test_ensemble_seed.py` | Member seeds và shared invariant. |
| `tests/test_s04_fixed_budget.py` | Allocation/draw/budget audit. |
| `tests/test_replay_checkpoint.py` | Replay save/resume synthetic equivalence. |
| `tests/test_tddi_ensemble3_replay_study.py` | P3 config, dry-run, sequential/skip/resume/ensemble gate. |
| `tests/test_member_prediction_export.py` | Prediction schema/order/round-trip. |
| `tests/test_offline_ensemble.py` | Mean probability và UE formulas. |
| `tests/test_offline_ue_audit.py` | Error detection/selective/rareness/P3 guard. |
| `tests/test_confidence_threshold.py` | Validation-only threshold và semantics. |
| `tests/test_offline_temperature_calibration.py` | Fit/freeze/load/argmax invariance. |
| `tests/test_build_final_report.py` | Tạo final CSV/MD. |
| `tests/test_classification_metrics.py` | Aggregate metric formulas. |
| `tests/test_classwise_evaluation.py` | Per-class trajectory/forgetting. |
| `tests/test_task_protocol_directions.py` | Protocol direction/layout. |

Chúng dùng synthetic data nhỏ; pytest chạy nhanh không phải full training.

### Tests nhánh khác — **OTHER TEST**

| Test | Nhánh |
|---|---|
| `tests/test_ewc_checkpoint.py`, `tests/test_tddi_paper_member_ewc.py` | EWC. |
| `tests/test_gem_agem.py` | GEM/A-GEM. |
| `tests/test_backbone_adapters.py` | TabM/DDI-GCN. |
| `tests/test_s02_prediction_export.py` | S02 export. |
| `tests/test_s03_calibration.py` | Calibration cũ. |
| `tests/test_s04_aggregation.py` | Aggregation cũ. |
| `tests/test_s01_hardening.py` | Shared/legacy invariants. |
| `tests/test_tddi_ensemble3_study.py` | EWC orchestrator compatibility. |

## 20. Muốn thay đổi gì thì sửa ở đâu?

| Muốn đổi | File chính |
|---|---|
| Hyperparameter/budget/batch/epoch/output | Main study config |
| Kiến trúc 3780→7560→7560 | `src/models/tddi_paper_member.py` |
| Focal/output/feature distillation | `train_cil.py::train_one_epoch` |
| Memory allocation/ranking | `FixedBudgetReplayBuffer` |
| Replay order/draws | `FixedReplaySampler` + config |
| Member seed | `src/utils/seed.py` |
| Head expansion/old-row copy | `train_cil.py` + `class_mapping.py` |
| Resume | `src/training/replay_checkpoint.py` |
| Three-member orchestration | `src/training/tddi_ensemble3_study.py` |
| Member prediction schema | `src/eval/member_predictions.py` |
| Ensemble/UE formulas | `src/eval/offline_ensemble.py` |
| UE quality audit | `src/eval/offline_ue_audit.py` |
| Threshold grid/rule | Threshold config; implementation ở `confidence_threshold.py` |
| Temperature calibration | Calibration config/module |
| Final Markdown/tables | `src/eval/build_final_report.py` |
| GPU commands | P3 GPU runbook |

Đổi chỉ cách trình bày final report từ artifact có sẵn thường không cần train lại.
Đổi backbone, loss, replay, seed, task protocol hoặc preprocessing thì phải train lại.

## 21. Checklist xác nhận đúng study

1. `run_config.json`: method là `replay_distill_fixed_budget_uniform`.
2. Variant là `tddi_paper_member`.
3. Task-file/hash là P3 `tail_to_head_tasks.json`.
4. Experiment seed 0; member IDs 0,1,2 và member seeds khác nhau.
5. Mỗi member hoàn thành task 0..7; final seen-all đủ 178 class.
6. Memory budget và replay draws là 6.800.
7. Ba member artifacts cùng sample/class order.
8. Ensemble dùng mean probabilities.
9. Primary threshold chọn từ validation; test chỉ load frozen artifact.
10. Selective metric luôn đi cùng coverage.
11. UE được đánh giá bằng error AUROC/AUPRC và AURC.
12. Final task 7/seen-all tách khỏi mean-across-stages phụ.

Nếu đủ các điểm trên, pipeline đúng là:

```text
replay_distill_fixed_budget_uniform
× tddi_paper_member
× ensemble 3 members
× P3 tail-to-head
× experiment seed 0
× 8 tasks / 178 classes
```
