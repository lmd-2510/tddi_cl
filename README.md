# DDI-CIL — T-DDI CB-Replay / EWC / A-GEM

Repository hiện tập trung vào study:

```text
class_balanced_fixed_budget_replay
× tddi_paper_member ensemble 3
× Ordered hoặc Shuffled LT-CIL
× experiment seed 0
× 8 task / 178 class
```

## Bài toán Class-Incremental Learning

Trong Class-Incremental Learning (CIL), các class mới xuất hiện tuần tự qua nhiều
task. Sau khi học task mới, model phải phân biệt tất cả class đã thấy mà không được
cung cấp task ID lúc inference.

Với study này:

```text
task 0 → task 1 → ... → task 7
                        │
                        └── model cuối dự đoán trên toàn bộ 178 class
```

Khó khăn chính là **catastrophic forgetting**: khi tối ưu cho dữ liệu task mới, trọng
số và output head có thể dịch chuyển làm hiệu năng trên class cũ giảm mạnh. Class mới
cũng thường có nhiều lượt cập nhật hơn exemplar cũ, tạo recency bias về phía task mới.

## Backbone T-DDI numerical member

Backbone T-DDI duy nhất dùng cho study chính là `tddi_paper_member`:

```text
3.780 numerical QSAR features
→ LayerNorm(3780)
→ Linear(3780, 7560) → GELU → Dropout
→ Linear(7560, 7560) → GELU → Dropout
→ expandable class head C_t
```

Output head mở rộng khi class mới xuất hiện. Các hàng ứng với class cũ được sao chép
sang head mới theo raw class ID.

Backbone này là paper-size numerical CIL member. Repository không tuyên bố toàn bộ
training recipe là bản tái tạo chính xác paper T-DDI.

## Fixed-budget replay

Method Replay chính thức là `class_balanced_fixed_budget_replay` (CB-Replay), một biến thể dành cho long-tail:

- Buffer chia đều dung lượng khả dụng theo class; ranking bằng raw sample-normalized class mean.

- Mỗi member có replay buffer bằng 4% development data: 27.778 slots.
- Exemplar cũ được trộn với dữ liệu task hiện tại từ task 1.
- Replay mục tiêu chiếm 12,5% số draws mỗi epoch, repeat cap 3/exemplar.
- Current samples dùng Focal Loss với `gamma=1.0`.
- Replay samples dùng Cross-Entropy.
- Baseline chính thức không dùng logit hoặc feature distillation.
- Checkpoint được lưu ở task boundary để có thể resume đúng task tiếp theo.

Replay cung cấp lại bằng chứng thật của class cũ nhằm giảm forgetting; hiệu quả vẫn
được đo sau toàn bộ trajectory tám task.

## Ba method chính thức

- CB-Replay: Focal current + CE replay; buffer equal-class, ranking sample-normalized class mean.
- EWC: Cross-Entropy + online Fisher penalty; không lưu exemplar. Mặc định lambda = 1000.
- A-GEM: Cross-Entropy trên current data; chiếu gradient theo minibatch memory ngẫu nhiên.
  Memory chọn seeded uniform từng task, tổng dung lượng tối đa 27.778 mẫu/member;
  reference minibatch mặc định 256. Memory chỉ lưu raw features và ID/label.

Cả ba dùng frozen fold/preprocessing và checkpoint `task_N.pt`.
Config JSON giữ baseline CB-Replay; chọn `--method ewc` hoặc `--method agem` khi
gọi launcher để dùng cùng hai protocol và cùng preprocessing, với output namespace riêng.

## Ensemble3 và uncertainty estimation

Study huấn luyện ba member độc lập và tuần tự:

| Member ID | Member seed |
| ---: | ---: |
| 0 | 409845317 |
| 1 | 215626784 |
| 2 | 3041879697 |

`experiment_seed=0` giữ protocol, task order, class map và cách dựng fold. Mỗi member
giữ hai fold để train, fold còn lại để validation; vì tập train khác nhau nên scaler
và exemplar IDs cũng riêng theo member. `member_seed` điều khiển initialization,
dropout và sampler/DataLoader order.

Ba member không phải ba experiment seed. Chúng thuộc cùng một experiment seed và được
dùng để tạo ensemble/uncertainty.

Sau khi cả ba member hoàn tất:

```text
member 0 probabilities ─┐
member 1 probabilities ─┼─→ mean probabilities → prediction + UE
member 2 probabilities ─┘
```

Offline ensemble tuyệt đối không trung bình raw logits. Các uncertainty/diversity
metric gồm predictive entropy, mutual information, probability variance và pairwise
disagreement.

## Hai protocol LT-CIL chính thức

Repo chỉ hỗ trợ hai protocol lấy theo hai scenario của paper *Long-Tailed Class
Incremental Learning* (ECCV 2022):

- `head_to_tail` — Ordered LT-CIL: sắp class theo số mẫu train giảm dần, vì vậy
  task đầu chứa head class và task cuối tiến dần về tail class.
- `shuffled` — Shuffled LT-CIL: lấy cùng danh sách class long-tail, xáo trộn bằng
  seed cố định rồi mới cắt thành các task. Mỗi task có mức mất cân bằng tự nhiên khác nhau.

Đây là cách áp dụng quy tắc thứ tự của paper lên tần suất tự nhiên của DDI. Repo giữ
layout 8 task hiện tại, không sao chép layout CIFAR/ImageNet của paper.

```text
Task layout: [38, 20, 20, 20, 20, 20, 20, 20]
Total:       178 raw classes
```

Hai task schedule canonical đã đóng băng:

```text
study_assets/task_protocols/head_to_tail_tasks.json
study_assets/task_protocols/shuffled_seed0_tasks.json
```

`head_to_tail` có `seed: null` vì thứ tự được xác định từ tần suất class. `shuffled`
luôn ghi seed tạo thứ tự; asset canonical hiện dùng seed 0. Các protocol cũ
`tail_to_head`, `frequency_balanced` và `constrained_mass_balanced` không còn hợp lệ
cho run mới.

## Study assets và outputs

Đầu vào đã đóng băng để tái hiện study nằm trong `study_assets/`:

```text
study_assets/
├── data_schema/
│   ├── feature_columns.json   # đúng 3.780 feature và đúng thứ tự
│   └── global_class_map.json  # raw class ID → dense index để audit
├── stratified_3fold_seed42/
│   ├── fold_assignments.parquet
│   └── fold_manifest.json
├── preprocessing_<protocol>_seed0_fold42/
│   └── member_<0|1|2>/
│       └── fold_preprocessing.json  # fit Task 0 train của từng member rồi freeze
└── task_protocols/
    ├── head_to_tail_tasks.json
    └── shuffled_seed0_tasks.json
```

`outputs/` chỉ dành cho artifact sinh ra khi chạy:

```text
outputs/ordered_cb_replay_seed0/
outputs/shuffled_cb_replay_seed0/
outputs/head_to_tail_ewc_seed0/
outputs/shuffled_ewc_seed0/
outputs/head_to_tail_agem_seed0/
outputs/shuffled_agem_seed0/
```

Các namespace output lớn được Git ignore. Không xóa run directory trên server nếu còn
cần checkpoint, resume, member predictions hoặc báo cáo đã tạo.

## Config stratified 3-fold hiện tại

Chỉ có ba config chính thức. Hai config train dùng cùng một baseline, chỉ khác
protocol, preprocessing artifact và output namespace:

```text
configs/ordered_baseline.json
configs/shuffled_seed0_baseline.json
configs/eval_macro_f1_entropy.json
```

Các launcher Bash và Python controller theo method cũ đã được loại khỏi `scripts/`.
Run mới dùng hai protocol và ba config chính thức ở trên.

Thông số chính:

| Nhóm | Giá trị |
| --- | --- |
| Microbatch | 64 |
| Effective batch | 1024 |
| Gradient accumulation | 16 |
| Epoch tối đa | 30/task |
| Early-stopping patience | 5 |
| Optimizer | AdamW |
| Learning rate | 0.001 |
| Weight decay | 0.0001 |
| Dropout | 0.2 |
| Activation | GELU |
| Input normalization | LayerNorm |
| Replay memory | 4% development cho mỗi member: 27.778 slots/member |
| Replay mỗi epoch | Mục tiêu 12,5%, repeat cap 3 |
| Preprocessing | StandardScaler riêng từng member, fit task 0 rồi freeze |
| Exemplar ranking | Sample-normalized class mean |

## Scripts chuẩn bị dữ liệu

`scripts/` chỉ giữ utility dữ liệu, preprocessing, cache và visualization; không còn
Bash launcher hay Python controller huấn luyện theo method:

| Script | Khi nào cần dùng |
| --- | --- |
| `convert_csv_to_parquet.py` | Khi máy mới chỉ có dataset CSV |
| `inspect_splits.py` | Khi cần audit schema/tạo lại feature list |
| `analyze_class_distribution.py` | Khi cần tính lại tần suất class |
| `build_cil_tasks.py` | Tạo lại Ordered head-to-tail và Shuffled task schedule |
| `check_leakage.py` | Khi cần kiểm tra overlap/leakage giữa các split |
| `build_stratified_3fold_assignments.py` | Tạo stratified 3-fold assignment |
| `audit_development_folds.py` | Kiểm tra fold assignment với Parquet nguồn |
| `prepare_fold_preprocessing.py` | Fit preprocessing task-0 riêng cho từng member rồi freeze |
| `build_source_shard_cache.py` | Tạo source-shard cache cho pipeline mới |
| `visualize_cil_run.py` | Tạo biểu đồ chẩn đoán từ artifact sau train |

Các script này không chạy trong mỗi epoch. Với full run hiện tại, training đọc thẳng
asset đã đóng băng trong `study_assets/`.

## Dry-run full 8 task

Dry-run kiểm tra config, seed, đường dẫn và command được lập kế hoạch. Nó không tạo
model và không train:

```bash
python src/training/fold_ensemble3_full.py \
  --config configs/ordered_baseline.json \
  --member-id 0
```

Chỉ thêm `--execute` khi thực sự muốn chạy trên máy GPU. Ba member phải chạy tuần tự,
không giữ hai model trên GPU đồng thời.

Chạy cùng protocol với EWC hoặc A-GEM (dry-run):

```bash
python src/training/fold_ensemble3_full.py --config configs/ordered_baseline.json --method ewc
python src/training/fold_ensemble3_full.py --config configs/shuffled_seed0_baseline.json --method agem
```

Đổi `--preprocessing-root`, `--fold-assignments`, `--fold-manifest`, `--train`,
`--validation`, `--test` khi asset nằm ở thư mục khác trên server.
Preprocessing nằm tại `member_N/fold_preprocessing.json`, không còn thư mục case `B`.
Bật source/task cache bằng `CIL_SOURCE_SHARD_CACHE_ROOT` và `CIL_COMMON_TASK_CACHE_ROOT`;
mỗi protocol có cache riêng, provenance sai sẽ bị từ chối.

## Nguyên tắc đánh giá

Kết quả chính của một run là model cuối sau task 7 được đánh giá trên toàn bộ 178 class:

```text
task_id=7
eval_task_id=seen_all
split=test_seen_all
```

Các metric chính gồm Accuracy, Macro-F1, Weighted F1 và forgetting.
Mean metric qua tám training stage chỉ mô tả trajectory, không được gọi là final model
performance.

Config threshold chính thức duy nhất dùng stratified 3-fold: chọn threshold trên
prediction OOF bằng Macro-F1 cao nhất với coverage tối thiểu 50%, dùng
normalized-entropy confidence, đóng băng trước khi áp dụng lên test và luôn báo cáo
coverage. Không còn policy threshold Accuracy/Balanced Accuracy riêng.

Ba ensemble member không được dùng thay cho nhiều experiment seed. Muốn báo cáo
`mean ± sample standard deviation` qua năm seed, cần chạy đầy đủ experiment seed 0–4,
mỗi seed gồm ba member riêng.

## Tổng kết kết quả sau train

Đầu mối báo cáo chính thức là `src/eval/report.py`. File này chỉ đọc artifact đã hoàn
tất; nó không train và không chạy inference. Nó tạo Markdown report và các bảng CSV
cho metric từng task, forgetting của ensemble/member và ensemble diversity/uncertainty.

Report tự đọc protocol từ full manifest/run config. Có thể truyền `--task-file` khi cần chỉ định rõ.

Với Ordered LT-CIL:

```bash
python src/eval/report.py \
  --full-root outputs/ordered_cb_replay_seed0 \
  --overwrite
```

Lệnh này không train, không inference và không cần GPU.

Với Shuffled LT-CIL:

```bash
python src/eval/report.py \
  --full-root /absolute/path/to/completed_run \
  --overwrite
```

Kết quả được ghi vào `<full-root>/final_results/`.

## Cấu trúc source hiện tại

```text
fold_ensemble3_full.py → fold_ensemble3_runner.py → fold_run_setup.py
                                                   ↓
                                              train_cil.py
                                                   ↓
                                              fold_training.py
```

`src/methods/` chỉ còn `replay.py`, `ewc.py`, `agem.py`.
`class_balanced_replay_buffer.py` lưu exemplar, `replay_sampler.py` lập lịch draws;
`regularization_memory.py` quản lý memory A-GEM và trạng thái không exemplar của EWC.
`replay_checkpoint.py` lưu checkpoint chung của cả ba method.

Đã bỏ teacher/distillation, sqrt quota, pipeline-input ranking, rotating-current,
GEM, DER++, WA và prepared-cache cũ. Checkpoint/config/threshold schema cũ không còn
tương thích; run mới dùng namespace mới, không tiếp tục checkpoint lịch sử.
Fold assignments và preprocessing phải được chuẩn bị trên dữ liệu server trước khi train.
