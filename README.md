# DDI-CIL — T-DDI Ensemble3 Replay-Distill

Repository hiện tập trung vào study:

```text
replay_distill_fixed_budget_uniform
× tddi_paper_member ensemble 3
× P3 tail-to-head
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

## Replay và distillation

Method chính là `replay_distill_fixed_budget_uniform`:

- Tổng ngân sách ba replay buffer là 4% development data: 27.778 slots.
- Member 0/1/2 lần lượt có ngân sách 9.260/9.259/9.259 slots.
- Exemplar cũ được trộn với dữ liệu task hiện tại từ task 1.
- Replay mục tiêu chiếm 12,5% số draws mỗi epoch, repeat cap 3/exemplar.
- Classification sử dụng Focal Loss với `gamma=1.0`.
- Frozen teacher từ task trước cung cấp logit distillation với `alpha=1.0` và
  `temperature=2.0`.
- Feature distillation dùng MSE với trọng số `0.5`.
- Checkpoint được lưu ở task boundary để có thể resume đúng task tiếp theo.

Replay cung cấp lại bằng chứng thật của class cũ; distillation giữ phân phối đầu ra và
latent representation gần teacher trước đó. Hai cơ chế phối hợp nhằm giảm forgetting,
nhưng forgetting vẫn phải được đo thực nghiệm sau toàn bộ trajectory.

## Ensemble3 và uncertainty estimation

Study huấn luyện ba member độc lập và tuần tự:

| Member ID | Member seed |
| ---: | ---: |
| 0 | 409845317 |
| 1 | 215626784 |
| 2 | 3041879697 |

`experiment_seed=0` giữ protocol, task order, class map và cách dựng fold. Mỗi member
giữ hai fold để train, fold còn lại để validation; vì tập train khác nhau nên scaler
B và exemplar IDs cũng riêng theo member. `member_seed` điều khiển initialization,
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

## Protocol P3

P3 sử dụng `tail_to_head`: các class hiếm được đưa vào trước, class phổ biến được đưa
vào sau.

```text
Task layout: [38, 20, 20, 20, 20, 20, 20, 20]
Total:       178 raw classes
```

Task schedule đã đóng băng:

```text
study_assets/task_protocols/tail_to_head_tasks.json
```

P3 task-file có `seed: null` vì tail-to-head được xác định từ tần suất class, không
dùng random permutation. Experiment seed vẫn được lưu trong manifest để đảm bảo
provenance của study.

## Study assets và outputs

Đầu vào đã đóng băng để tái hiện study nằm trong `study_assets/`:

```text
study_assets/
├── data_schema/
│   └── feature_columns.json   # đúng 3.780 feature và đúng thứ tự
├── preprocessing/
│   └── scaler.pkl             # StandardScaler fit chỉ trên train split
└── task_protocols/
    └── tail_to_head_tasks.json
```

`outputs/` chỉ dành cho artifact sinh ra khi chạy:

```text
outputs/full/                  # full study
outputs/pilot/                 # pilot runs
outputs/smoke/                 # smoke tests
outputs/remote_preflight/      # kiểm tra máy GPU
```

Các namespace output lớn được Git ignore. Không xóa `outputs/full` trên server nếu còn
cần checkpoint, resume, member predictions hoặc báo cáo đã tạo.

## Config stratified 3-fold hiện tại

Repo chỉ giữ các config full gắn với kết quả P3 và run P4 kế tiếp:

```text
configs/full_tddi_p3_fold_ensemble3_seed0.json
configs/full_tddi_p3_fold_ensemble3_seed0_epochs25.json
configs/full_tddi_p3_fold_ensemble3_seed0_replay12p5.json
configs/full_tddi_p4_fold_ensemble3_seed0.json
```

Các config smoke/pilot đã hoàn thành vai trò kiểm chứng kỹ thuật và đã được loại bỏ.
Hướng dẫn chạy P4 hiện tại nằm tại
`docs/TDDI_P4_EPOCH20_MEMBER0_RUNBOOK.md`.

Thông số chính:

| Nhóm | Giá trị |
| --- | --- |
| Microbatch | 64 |
| Effective batch | 1024 |
| Gradient accumulation | 16 |
| Epoch tối đa | 20/task |
| Early-stopping patience | 5 |
| Optimizer | AdamW |
| Learning rate | 0.001 |
| Weight decay | 0.0001 |
| Dropout | 0.2 |
| Activation | GELU |
| Input normalization | LayerNorm |
| Replay memory | Tổng 4% development: 9.260 / 9.259 / 9.259 slots |
| Replay mỗi epoch | Mục tiêu 12,5%, repeat cap 3 |
| Preprocessing | StandardScaler riêng từng member, fit task 0 rồi freeze |
| Exemplar ranking | Sample-normalized class mean |

## Scripts chuẩn bị dữ liệu

`scripts/` chỉ giữ sáu utility có ích để tái tạo hoặc kiểm tra study asset:

| Script | Khi nào cần dùng |
| --- | --- |
| `convert_csv_to_parquet.py` | Khi máy mới chỉ có dataset CSV |
| `inspect_splits.py` | Khi cần audit schema/tạo lại feature list |
| `preprocess_features.py` | Khi cần fit lại scaler từ train split |
| `analyze_class_distribution.py` | Khi cần tính lại tần suất class |
| `build_cil_tasks.py` | Khi cần tạo lại P3/P4 task schedule |
| `check_leakage.py` | Khi cần kiểm tra overlap/leakage giữa các split |

Các script này không chạy trong mỗi epoch. Với full run hiện tại, training đọc thẳng
asset đã đóng băng trong `study_assets/`.

## Dry-run full 8 task

Dry-run kiểm tra config, seed, đường dẫn và command được lập kế hoạch. Nó không tạo
model và không train:

```bash
python src/training/fold_ensemble3_full.py \
  --config configs/full_tddi_p3_fold_ensemble3_seed0.json \
  --member-id 0
```

Chỉ thêm `--execute` khi thực sự muốn chạy trên máy GPU. Ba member phải chạy tuần tự,
không giữ hai model trên GPU đồng thời.

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

Config chính dùng stratified 3-fold: threshold được chọn từ prediction OOF, dùng
normalized-entropy confidence, đóng băng trước khi áp dụng lên test và luôn báo cáo
coverage. Các config threshold còn lại nằm trực tiếp trong `configs/`.

Ba ensemble member không được dùng thay cho nhiều experiment seed. Muốn báo cáo
`mean ± sample standard deviation` qua năm seed, cần chạy đầy đủ experiment seed 0–4,
mỗi seed gồm ba member riêng.

## Tài liệu chính

1. `docs/TDDI_P4_EPOCH20_MEMBER0_RUNBOOK.md` — P4: chạy riêng từng member hoặc một lệnh nohup chạy 0 → 1 → 2 tuần tự rồi ensemble/UE.
2. `docs/ensemble3_p3_final_report.md` — báo cáo P3 baseline.
3. `docs/ensemble3_p3_12,5%_final_report.md` — báo cáo P3 được giữ để đối chiếu.
4. `docs/TDDI_ENSEMBLE3_REPLAY_DISTILL_P3_RESULTS.md` — tổng hợp kết quả P3 và giải thích metric.

Final report mở rộng được dựng lại từ artifact đã có bằng:

```bash
python src/eval/report.py \
  --full-root outputs/stratified_ensemble3/full_p3_seed0_8tasks \
  --task-file study_assets/task_protocols/tail_to_head_tasks.json \
  --overwrite
```

Lệnh này không train, không inference và không cần GPU.
