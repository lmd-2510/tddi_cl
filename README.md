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

Các preset `small`, `base`, `large` còn lại chỉ là MLP baseline generic và không được
gọi là T-DDI.

## Replay và distillation

Method chính là `replay_distill_fixed_budget_uniform`:

- Replay buffer có tổng ngân sách cố định 6.800 exemplar.
- Exemplar cũ được trộn với dữ liệu task hiện tại từ task 1.
- Mỗi epoch dùng 6.800 replay draws.
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

`experiment_seed=0` giữ protocol, task order, class map, preprocessing và exemplar
identity dùng chung. `member_seed` chỉ tạo khác biệt ở initialization, dropout và
sampler/DataLoader order.

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

## Config chính

```text
configs/tddi_ensemble3_replay_distill_p3_seed0.json
```

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
| Replay memory | 6.800 exemplars |

## Scripts chuẩn bị dữ liệu

`scripts/` chỉ giữ sáu utility có ích để tái tạo hoặc kiểm tra study asset:

| Script | Khi nào cần dùng |
| --- | --- |
| `convert_csv_to_parquet.py` | Khi máy mới chỉ có dataset CSV |
| `inspect_splits.py` | Khi cần audit schema/tạo lại feature list |
| `preprocess_features.py` | Khi cần fit lại scaler từ train split |
| `analyze_class_distribution.py` | Khi cần tính lại tần suất class |
| `build_cil_tasks.py` | Khi cần tạo lại P3 task schedule |
| `check_leakage.py` | Khi cần kiểm tra overlap/leakage giữa các split |

Các script này không chạy trong mỗi epoch. Với full run hiện tại, training đọc thẳng
asset đã đóng băng trong `study_assets/`.

## Dry-run

Dry-run kiểm tra config, seed, đường dẫn và command được lập kế hoạch. Nó không tạo
model và không train:

```bash
python src/training/tddi_ensemble3_study.py \
  --config configs/tddi_ensemble3_replay_distill_p3_seed0.json \
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

Các metric chính gồm Accuracy, Balanced Accuracy, Macro-F1, Weighted F1 và forgetting.
Mean metric qua tám training stage chỉ mô tả trajectory, không được gọi là final model
performance.

Confidence threshold phải được chọn duy nhất trên validation, lưu thành frozen
artifact rồi mới áp dụng lên test. Metric threshold luôn phải báo cáo cùng coverage.
Chất lượng UE cần được đánh giá bằng error-detection AUROC/AUPRC, AURC và risk–coverage;
diversity cao một mình chưa chứng minh uncertainty hữu ích.

Ba ensemble member không được dùng thay cho nhiều experiment seed. Muốn báo cáo
`mean ± sample standard deviation` qua năm seed, cần chạy đầy đủ experiment seed 0–4,
mỗi seed gồm ba member riêng.

## Tài liệu chính

1. `docs/TDDI_PAPER_REPLAY_DISTILL_P3_8TASK_GPU_RUNBOOK.md` — setup và lệnh chạy trên
   máy GPU.
2. `docs/TDDI_REPLAY_DISTILL_ENSEMBLE3_FILES_GUIDE.md` — vai trò của các file đã triển
   khai.
3. `docs/TDDI_ENSEMBLE3_REPLAY_DISTILL_P3_RESULTS.md` — kết quả P3 và giải thích metric.

Final report mở rộng được dựng lại từ artifact đã có bằng:

```bash
python src/eval/build_final_report.py \
  --full-root outputs/full/tddi_ensemble3_replay_distill_p3_seed0_8tasks_v1 \
  --task-file study_assets/task_protocols/tail_to_head_tasks.json \
  --overwrite
```

Lệnh này không train, không inference và không cần GPU.
