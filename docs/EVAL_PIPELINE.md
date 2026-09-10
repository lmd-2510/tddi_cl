# Evaluation pipeline hiện tại

Repo hỗ trợ hai chế độ ensemble dùng chung metric, normalized entropy và threshold code.

## Chế độ ensemble

| `ensemble.mode` | Train/validation | Nguồn chọn threshold |
|---|---|---|
| `stratified_3fold` | Gộp train+validation, member `k` train trên hai fold và giữ fold `k` để early stopping/prediction | `offline_ensemble/task_N/oof.npz` |
| `seeded` | Cả ba member dùng split train/validation ban đầu với seed khác nhau | `offline_ensemble/task_N/validation.npz` |

Config train chính `train_tddi_p3_replay_distill_ensemble3_stratified_3fold_seed0.json` dùng `stratified_3fold`, `fold_count=3`, `fold_seed=42`. Config `train_tddi_p3_replay_distill_ensemble3_seeded_seed0.json` giữ chế độ seeded và output root riêng để không trộn artifact.

## Dòng dữ liệu 3-fold

```text
train + validation
        ↓ StratifiedKFold(shuffle=True, random_state=42)
fold A / fold B / fold C
        ↓
member 0 train B+C, OOF A
member 1 train A+C, OOF B
member 2 train A+B, OOF C
        ↓
ghép ba held-out prediction thành một OOF artifact
        ↓
chọn threshold trên OOF

test → ba member dự đoán → mean probability → normalized entropy → frozen threshold
```

Fold assignment được dựng một lần từ stable `drug_a|drug_b` ID và giữ nguyên qua mọi CIL task. OOF folds phải rời nhau; code sẽ dừng nếu có sample trùng hoặc thiếu fold.

## Uncertainty và threshold

UE vận hành chỉ dùng:

```text
p_mean = mean(member probabilities)
H = -sum(p_mean * log(p_mean))
normalized_entropy = H / log(number_of_seen_classes)
entropy_confidence = 1 - normalized_entropy
```

Grid threshold là `0.50..0.99`, bước `0.01`. Code chọn threshold nhỏ nhất có Accuracy nguồn đạt ít nhất `0.95`; nếu không có candidate đạt mục tiêu thì dừng rõ ràng, không tự đổi sang tối ưu Macro-F1.

Ba tier khi đánh giá test:

- high: `confidence >= t_high`;
- medium: `0.50 <= confidence < t_high`;
- low: `confidence < 0.50`.

## Metric

Metric chính là Accuracy, Macro/Weighted F1, Macro Precision/Recall và Weighted Precision. `balanced_accuracy` còn là alias tương thích của Macro Recall. CIL giữ task matrix, average forgetting và average incremental Macro-F1. AURC là metric UE tùy chọn duy nhất trong report; ECE, Brier, NLL và temperature calibration đã bị loại khỏi pipeline.

## Lệnh chạy

Dry-run study:

```bash
python src/training/tddi_ensemble3_study.py \
  --config configs/train_tddi_p3_replay_distill_ensemble3_stratified_3fold_seed0.json
```

Train và tạo OOF/test ensemble:

```bash
python src/training/tddi_ensemble3_study.py \
  --config configs/train_tddi_p3_replay_distill_ensemble3_stratified_3fold_seed0.json \
  --execute
```

Sau khi study hoàn thành, với từng task chọn threshold từ `oof.npz` rồi áp dụng lên `test.npz`:

```bash
python src/eval/threshold.py select \
  --ensemble <ROOT>/offline_ensemble/task_7/oof.npz \
  --config configs/eval_tddi_p3_ensemble_entropy_threshold.json \
  --threshold-out <ROOT>/threshold/task_7/frozen_threshold.json \
  --report-out <ROOT>/threshold/task_7/oof_report.json

python src/eval/threshold.py evaluate \
  --ensemble <ROOT>/offline_ensemble/task_7/test.npz \
  --config configs/eval_tddi_p3_ensemble_entropy_threshold.json \
  --threshold-artifact <ROOT>/threshold/task_7/frozen_threshold.json \
  --report-out <ROOT>/threshold/task_7/test_report.json
```

Không dùng lại artifact `_v1`: schema prediction/ensemble và protocol tạo member đã thay đổi.
