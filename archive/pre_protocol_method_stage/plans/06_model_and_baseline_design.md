# Model and Baseline Design

## Objective

Định nghĩa baseline model family cho descriptor-only DDI classification và khung training static/CIL tối thiểu để so sánh các method continual learning.

## Modeling Boundaries

- chỉ dùng descriptor numeric làm input baseline chính
- không dùng SMILES encoder
- không dùng `drugid`, `drugname`, hoặc text field làm feature
- không trộn exploratory interpretation vào logic training

## Main Baseline Family

### MLP-base

```text
Input: 3780 descriptor features
LayerNorm hoặc BatchNorm1d
Linear 3780 -> 1024
GELU/ReLU
Dropout 0.2
Linear 1024 -> 512
GELU/ReLU
Dropout 0.2
Linear 512 -> num_seen_classes
```

### Variants

1. `MLP-small`
   - `3780 -> 512 -> 256 -> head`
2. `MLP-base`
   - `3780 -> 1024 -> 512 -> head`
3. `MLP-large`
   - `3780 -> 2048 -> 1024 -> 512 -> head`

Main khuyến nghị ở vòng đầu: **MLP-base**

## Output Head Policy

- head thay đổi theo `num_seen_classes`
- tại task `t`, output dimension = số seen classes
- label phải remap từ global class id sang local seen-class index

## Label Mapping Requirements

- lưu `global_class_map.json`
- lưu `seen_class_map_task_{t}.json`
- không suy luận map ngầm từ thứ tự xuất hiện trong minibatch
- không giả định class IDs liên tục kiểu `0..177`

Global mapping phải được tạo từ class values thực:

```python
unique_classes = sorted(train["class"].unique())
global_class_map = {cls: i for i, cls in enumerate(unique_classes)}
num_classes = len(unique_classes)
```

Không dùng:

```python
num_classes = max(class_values) + 1
```

## Loss Functions

### Baseline

- `CrossEntropyLoss`

### Planned Alternatives

- class-balanced cross entropy
- focal loss nếu imbalance gây lỗi rõ rệt ở tail classes

## Optimizer and Training Defaults

- optimizer: `AdamW`
- learning rate: `1e-3` hoặc `3e-4`
- weight decay: `1e-4`
- batch size target: `512` hoặc `1024`, tùy RAM/VRAM
- early stopping theo validation `Macro-F1` trên seen classes

## Training-Time Logging Requirements

Training loop không được chạy “im lặng”. Cần in rõ tiến trình để người chạy biết model đang ở đâu, có đang học hay bị treo, và checkpoint nào đã được lưu.

### Console Logging

Mỗi run phải in ra màn hình:

- `run_id`, `protocol`, `seed`, `method`, `model`
- đường dẫn data/config/task file
- `task_id`, số seen classes, số current classes
- số train/validation/test samples của task hiện tại
- với replay: số replay samples và `memory_per_class`
- với distillation: `alpha`, `temperature`

### Per-Epoch Logging

Ít nhất mỗi epoch phải in:

- `epoch/current_epoch`
- `train_loss`
- `val_loss` nếu có
- `val_macro_f1`
- `val_balanced_accuracy`
- `best_val_macro_f1_so_far`
- learning rate hiện tại
- thời gian mỗi epoch

### Per-Task Logging

Sau mỗi task phải in:

- metric trên seen classes
- performance theo task group cũ/mới
- đường dẫn checkpoint vừa lưu
- tóm tắt memory sau update

### Progress Bars

Khuyến nghị dùng `tqdm` hoặc progress bar tương đương cho:

- train dataloader
- validation dataloader
- task loop nếu run nhiều task liên tiếp

### Warning Conditions

Console log phải hiện rõ warning khi:

- phát hiện `NaN/inf`
- metric validation giảm liên tục
- class/task hiện tại quá ít mẫu
- checkpoint không được lưu
- memory không đủ exemplar cho một lớp cũ

## Data Interface Proposal

Planned modules:

```text
src/models/mlp.py
src/training/train_static.py
src/training/train_cil.py
src/data/ddi_dataset.py
src/data/class_mapping.py
```

## Dataset Object Expectations

- load theo Parquet hoặc artifact preprocess
- nhận `feature_cols` explicit
- tách `X`, `y`, và metadata nếu cần audit
- hỗ trợ subset theo class list cho từng task

## Static Baseline Purpose

Trước khi chạy CIL phải có static baseline để:

- xác nhận pipeline đọc đúng dữ liệu
- xác nhận không có leakage
- tạo upper sanity check cho descriptor-only classification

## CIL Baseline Methods Covered Elsewhere

File này chỉ định nghĩa model family. Logic continual methods được mở rộng ở:

- `07_replay_and_distillation_design.md`
- `08_metrics_and_evaluation_protocol.md`

## Pseudocode

```python
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, num_classes):
        ...

    def forward(self, x):
        return self.head(self.backbone(x))
```

```python
seen_classes = load_seen_classes(task_id)
local_map = build_local_map(seen_classes)
y_local = remap_labels(y_global, local_map)
logits = model(x)
loss = cross_entropy(logits, y_local)
```

## Planned Commands

```bash
python src/training/train_static.py \
  --config configs/model_mlp_base.yaml \
  --data-config configs/data.yaml \
  --outdir outputs/runs/static_mlpbase
```

```bash
python src/training/train_cil.py \
  --data-config configs/data.yaml \
  --model-config configs/model_mlp_base.yaml \
  --task-file outputs/tasks/random_seed0_tasks.json \
  --method sequential \
  --outdir outputs/runs/random_seed0_sequential_mlpbase
```

## Acceptance Criteria

- [ ] Mô hình baseline dùng đúng `3780` descriptor features
- [ ] Có policy rõ ràng cho expandable head và label remapping
- [ ] Có static baseline path riêng trước continual runs
- [ ] Không có text/meta feature nào lọt vào model input
- [ ] Training plan có spec rõ ràng cho console logging và progress bars
- [ ] `num_classes` và head dim được suy ra từ `unique_classes`, không phải `max(class)+1`

## Definition of Done

- [ ] Kiến trúc MLP và config space đã được cố định ở mức đủ để triển khai code
- [ ] Ranh giới giữa static baseline và continual methods đã rõ
