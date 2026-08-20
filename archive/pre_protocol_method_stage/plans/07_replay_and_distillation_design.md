# Replay and Distillation Design

## Objective

Thiết kế các continual baselines chính, tập trung vào replay, replay + distillation, và upper/lower bounds để đo catastrophic forgetting trên DDI2025-CIL.

## Required Baselines

1. `joint_seen`
2. `sequential_finetune`
3. `replay_random`
4. `replay_balanced`
5. `replay_distill`
6. `ewc` hoặc `si` như extension nếu còn thời gian

## Method Definitions

### 1. joint_seen

- upper bound tham chiếu
- tại task `t`, train trên toàn bộ seen-class training data
- không phải continual method thực sự
- dùng để định lượng gap do forgetting

### 2. sequential_finetune

- train nối tiếp từng task
- không replay
- không distillation
- lower bound để đo forgetting mạnh đến mức nào

### 3. replay_random

- giữ memory exemplar từ old classes
- sampling random trong mỗi class hoặc toàn cục, cần ghi rõ
- dùng old exemplars + current task data

### 4. replay_balanced

- replay class-balanced
- ưu tiên mỗi old class có số exemplar gần bằng nhau
- phù hợp long-tail hơn random replay

### 5. replay_distill

- replay + teacher/student distillation
- teacher là frozen model sau task trước
- distillation chỉ áp dụng trên logits của old classes

## Memory Design

### Main Setting

```text
memory_per_class = 50
178 * 50 = 8900 samples
```

### Ablations

- `20 samples/class`
- `50 samples/class`
- `100 samples/class`

## Memory Constraints

- không chứa future classes
- chỉ lấy từ train split
- phải lưu class distribution của memory sau mỗi task
- nếu task đầu có ít lớp hiếm, cần log số exemplar thực có thể lấy

## Replay Batch Composition

Default target:

```text
batch_size = 512
new samples = 256
replay samples = 256
```

Nếu task hiện tại nhỏ hoặc class imbalance quá mạnh:

- dùng balanced sampler theo class
- hoặc adapt tỷ lệ replay/new, nhưng phải lưu config rõ ràng

## Distillation Loss

```text
loss = CE(student_logits, labels) + alpha * KL(student_old_logits / T, teacher_old_logits / T) * T^2
```

Default:

- `alpha = 1.0`
- `temperature = 2.0`

## Required Artifacts

```text
outputs/checkpoints/task_{t}_model.pt
outputs/checkpoints/task_{t}_teacher.pt
outputs/memory/memory_after_task_{t}.parquet
outputs/memory/memory_summary.csv
```

## Planned Modules

```text
src/methods/joint_seen.py
src/methods/sequential.py
src/methods/replay.py
src/methods/replay_distill.py
src/methods/ewc.py
src/data/replay_buffer.py
```

## Pseudocode

```python
for task_id in tasks:
    current_data = load_current_task_data(task_id)

    if method == "sequential":
        train_data = current_data
    elif method in {"replay_random", "replay_balanced", "replay_distill"}:
        replay_data = memory.sample(...)
        train_data = merge(current_data, replay_data)

    if method == "replay_distill" and task_id > 0:
        teacher = load_frozen_previous_model()
        loss = ce_loss + alpha * distill_loss

    train_one_task(...)
    update_memory_from_current_task(...)
```

## Method-Specific Checklists

### Replay

- [ ] Memory không chứa future classes
- [ ] Memory chỉ lấy từ train split
- [ ] Mỗi class cũ có `<= memory_per_class` samples
- [ ] Memory summary được lưu sau mỗi task

### Distillation

- [ ] Teacher là frozen model từ task trước
- [ ] Distillation chỉ áp dụng trên old class logits
- [ ] Có log rõ `alpha` và `temperature`

## Planned Commands

```bash
python src/training/train_cil.py \
  --data-config configs/data.yaml \
  --model-config configs/model_mlp_base.yaml \
  --task-file outputs/tasks/random_seed0_tasks.json \
  --method replay \
  --memory-per-class 50 \
  --outdir outputs/runs/random_seed0_replay50_mlpbase
```

```bash
python src/training/train_cil.py \
  --data-config configs/data.yaml \
  --model-config configs/model_mlp_base.yaml \
  --task-file outputs/tasks/random_seed0_tasks.json \
  --method replay_distill \
  --memory-per-class 50 \
  --temperature 2.0 \
  --distill-alpha 1.0 \
  --outdir outputs/runs/random_seed0_replaydistill50_mlpbase
```

## Acceptance Criteria

- [ ] Có upper bound và lower bound rõ ràng
- [ ] Memory design class-balanced được mô tả đủ để code lại
- [ ] Distillation logic không đụng future classes
- [ ] Artifacts đủ để audit từng task

## Definition of Done

- [ ] Baseline continual methods đã được định nghĩa rõ
- [ ] Memory và distillation có config mặc định + ablation path
