# Class-Incremental Protocol

## Objective

Thiết kế protocol CIL chặt chẽ trên 178 classes để benchmark continual learning, tránh dùng future classes khi train task hiện tại, và đánh giá sau mỗi task trên toàn bộ seen classes.

## Scope

- benchmark chính: **class-incremental**, không phải domain-incremental
- baseline chính: descriptor-only
- evaluate chính: **seen-class evaluation**
- unseen/future classes chỉ dùng cho phân tích phụ nếu có

## Global Protocol Rules

- không dùng future classes khi train task hiện tại
- sau khi train task `t`, evaluate trên tất cả sample thuộc **seen classes up to task t**
- không dùng test set để chọn class order hay hyperparameter
- mỗi class xuất hiện đúng một lần trong task order của một protocol

## Main Task Layout

```text
8 tasks
Task 0: 38 classes
Task 1: +20 classes
Task 2: +20 classes
Task 3: +20 classes
Task 4: +20 classes
Task 5: +20 classes
Task 6: +20 classes
Task 7: +20 classes
Total: 178 classes
```

## Protocol A: Random-CIL Main Benchmark

### Goal

Benchmark chuẩn để đo forgetting mà không cài bias thủ công vào class order.

### Design

- random shuffle 178 classes
- chạy 5 seeds: `0, 1, 2, 3, 4`
- mỗi seed sinh một class order độc lập
- Task 0 nhận 38 classes đầu, Task 1-7 mỗi task 20 classes

### Use

- benchmark chính trong báo cáo đầu tiên
- dùng để so sánh sequential, replay, replay+distill, joint_seen

## Protocol B: Frequency-Balanced CIL

### Goal

Giảm rủi ro task cực lệch do long-tail.

### Design

- chia classes theo frequency bins từ train counts
- mỗi task nhận mix gồm head, medium, tail
- không để một task chỉ toàn lớp rất hiếm hoặc chỉ toàn lớp rất lớn

### Use

- robustness analysis
- kiểm tra độ ổn định của method khi class order bớt ngẫu nhiên hơn

## Protocol C: Long-Tail Emerging CIL

### Goal

Mô phỏng kịch bản lớp phổ biến xuất hiện trước, lớp hiếm xuất hiện muộn.

### Design

- sắp class theo frequency giảm dần hoặc bins có thứ tự head -> medium -> tail
- task sau chứa nhiều lớp hiếm hơn

### Use

- phân tích rare-class forgetting
- phân tích calibration drift trên lớp mới hiếm

## Required Script

```text
scripts/build_cil_tasks.py
```

## Inputs

```text
outputs/class_distribution/class_counts_train.csv
configs/task_config.yaml
```

## Planned Outputs

```text
outputs/tasks/random_seed0_tasks.json
outputs/tasks/random_seed1_tasks.json
outputs/tasks/random_seed2_tasks.json
outputs/tasks/random_seed3_tasks.json
outputs/tasks/random_seed4_tasks.json
outputs/tasks/frequency_balanced_tasks.json
outputs/tasks/long_tail_tasks.json
outputs/tasks/task_summary.csv
outputs/tasks/task_protocol_summary.md
```

## Task JSON Schema

```json
{
  "protocol": "random",
  "seed": 0,
  "num_classes": 178,
  "tasks": [
    {
      "task_id": 0,
      "classes": [1, 7, 19],
      "num_classes": 38
    }
  ]
}
```

## Task Construction Checklist

- [ ] Mỗi class xuất hiện đúng một lần trong task order
- [ ] Tổng class = 178
- [ ] Không class nào bị mất
- [ ] Không class nào bị lặp
- [ ] Class order lấy từ unique class IDs thực, không dùng `range(178)`
- [ ] Task 0 có 38 classes
- [ ] Task 1-7 mỗi task có 20 classes
- [ ] Train/validation/test đều có subset cho seen classes ở mỗi task
- [ ] Ghi rõ số sample train/validation/test ở từng task
- [ ] Ghi rõ số sample rare classes trong từng task

## Sample Filtering Rule Per Task

Tại task `t`:

- train trên sample có `class in current_task_classes` đối với sequential cơ bản
- với replay/joint_seen sẽ có thêm logic method-specific
- validation/test chính sau task `t` dùng `class in seen_classes[:t]`

## Label Mapping

Phải tách rõ:

1. **global class ids**: 178 class gốc
2. **seen-class local ids**: mapping cho head hiện tại

Không được giả định global class ids liên tục kiểu `0..177`. Số lớp là `178` nhưng class IDs thực có thể bỏ số. Vì vậy mọi task file và label map phải dựa trên `sorted(unique_classes)` thay vì `range(num_classes)`.

Ví dụ:

```python
unique_classes = sorted(train["class"].unique())
global_class_map = {cls: i for i, cls in enumerate(unique_classes)}

seen_classes = sorted(all_classes_seen_so_far)
local_map = {global_cls: i for i, global_cls in enumerate(seen_classes)}
y_local = y_global.map(local_map)
```

Mapping này phải được lưu thành artifact.

## Pseudocode

```python
counts = load_train_class_counts()
all_classes = list(counts["class"])
assert len(all_classes) == len(set(all_classes)) == 178

if protocol == "random":
    rng.shuffle(all_classes)
elif protocol == "frequency_balanced":
    all_classes = build_frequency_balanced_order(counts)
elif protocol == "long_tail":
    all_classes = build_head_to_tail_order(counts)

tasks = split_into_tasks(all_classes, base=38, inc=20, num_tasks=8)
validate_task_spec(tasks)
save_task_json(tasks)
```

## Planned Commands

```bash
python scripts/build_cil_tasks.py \
  --class-counts outputs/class_distribution/class_counts_train.csv \
  --num-classes 178 \
  --protocol random \
  --num-tasks 8 \
  --base-task-classes 38 \
  --increment-classes 20 \
  --seeds 0 1 2 3 4 \
  --outdir outputs/tasks
```

## Required Tables in `task_protocol_summary.md`

1. task -> classes
2. task -> sample counts by split
3. task -> rare class counts
4. protocol -> order logic and assumptions

## Acceptance Criteria

- [ ] Người khác có thể sinh lại task JSON chỉ từ counts + config
- [ ] Không có ambiguity về seen classes và local label mapping
- [ ] Có phân biệt rõ benchmark chính và benchmark phụ
- [ ] Không có claim temporal realism quá mức

## Definition of Done

- [ ] Có `scripts/build_cil_tasks.py`
- [ ] Có đủ task JSON và summary artifacts
- [ ] Protocol A/B/C đều có logic, checklist, và risk note rõ ràng
