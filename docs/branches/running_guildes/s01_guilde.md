# Hướng dẫn chạy full experiment cho S01

Guide này dùng để tạo đầy đủ hai output của S01 cho từng method và seed:

- `class_trajectory.csv`;
- `class_forgetting.csv`.

Ma trận chạy chính:

```text
seeds   = 0, 1, 2, 3, 4
methods = sequential, replay, replay_distill, joint_seen
tasks   = 8
device  = mps
```

Tổng cộng có **20 CIL runs**. EWC được để ở phần chạy bổ sung vì đây là baseline optional trong experiment roadmap.

> **Trạng thái:** 20/20 main runs đã hoàn thành ngày 2026-08-03. Xem [S01 full experiment results](../../results/s01_results.md).

Các run này không được backfill S02. Với clean runs tạo O01–O04 trong cùng execution,
xem [hướng dẫn S02](s02_guilde.md).

## 1. Chuẩn bị

Chạy các lệnh từ thư mục gốc `DDI-CIL`.

Các file sau phải tồn tại:

```text
train_extracted.parquet
validation_extracted.parquet
test_extracted.parquet
outputs/audit/feature_columns.json
outputs/preprocess/scaler.pkl
outputs/tasks/random_seed0_tasks.json
...
outputs/tasks/random_seed4_tasks.json
```

Kiểm tra nhanh:

```bash
test -f train_extracted.parquet
test -f validation_extracted.parquet
test -f test_extracted.parquet
test -f outputs/audit/feature_columns.json
test -f outputs/preprocess/scaler.pkl
test -f outputs/tasks/random_seed4_tasks.json
```

## 2. Kiểm tra MPS

```bash
.venv/bin/python -c "import torch; print('MPS built:', torch.backends.mps.is_built()); print('MPS available:', torch.backends.mps.is_available())"
```

Chỉ chạy full experiment khi kết quả có:

```text
MPS built: True
MPS available: True
```

Bật fallback cho một số Torch operation chưa được MPS hỗ trợ:

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

## 3. Chạy unit tests trước

```bash
.venv/bin/python -m unittest discover -s tests -v
```

Chỉ tiếp tục khi toàn bộ tests đều `OK`.

## 4. Chạy full S01 experiments

Copy toàn bộ block dưới đây vào terminal:

```bash
set -eo pipefail

S01_PYTHON=".venv/bin/python"
S01_OUT_ROOT="outputs/runs_s01_locked"
S01_SEEDS=(0 1 2 3 4)
S01_METHODS=(sequential replay replay_distill joint_seen)

mkdir -p "${S01_OUT_ROOT}"
export PYTORCH_ENABLE_MPS_FALLBACK=1

for method in "${S01_METHODS[@]}"; do
  for seed in "${S01_SEEDS[@]}"; do
    task_file="outputs/tasks/random_seed${seed}_tasks.json"
    outdir="${S01_OUT_ROOT}/random_seed${seed}_${method}_mlpbase"
    extra_args=()

    if [[ "${method}" == "replay" ]]; then
      extra_args+=(--memory-per-class 50)
    elif [[ "${method}" == "replay_distill" ]]; then
      extra_args+=(
        --memory-per-class 50
        --distill-alpha 1.0
        --temperature 2.0
        --feature-distill-weight 0.5
      )
    fi

    echo "Running method=${method} seed=${seed}"
    if [[ -e "${outdir}" ]]; then
      echo "Refusing to reuse existing run directory: ${outdir}" >&2
      exit 1
    fi

    "${S01_PYTHON}" src/training/train_cil.py \
      --train train_extracted.parquet \
      --validation validation_extracted.parquet \
      --test test_extracted.parquet \
      --feature-cols outputs/audit/feature_columns.json \
      --scaler outputs/preprocess/scaler.pkl \
      --task-file "${task_file}" \
      --outdir "${outdir}" \
      --method "${method}" \
      --variant base \
      --batch-size 1024 \
      --epochs 20 \
      --patience 5 \
      --seed "${seed}" \
      --device mps \
      "${extra_args[@]}"
  done
done
```

Không thêm các option `--max-*-rows-*` khi chạy full experiment.

`train_cil.py` từ chối output directory đã có nội dung. Không chạy lại vào `outputs/runs_s01_full`; đó là artifact legacy. Mỗi clean rerun phải dùng root mới như `outputs/runs_s01_locked`.

Nếu MPS hết memory, dùng `--batch-size 256` cho **toàn bộ 20 runs**. Không trộn kết quả từ các batch size khác nhau trong cùng một bảng so sánh.

`joint_seen` sẽ chạy chậm nhất vì phải load lại toàn bộ dữ liệu của các seen classes sau mỗi task; đây là hành vi bình thường của oracle baseline.

### Giải thích replay và distillation parameters

Các tham số sau được dùng cho `replay` hoặc `replay_distill`:

```bash
--memory-per-class 50
--distill-alpha 1.0
--temperature 2.0
--feature-distill-weight 0.5
```

#### `--memory-per-class 50`

Giới hạn số exemplar thật được lưu cho mỗi class đã học:

```text
tối đa 50 samples/class
```

Ví dụ:

| Sau task | Seen classes | Tổng memory tối đa |
| --- | ---: | ---: |
| 0 | 38 | 1.900 samples |
| 1 | 58 | 2.900 samples |
| 7 | 178 | 8.900 samples |

Class có ít hơn 50 train samples chỉ lưu được số mẫu thực tế của class đó. Replay buffer chọn exemplar đại diện và trộn chúng với dữ liệu của task mới trong quá trình training.

Tham số này áp dụng cho cả `replay` và `replay_distill`. Đây là **per-class memory baseline**, nên total memory tăng theo số seen classes. Nó chưa phải fixed-total-memory baseline của S04.

#### `--distill-alpha 1.0`

Trọng số của logit distillation loss:

```text
distill_alpha = 1.0
```

Teacher là checkpoint tốt nhất của task trước. Student được khuyến khích giữ phân bố dự đoán của teacher trên các old classes.

Giá trị `1.0` có nghĩa thành phần logit distillation được nhân hệ số 1.0 trước khi cộng vào total loss. Điều này không có nghĩa độ lớn gradient của classification và distillation luôn bằng nhau.

Tham số chỉ có tác dụng với `replay_distill` và bắt đầu từ task 1; task 0 chưa có teacher model.

#### `--temperature 2.0`

Temperature làm mềm phân bố xác suất dùng cho knowledge distillation:

```text
teacher_probability = softmax(teacher_logits / 2.0)
student_probability = softmax(student_logits / 2.0)
```

Với `temperature=2.0`, xác suất bớt tập trung vào class có logit lớn nhất. Student có thể học thêm quan hệ tương đối giữa các old classes thay vì chỉ học hard prediction của teacher.

Trong implementation, KL loss được nhân với `temperature²` để giữ gradient scale tương đối ổn định:

```text
logit_distillation_loss = temperature² × KL(teacher || student)
```

Temperature quá thấp làm distribution gần hard label; quá cao có thể làm distribution quá phẳng.

#### `--feature-distill-weight 0.5`

Trọng số của feature distillation loss giữa latent representation của teacher và student:

```text
feature_distillation_loss = MSE(student_features, teacher_features)
```

Giá trị `0.5` giúp hạn chế backbone thay đổi quá mạnh khi học task mới, nhưng vẫn cho student đủ khả năng thích nghi.

Tham số chỉ có tác dụng với `replay_distill` từ task 1 trở đi.

#### Total training loss của `replay_distill`

Từ task 1, loss tổng có dạng:

```text
total_loss
  = focal_classification_loss
  + 1.0 × logit_distillation_loss
  + 0.5 × feature_distillation_loss
```

Trong đó:

- focal classification loss học từ dữ liệu task mới và exemplars trong replay buffer;
- logit distillation giữ cách dự đoán trên old classes;
- feature distillation giữ latent representation không thay đổi quá nhanh.

Giữ nguyên các giá trị trên cho toàn bộ 5 seeds. Nếu thay đổi một tham số, kết quả phải được lưu dưới tên experiment khác và không gộp trực tiếp với baseline này.

#### Hướng điều chỉnh và ảnh hưởng tới training

| Parameter | Khi giảm | Khi tăng | Rủi ro khi tăng quá cao |
| --- | --- | --- | --- |
| `memory-per-class` | Training nhanh hơn, ít memory hơn nhưng old-class forgetting thường tăng | Có nhiều exemplar đa dạng hơn, old classes thường được giữ tốt hơn | Tăng thời gian mỗi epoch, RAM/disk usage và có thể lặp nhiều exemplar tương tự |
| `distill-alpha` | Student linh hoạt hơn khi học class mới nhưng dễ lệch khỏi teacher | Giữ old-class predictions mạnh hơn, giảm model drift | Student bị ràng buộc quá mạnh, new-class Recall/F1 có thể giảm |
| `temperature` | Distribution sắc hơn, chủ yếu học class teacher tin nhất | Distribution mềm hơn, truyền thêm quan hệ giữa các old classes | Distribution gần uniform, tín hiệu teacher trở nên kém phân biệt |
| `feature-distill-weight` | Backbone thích nghi nhanh hơn nhưng latent representation dễ drift | Giữ latent features ổn định hơn qua tasks | Backbone khó học representation cần cho class mới |

Khoảng giá trị có thể dùng cho ablation nhỏ:

```text
memory-per-class       = 20, 50, 100
distill-alpha          = 0.5, 1.0, 2.0
temperature            = 1.0, 2.0, 4.0
feature-distill-weight = 0.0, 0.25, 0.5, 1.0
```

Không nên chạy toàn bộ tích Descartes của các giá trị trên. Chỉ thay **một parameter mỗi lần**, chạy seed 0 trước, sau đó mới mở rộng cấu hình tốt sang 5 seeds.

##### Khi old-class forgetting cao

Điều chỉnh theo thứ tự:

1. tăng `memory-per-class`, ví dụ từ 50 lên 100;
2. nếu memory đã đủ, tăng nhẹ `distill-alpha`, ví dụ từ 1.0 lên 2.0;
3. nếu latent features thay đổi mạnh, tăng `feature-distill-weight`, ví dụ từ 0.5 lên 1.0.

Việc tăng memory không giúp thêm cho class có tổng số train samples thấp hơn giới hạn hiện tại; ví dụ class chỉ có 5 samples vẫn chỉ lưu được tối đa 5 exemplars.

Theo dõi:

- old-class F1;
- mean forgetting;
- số class có F1 bằng 0;
- new-class F1 để chắc chắn model không bị quá ổn định.

##### Khi new-class performance thấp

Model có thể đang bị teacher hoặc old features ràng buộc quá mạnh. Thử:

1. giảm `distill-alpha` từ 1.0 xuống 0.5;
2. giảm `feature-distill-weight` từ 0.5 xuống 0.25 hoặc 0;
3. nếu replay chiếm quá nhiều thời gian hoặc ảnh hưởng learning balance, thử giảm `memory-per-class`.

Theo dõi đồng thời old-class forgetting. New-class F1 tăng nhưng forgetting tăng mạnh không phải là cải thiện tổng thể.

##### Khi teacher distribution quá sắc hoặc quá phẳng

- Nếu teacher gần như chỉ đưa xác suất cho một class, tăng `temperature` từ 1.0 lên 2.0 hoặc 4.0.
- Nếu probabilities trở nên gần uniform và distillation không còn phân biệt class, giảm `temperature`.
- Do implementation đã nhân KL loss với `temperature²`, thay đổi temperature chủ yếu điều chỉnh độ mềm của distribution; nó không thay thế vai trò của `distill-alpha`.

##### Khi training quá chậm hoặc MPS thiếu memory

Thử theo thứ tự:

1. giảm `batch-size` từ 512 xuống 256;
2. giảm `memory-per-class` nếu thời gian mỗi epoch vẫn quá lớn;
3. giữ cùng batch size và memory setting cho mọi method/seed trong bảng so sánh chính.

Không giảm `epochs` chỉ để xử lý thiếu memory. `epochs` ảnh hưởng số bước tối đa và early stopping behavior, còn `batch-size` và replay memory tác động trực tiếp hơn tới tài nguyên.

##### Cách đọc trade-off

Một cấu hình tốt không chỉ có final Macro-F1 cao. Nên ưu tiên cấu hình đạt cân bằng giữa:

```text
old-class F1 cao
+ forgetting thấp
+ new-class F1 không giảm mạnh
+ training cost chấp nhận được
```

Khi lưu ablation, dùng output directory thể hiện rõ parameter, ví dụ:

```text
outputs/runs_s01_ablation/seed0_replay_distill_mem100_alpha1_temp2_feat05
```

## 5. Theo dõi một run

Ví dụ theo dõi `replay_distill`, seed 0:

```bash
tail -f outputs/runs_s01_locked/random_seed0_replay_distill_mlpbase/train.log
```

Một run hoàn thành phải đi qua `task_id=0` đến `task_id=7`.

Không chạy hai process MPS cùng lúc. Chạy tuần tự sẽ ổn định hơn và tránh hết unified memory.

## 6. Kiểm tra kết quả

Sau khi chạy xong, số file trajectory và forgetting đều phải bằng 20:

```bash
find outputs/runs_s01_locked -name class_trajectory.csv | wc -l
find outputs/runs_s01_locked -name class_forgetting.csv | wc -l
```

Kết quả mong đợi:

```text
20
20
```

Kiểm tra schema, số dòng và duplicate key của toàn bộ runs:

```bash
.venv/bin/python - <<'PY'
import json
from pathlib import Path

import pandas as pd

root = Path("outputs/runs_s01_locked")
trajectory_files = sorted(root.glob("*/class_trajectory.csv"))

if len(trajectory_files) != 20:
    raise SystemExit(f"Expected 20 runs, found {len(trajectory_files)}")

for path in trajectory_files:
    frame = pd.read_csv(path)
    keys = ["seed", "method", "train_task", "class_id"]

    if len(frame) != 864:
        raise SystemExit(f"{path}: expected 864 rows, found {len(frame)}")
    if frame.duplicated(keys).any():
        raise SystemExit(f"{path}: duplicated trajectory keys")
    if frame.loc[frame["train_task"] == 7, "class_id"].nunique() != 178:
        raise SystemExit(f"{path}: final task does not contain 178 classes")

    run_dir = path.parent
    events = pd.read_csv(run_dir / "events.csv")
    event_counts = events["event_type"].value_counts()
    expected_counts = {
        "run_started": 1,
        "task_started": 8,
        "training_protocol": 8,
        "task_completed": 8,
        "run_completed": 1,
    }
    for event_type, expected in expected_counts.items():
        actual = int(event_counts.get(event_type, 0))
        if actual != expected:
            raise SystemExit(f"{run_dir}: {event_type} expected {expected}, found {actual}")
    if events["run_id"].nunique() != 1:
        raise SystemExit(f"{run_dir}: events contain more than one run_id")

    config = json.loads((run_dir / "run_config.json").read_text())
    if config["run_id"] != events["run_id"].iloc[0]:
        raise SystemExit(f"{run_dir}: config/event run_id mismatch")
    audit = pd.read_csv(run_dir / "training_audit.csv")
    if len(audit) != 8 or audit["task"].tolist() != list(range(8)):
        raise SystemExit(f"{run_dir}: invalid training audit")

print("All 20 S01 runs are valid.")
PY
```

Mỗi run directory cần có ít nhất:

```text
class_trajectory.csv
class_forgetting.csv
metrics.csv
task_matrix.csv
forgetting.csv
run_summary.md
run_config.json
training_audit.csv
train.log
checkpoints/
```

### Expected results

Khi một run hoàn thành đủ 8 tasks, kết quả cấu trúc mong đợi là:

| File | Kết quả mong đợi cho một run |
| --- | --- |
| `class_trajectory.csv` | 864 dòng |
| `class_forgetting.csv` | 864 dòng |
| `metrics.csv` | 44 dòng gồm task-group và seen-all evaluations |
| `task_matrix.csv` | 8 dòng, tương ứng 8 train tasks |
| `forgetting.csv` | 9 dòng: 8 tasks và `mean_old_tasks` |
| `run_config.json` | Một run ID, full CLI arguments, git state và resolved protocol |
| `training_audit.csv` | 8 dòng audit sampler, memory và optimizer steps |
| `seen_class_map_task_*.json` | 8 files |
| `checkpoints/task_*_model.pt` | 8 checkpoints |

Số class trong trajectory sau từng task phải là:

| `train_task` | Số seen classes | Số dòng tại task đó |
| --- | ---: | ---: |
| 0 | 38 | 38 |
| 1 | 58 | 58 |
| 2 | 78 | 78 |
| 3 | 98 | 98 |
| 4 | 118 | 118 |
| 5 | 138 | 138 |
| 6 | 158 | 158 |
| 7 | 178 | 178 |
| **Tổng** |  | **864** |

Với 20 main runs, tổng kết quả mong đợi là:

```text
20 run directories
20 class_trajectory.csv files
20 class_forgetting.csv files
17,280 class trajectory rows nếu gộp tất cả runs
17,280 class forgetting rows nếu gộp tất cả runs
```

Các điều kiện dữ liệu cần đúng:

- khóa `(seed, method, train_task, class_id)` là duy nhất;
- mỗi class xuất hiện từ `first_task` của nó đến task 7;
- `train_count` của một class không thay đổi giữa các tasks;
- full dataset hiện có `train_count` từ 3 đến 73.634;
- `test_count` phải lớn hơn 0 trong full run;
- precision, recall, F1 và confidence nằm trong `[0, 1]`;
- entropy không âm và không lớn hơn `log(số seen classes)` ngoài sai số số thực nhỏ;
- `best_f1 >= current_f1` và forgetting không âm;
- forgetting tại lần đầu class xuất hiện phải bằng 0.

Xu hướng hiệu năng mong đợi:

- `sequential` thường có forgetting cao nhất;
- `replay` và `replay_distill` thường giữ old classes tốt hơn `sequential`;
- `joint_seen` là oracle baseline và thường cho hiệu năng tốt nhất hoặc gần tốt nhất;
- `replay_distill` không bắt buộc thắng `replay` ở mọi seed, nên kết luận phải dựa trên mean ± std của 5 seeds;
- EWC thường yếu hơn replay-based methods trong repo hiện tại.

Không đặt trước một giá trị Macro-F1 cụ thể cho từng run. Giá trị chính xác phụ thuộc method, seed và quá trình tối ưu; dấu hiệu quan trọng là output đúng schema và xu hướng tổng hợp qua 5 seeds hợp lý. S01 chưa bao gồm calibrated confidence, vì vậy không dùng confidence/entropy hiện tại để kết luận calibration.

Kết quả thực tế của full run đã đạt toàn bộ structural expectations trong phần này. Báo cáo mean ± std và nhận xét được lưu tại [S01 full experiment results](../../results/s01_results.md).

## 7. Chạy EWC bổ sung

Chỉ chạy phần này sau khi 20 main runs đã hoàn tất:

```bash
set -eo pipefail

S01_PYTHON=".venv/bin/python"
S01_OUT_ROOT="outputs/runs_s01_locked"
S01_SEEDS=(0 1 2 3 4)

export PYTORCH_ENABLE_MPS_FALLBACK=1

for seed in "${S01_SEEDS[@]}"; do
  "${S01_PYTHON}" src/training/train_cil.py \
    --train train_extracted.parquet \
    --validation validation_extracted.parquet \
    --test test_extracted.parquet \
    --feature-cols outputs/audit/feature_columns.json \
    --scaler outputs/preprocess/scaler.pkl \
    --task-file "outputs/tasks/random_seed${seed}_tasks.json" \
    --outdir "${S01_OUT_ROOT}/random_seed${seed}_ewc_mlpbase" \
    --method ewc \
    --variant base \
    --batch-size 512 \
    --epochs 20 \
    --patience 5 \
    --seed "${seed}" \
    --device mps \
    --ewc-lambda 1000.0
done
```

## 8. Lưu ý

- Mỗi run phải dùng một output directory riêng.
- Pipeline fail-fast nếu output directory đã có bất kỳ nội dung nào; không append hoặc resume tại chỗ.
- Không dùng output từ smoke run để phân tích chính thức.
- Không commit checkpoints, CSV results hoặc raw data vào Git.
- Nếu một run bị dừng, nên chuyển directory cũ sang tên backup rồi chạy lại bằng directory sạch.
- `class_trajectory.csv` và `class_forgetting.csv` được cập nhật sau mỗi task, nên có thể dùng để kiểm tra tiến độ của partial run.
- Không bật S02 bằng cách chạy lại vào các S01 directories; exporter chỉ dành cho clean run có cùng provenance từ đầu.
