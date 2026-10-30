# Original-split preprocessing for a future single-model workflow

This is an optional preprocessing path for future experiments before continual
model merging. The intended future workflow will train one model with seed 0
on the original, independent train split, select checkpoints on the original
validation split, and report on the original test split. Train and validation
will never be merged or reassigned.

The baseline is naive sequential fine-tuning: task `t` trains only the current
task's training rows, starting from task `t-1`'s best checkpoint. It has no
replay, distillation, EWC penalty, A-GEM projection, or memory budget.

The classifier remains global over all seen classes and expands
`38 -> 58 -> ... -> 178`. Inference receives no task identity. Validation and
test at task `t` contain the all-seen prefix only.

Preprocessing uses the existing external policy. The standard scaler is fitted
once from original Task-0 training rows and remains frozen. A method-neutral
cache is created on demand per reached task; future task shards are not opened
or materialised. RealMLP-TD's learned PBLD/front-scale/SELU layers remain model
parameters and its train-dependent initialization only receives current train
rows. TabM retains its 32 internal branches; they form one model and are
independent of the retained three-fold external ensemble workflow.

## Commands

Prepare the frozen scaler and all eight method-neutral task caches without
creating or training a model:

```bash
bash scripts/prepare_single_model_data.sh start configs/ordered_preprocessing.json
```

Use `THROUGH_TASK=0` (or another reached boundary) to materialise only a task
prefix. Future task features never contribute to fitted statistics; the only
fitted transform remains the Task-0 training scaler.

No single-model training launcher is included yet. This stage prepares only
the immutable preprocessing artifact and method-neutral caches; the future
training workflow will consume them without refitting.

The three-fold training/ensemble pipeline remains available in parallel. Its
fold assignments, member-specific preprocessing, checkpoints, and offline
ensemble artifacts are separate from this original-split preprocessing path.
