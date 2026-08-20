# P0-P8 T-DDI Protocol Guide

This guide separates the repository defaults from the locked experiment
configuration used to compare task protocols.

## Repository defaults

The existing `main` command-line defaults remain backward compatible:

- model variant: `base`
- CIL method: `sequential`
- batch size: `512`
- epochs: `20`
- learning rate: `0.001`
- weight decay: `0.0001`
- dropout: `0.2`
- activation: `gelu`
- normalization: `layernorm`
- early-stopping patience: `5`

`tddi` is an opt-in alias for the existing numerical `base` architecture
with hidden dimensions `(1024, 512)`. The defaults above do not change.

## Locked protocol-comparison configuration

`scripts/run_backbone_protocols.sh` pins the configuration used for the
P0-P8 comparison:

- model variant: explicitly selected by the caller; use `tddi` for the
  reported T-DDI comparison
- method: `replay_distill_fixed_budget_uniform`
- batch size: `1024`
- epochs: `20`
- patience: `5`
- total replay-memory budget: `6800`
- replay draws per epoch after task 0: `6800`
- training seeds: `0 1 2 3 4`
- task sizes: `38 + 7 x 20 = 178` classes

The remaining optimizer and model values use the unchanged repository
defaults: AdamW, learning rate `0.001`, weight decay `0.0001`, dropout `0.2`,
GELU, LayerNorm, focal gamma `1.0`, distillation alpha `1.0`, temperature
`2.0`, and feature-distillation weight `0.5`.

## Protocol definitions

- P0 `random`: seeded random class order.
- P1 `frequency_balanced`: distribute three frequency-rank strata across
  tasks.
- P2 `head_to_tail`: descending train frequency.
- P3 `tail_to_head`: ascending train frequency.
- P4 `constrained_mass_balanced`: preserve proportional train-frequency-bin
  quotas while balancing train sample mass.
- P5 `multi_factor_balanced`: balance sample mass, effective support, unique
  drugs, and descriptor diversity under the same rarity constraints.
- P6 `difficulty_balanced`: balance train mass and validation-derived class
  difficulty/NLL under the same rarity constraints.
- P7 `confusion_spread`: spread validation-confusable classes across tasks
  while balancing train mass and rarity composition.
- P8 `controlled_rarity_drift`: retain every rarity group in every task while
  gradually reducing head and increasing ultra-tail representation.

P0 and P4-P8 have five seeded schedules. P1-P3 use one deterministic schedule
with five independent training seeds. P0-P5 and P8 use train-only construction
signals. P6 and P7 additionally use validation predictions from a static T-DDI
reference. No protocol uses the test split for construction.

The rarity bins used by P4-P8 are:

- `ultra_tail`: at most 20 train examples
- `tail`: 21-100 train examples
- `medium`: 101-1000 train examples
- `head`: more than 1000 train examples

## Build P0-P4

```bash
.venv/bin/python scripts/build_cil_tasks.py \
  --class-counts outputs/class_distribution/class_counts_train.csv \
  --validation-counts outputs/class_distribution/class_counts_validation.csv \
  --test-counts outputs/class_distribution/class_counts_test.csv \
  --outdir outputs/tasks \
  --protocol all
```

## Build the static reference and P5-P8

Train the static reference:

```bash
.venv/bin/python src/training/train_static.py \
  --train train_extracted.parquet \
  --validation validation_extracted.parquet \
  --test test_extracted.parquet \
  --feature-cols outputs/audit/feature_columns.json \
  --scaler outputs/preprocess/scaler.pkl \
  --outdir outputs/advanced_protocols/static_tddi_reference \
  --variant tddi \
  --batch-size 1024 \
  --epochs 20 \
  --patience 5 \
  --seed 0 \
  --device auto
```

Extract train/validation signals. This step never loads the test split:

```bash
.venv/bin/python scripts/prepare_advanced_protocol_signals.py \
  --train train_extracted.parquet \
  --validation validation_extracted.parquet \
  --feature-cols outputs/audit/feature_columns.json \
  --scaler outputs/preprocess/scaler.pkl \
  --checkpoint outputs/advanced_protocols/static_tddi_reference/best_model.pt \
  --class-map outputs/advanced_protocols/static_tddi_reference/global_class_map.json \
  --outdir outputs/advanced_protocols/signals \
  --variant tddi \
  --batch-size 1024 \
  --device auto
```

Build P5-P8:

```bash
.venv/bin/python scripts/build_advanced_protocols.py \
  --class-counts outputs/class_distribution/class_counts_train.csv \
  --class-stats outputs/advanced_protocols/signals/class_protocol_stats.csv \
  --difficulty outputs/advanced_protocols/signals/validation_class_difficulty.csv \
  --confusion-edges outputs/advanced_protocols/signals/validation_confusion_edges.csv \
  --outdir outputs/tasks \
  --protocol all
```

## Run the comparison

Run one protocol with the canonical T-DDI name:

```bash
BACKBONE_DEVICE=cuda bash scripts/run_backbone_protocols.sh P4 tddi
```

Run P5-P8 sequentially in one tmux session:

```bash
tmux new-session -d -s ddi_tddi_p5_p8 \
  "cd $PWD && for protocol in P5 P6 P7 P8; do BACKBONE_DEVICE=cuda bash scripts/run_backbone_protocols.sh \$protocol tddi || exit \$?; done"
```

The runner skips complete run directories, refuses to overwrite incomplete
runs, and stops before a run when available RAM is below 12 GiB or free disk
space is below 50 GiB.
