# DDI-CIL

Class-incremental learning on the DDI2025 descriptor dataset using the original
train/validation/test split.

## Current experiment contract

- One model per backbone, protocol and seed.
- Fixed seed: `0`.
- Protocols: Ordered (`head_to_tail`) and Shuffled (`seed=0`).
- Task layout: `38 + 7 x 20 = 178` classes.
- Global all-seen classifier; no task identity is supplied at inference.
- Original train and validation sources remain independent.
- Task-0 training data is the only source used to fit the frozen scaler.
- Naive sequential baseline trains current-task samples only.

Backbones:

- T-DDI
- TabM basic (32 internal branches are part of one model)
- RealMLP-TD numerical

## Prepare data without training

```bash
cd "$HOME/DrugDrug/cil-tddi/cil-tddi"
conda activate ai_env

bash scripts/prepare_single_model_data.sh check configs/ordered_preprocessing.json
bash scripts/prepare_single_model_data.sh start configs/ordered_preprocessing.json
```

This creates the Task-0-fitted preprocessing artifact and reusable,
method-neutral task caches. It does not create a model or start training.

Status:

```bash
bash scripts/prepare_single_model_data.sh status configs/ordered_preprocessing.json
```

There is currently no training launcher. Training will be designed after the
preprocessing contract is validated. The two available configs are stored
directly under `configs/`:

- `ordered_preprocessing.json`
- `shuffled_seed0_preprocessing.json`

See [docs/SINGLE_MODEL_WORKFLOW.md](docs/SINGLE_MODEL_WORKFLOW.md) for the
preprocessing, cache and resume contracts.
