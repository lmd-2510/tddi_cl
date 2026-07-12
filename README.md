# DDI-CIL

Class-Incremental Learning (CIL) for drug-drug interaction (DDI) classification on the **DDI2025** dataset — 868,069 drug pairs labeled across 178 interaction types, introduced by the T-DDI paper (Kha et al., "Robust Prediction of Drug Interactions using Chemical Descriptors", npj Digital Medicine). Each drug pair is represented by 3,780 QSAR physicochemical descriptors (MR_VSA, EState_VSA, SlogP_VSA, LabuteASA, MTPSA, PEOE_VSA, VSA_EState families) computed from SMILES via RDKit/PyBioMed.

This repo extends that dataset into a **class-incremental** setting: interaction classes are revealed across 8 sequential tasks (38 base classes + 7 increments of 20 classes each), and evaluates how well different continual-learning strategies retain accuracy on previously seen classes while learning new ones — a problem the original paper does not address (it trains on all 178 classes jointly).

## Pipeline

```
scripts/inspect_splits.py           # schema/leakage sanity checks on the raw parquet splits
scripts/analyze_class_distribution.py
scripts/check_leakage.py
scripts/preprocess_features.py      # fits scaler (+ optional PCA) on train split only
scripts/build_cil_tasks.py          # builds task schedules (random / frequency_balanced / long_tail protocols)
src/training/train_static.py        # offline baseline: all 178 classes trained jointly
src/training/train_cil.py           # class-incremental training entrypoint (all CIL methods below)
```

Run `run_smoke.sh` for a fast end-to-end sanity check on a tiny row-capped subset, or `run_full.sh` for the full pipeline.

## CIL methods (`train_cil.py --method ...`)

| Method | Idea |
|---|---|
| `sequential` | Fine-tune on each new task only — no anti-forgetting mechanism (worst-case baseline) |
| `joint_seen` | Oracle: reloads full real data for every class seen so far, every task (not a real CIL method — measures the ceiling if memory weren't constrained) |
| `replay` | Rehearsal from a bounded exemplar buffer (herding-selected, class-balanced sampling) |
| `replay_distill` | `replay` + knowledge distillation (logit KL + feature MSE) from the previous task's model |
| `ewc` | Online Elastic Weight Consolidation — regularizes toward previous-task weights, weighted by accumulated Fisher information, growing with the classifier head |

Shared components: `src/data/replay_buffer.py` (herding exemplar selection), `src/methods/ewc.py` (Fisher computation, penalty, head-growth remapping), `src/models/mlp.py` (LayerNorm + 2-layer MLP backbone, scale-free with input dimensionality), focal loss (`FocalLoss` in `train_cil.py`, matching the original T-DDI paper's loss for this exact class-imbalance problem).

## Key findings

- The offline **`static`** baseline (all classes trained jointly) reaches **macro-F1 ≈ 0.833**, closely reproducing the original T-DDI paper's own reported result (0.832) — a good sanity check that this reimplementation is faithful.
- Naive **`replay`**/**`replay_distill`** already recover most of that ceiling's balanced accuracy while cutting forgetting drastically vs. `sequential`.
- A **Weight Alignment** bias-correction step (classifier recency-bias rescaling) was tried and found to be actively harmful here — removing it recovered balanced accuracy and forgetting beyond the original baseline, at some cost to macro-F1 on rare classes (herding trades exemplar diversity for typicality).
- **EWC alone**, even after tuning its penalty strength across several orders of magnitude, stays far below replay-based methods — consistent with known limits of pure regularization in true (task-ID-free) class-incremental settings.
- **PCA dimensionality reduction** (3,780 → 555 dims, 95% variance) trains faster but does not meaningfully improve the accuracy/forgetting trade-off — independently confirming the original paper's own feature-selection ablation, which found the full descriptor set essential for long-tail performance.
- **Focal loss** (the original paper's own loss function) gives a small, seed-noise-level macro-F1 improvement over plain cross-entropy for the replay-based methods.

## Requirements

See `requirements.txt` (numpy, pandas, pyarrow, matplotlib, scikit-learn, torch). Data (`*_extracted.parquet`) is not tracked in git — see `.gitignore`.
