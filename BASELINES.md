# Baselines & Method Provenance

This file lists every baseline/method run in this project, what it measures, and which paper it comes from. See `README.md` for the pipeline overview and `outputs/runs*/` for raw per-seed logs.

## Reference (non-CIL) baselines

| Baseline | What it measures | Source | Result (this repo) |
|---|---|---|---|
| `static` | Offline: all 178 classes trained jointly in one pass — reproduces the original dataset paper's own setup, not a CIL method | Kha et al., *"Robust Prediction of Drug Interactions using Chemical Descriptors"*, npj Digital Medicine (T-DDI paper) — reports macro-F1 = 0.832 on this same DDI2025 dataset | macro-F1 = **0.8330**, bal.acc = 0.8280 (1 seed) — matches the paper's 0.832 |

## Head-to-head vs. the paper's own published numbers

Only metrics reported by **both** sides are compared below. The paper (Kha et al., T-DDI, npj Digital Medicine) never reports AUC/ROC-AUC anywhere in its text, tables, or figures — it reports Accuracy and Precision/Recall/F1 (weighted + macro) only. Our own pipeline (`train_static.py`/`train_cil.py`) doesn't compute AUC either, so AUC is excluded from this comparison rather than estimated.

Two different paper numbers exist for T-DDI itself, and it matters which one you compare against:
- **Table 1** reports 3-fold cross-validation performance on the *development* split (train+val), averaged over folds.
- **Results text** reports a single run's performance on the **held-out test split**, both without and with the Uncertainty Estimator (UE) high-confidence filter.

| Metric | Paper — T-DDI, 3-fold CV (Table 1) | Paper — T-DDI, held-out test, no UE | Paper — T-DDI, held-out test, UE high-conf (87.9% coverage) | **This repo — `static`, held-out test** |
|---|---|---|---|---|
| Accuracy | 0.9374 ± 0.0004 | 0.9434 | 0.9796 | **0.9245** |
| Weighted F1 | 0.9370 ± 0.0004 | 0.9430 | 0.9794 | **0.9241** |
| Macro Precision | 0.8828 ± 0.0031 | 0.9023 | 0.9249 | — *(not currently logged)* |
| Macro Recall | 0.8043 ± 0.0031 | 0.8185 | 0.8869 | — *(not currently logged)* |
| **Macro F1** | **0.8321 ± 0.0032** | **0.8452** | **0.8992** | **0.8330** |
| AUC / ROC-AUC | not reported | not reported | not reported | not computed |

**Reading this:** our single-model `static` reproduction (macro-F1 = 0.8330) lands almost exactly on the paper's own 3-fold CV number (0.8321) — the intended apples-to-apples comparison, since neither uses the UE high-confidence filter or the 3-model ensemble the paper uses for its headline held-out-test number (0.8452/0.8992). We do not currently log macro precision/recall separately for `static` (only accuracy, macro-F1, weighted-F1, balanced accuracy) — would need a one-line addition to `compute_classification_metrics` output if wanted.

### Full Table 1 reproduction — other baselines the paper compared against (DDI2025, 3-fold CV)

Not run in this repo; reported here for reference since they're the field context our `static`/CIL results sit inside.

| Model | Accuracy | Weighted F1 | Macro Precision | Macro Recall | Macro F1 |
|---|---|---|---|---|---|
| RandomForest | 0.8000±0.0014 | 0.7961±0.0014 | 0.7933±0.0153 | 0.6033±0.0058 | 0.6633±0.0058 |
| XGBoost | 0.7854±0.0007 | 0.7831±0.0006 | 0.6167±0.0058 | 0.7700±0.0000 | 0.6700±0.0000 |
| KNN | 0.6945±0.0009 | 0.6896±0.0011 | 0.5900±0.0100 | 0.4667±0.0058 | 0.5007±0.0093 |
| CNN | 0.6171±0.0209 | 0.6133±0.0208 | 0.5200±0.0173 | 0.2300±0.0173 | 0.2867±0.0231 |
| TabM | 0.9131±0.0011 | 0.9120±0.0012 | 0.8685±0.0075 | 0.8248±0.0045 | 0.8344±0.0036 |
| TabNet | 0.8827±0.0121 | 0.8814±0.0129 | 0.7736±0.0345 | 0.6861±0.0361 | 0.7048±0.0369 |
| FT-Transformer | 0.9184±0.0032 | 0.9181±0.0032 | 0.8195±0.0152 | 0.7575±0.0167 | 0.7733±0.0150 |
| SAINT | 0.9064±0.0025 | 0.9054±0.0019 | 0.7691±0.0156 | 0.7002±0.0325 | 0.7193±0.0013 |
| BiSHop | 0.9275±0.0015 | 0.9272±0.0016 | 0.8594±0.0068 | 0.8083±0.0101 | 0.8215±0.0036 |
| TabTransformer | 0.8329±0.0024 | 0.8274±0.0024 | 0.6951±0.0129 | 0.5981±0.0073 | 0.6168±0.0093 |
| **T-DDI (paper's own model)** | **0.9374±0.0004** | **0.9370±0.0004** | **0.8828±0.0031** | **0.8043±0.0031** | **0.8321±0.0032** |

Note: this repo's MLP backbone (`static`, `joint_seen`, and every CIL method's per-task head) is architecturally closest to the paper's own **T-DDI** row — LayerNorm + numerical-only MLP, no categorical/TabTransformer branch — not to the plain `TabTransformer` row above, which the paper found notably worse (0.6168 macro-F1) than its own numerical-only design.

## CIL reference bounds (standard in the CIL literature, not method contributions)

| Baseline | What it measures | Source | Result (5 seeds, mean±std) |
|---|---|---|---|
| `sequential` ("Finetune") | Fine-tune on each new task only, no anti-forgetting — canonical CIL lower bound | Standard CIL baseline, used e.g. in Rebuffi et al., *"iCaRL: Incremental Classifier and Representation Learning"*, CVPR 2017 | macro-F1 = 0.0406±0.0091, bal.acc = 0.1174±0.0199, forgetting = 0.8868±0.0261 |
| `joint_seen` ("Joint"/Oracle) | Reloads full real data for every class seen so far, every task — canonical CIL upper bound (measures the ceiling if memory weren't constrained) | Standard CIL upper-bound baseline, same lineage as above (iCaRL, PODNet, etc.) | macro-F1 = 0.8405±0.0067, bal.acc = 0.8327±0.0089 — essentially matches `static`, confirming the ceiling |

## CIL methods under test

| Method | Technique | Source paper | Best result so far (v5, 5 seeds) |
|---|---|---|---|
| `replay` | Bounded exemplar buffer with **herding** selection (nearest-to-class-mean) + class-balanced sampling | Herding: Rebuffi et al., *iCaRL*, CVPR 2017 | macro-F1 = 0.3316±0.0214, bal.acc = 0.7801±0.0129, forgetting = 0.2155±0.0104 |
| `replay_distill` | `replay` + knowledge distillation (logit KL + feature MSE) from the previous task's model | Distillation: Hinton et al., *"Distilling the Knowledge in a Neural Network"*, 2015; applied to CIL in Li & Hoiem, *"Learning without Forgetting"*, TPAMI 2018 | macro-F1 = 0.3105±0.0109, bal.acc = 0.7693±0.0086, forgetting = 0.1999±0.0067 |
| `ewc` | Online Elastic Weight Consolidation — penalizes drift from previous-task weights, scaled by accumulated Fisher information, head grows across tasks | Kirkpatrick et al., *"Overcoming catastrophic forgetting in neural networks"*, PNAS 2017 (original EWC); Online variant: Schwarz et al., *"Progress & Compress"*, ICML 2018 | macro-F1 = 0.0491±0.0057, bal.acc = 0.1022±0.0160, forgetting = 0.7331±0.0330 |

## Techniques tried and rejected/kept, with source

| Technique | Outcome in this repo | Source paper |
|---|---|---|
| Weight Alignment (classifier bias correction) | **Removed** — actively harmful for this feature space (regressed all 3 metrics in v2) | Zhao et al., *"Maintaining Discrimination and Fairness in Class Incremental Learning"*, CVPR 2020 |
| Focal loss | **Kept** — small, consistent gain when combined with larger replay memory (v5); this is also the loss function the original T-DDI paper itself uses for this exact 178-class imbalance problem | Lin et al., *"Focal Loss for Dense Object Detection"*, ICCV 2017; also used by Kha et al. (T-DDI) for this dataset |
| PCA dimensionality reduction (3,780→555 dims, 95% variance) | **Not adopted** — trains faster but doesn't improve accuracy/forgetting trade-off; independently confirms the T-DDI paper's own feature-selection ablation (full descriptor set needed for long-tail classes) | General technique; cross-checked against Kha et al. (T-DDI) feature-selection ablation |
| Class-balanced sampling (`WeightedRandomSampler`, inverse class-frequency) | **Kept** — standard long-tail mitigation, part of the herding+balanced-sampling default combo | Standard long-tail/imbalanced-classification technique (e.g. surveyed in Zhang et al., *"Deep Long-Tailed Learning: A Survey"*, TPAMI 2023) |

## Version timeline (what changed each round)

| Version | Config vs. previous | replay macro-F1 | replay_distill macro-F1 | ewc macro-F1 |
|---|---|---|---|---|
| v1 | Baseline: herding-free (random exemplars), memory=50 | 0.4220±0.0329 | 0.4409±0.0181 | — |
| v2 | + herding, balanced sampling, Weight Alignment, EWC, distill (all at once) | 0.2596±0.0848 (regressed) | 0.2930±0.0111 | 0.0319±0.0145 |
| v3 | Same as v2 minus Weight Alignment (root-caused as the regression's cause) | 0.2910±0.0216 | 0.2771±0.0127 | 0.0426±0.0111 |
| v4 | v3 + PCA (555-dim) | 0.3044±0.0274 | 0.2803±0.0125 | 0.0426±0.0093 |
| v5 | v3 + memory-per-class 150→300 + focal loss | **0.3316±0.0214** | **0.3105±0.0109** | 0.0491±0.0057 |

Note: v1 has higher raw macro-F1 than v3–v5 because random exemplar selection happens to favor rare-class recall over typicality; v3–v5 (herding) trade some of that for materially better balanced accuracy and forgetting (e.g. replay: bal.acc 0.728→0.780, forgetting 0.356→0.216 from v1→v5). See `README.md` § Key findings for the full discussion.
