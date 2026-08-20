# v6: Expanded CIL method survey — backbone, scenario, method list, new metrics

Status: **DRAFT — awaiting confirmation before any code changes.** This consolidates everything discussed and agreed in the "discuss → write down → implement" round following the v5 report. Nothing in this document has been implemented yet.

## 1. Backbone

**Decision (confirmed):** TabTransformer, numerical-only variant — self-attention applied directly over the 3,780 continuous QSAR descriptors (each descriptor treated as a token), no SMILES/categorical embedding branch, no CLS token. This mirrors our data (100% numerical, no categorical/SMILES fields retained post-preprocessing) and is the fairest apples-to-apples swap against the current LayerNorm+MLP backbone.

Rejected alternative: the paper's full TabTransformer config (CLS token + SMILES embeddings + categorical embeddings) — would require a new SMILES-tokenization/categorical-binning pipeline we don't have, and the paper's own ablation (Table 2) already shows this config is *worse* than numerical-only (Macro F1 0.7022 vs 0.8452 on held-out test).

Library: `tab-transformer-pytorch` (same package the paper's Code Availability section cites, version 0.4.2) — install and wrap its numerical-only path, or hand-roll a minimal transformer-encoder-over-tokens if the library's API doesn't cleanly support "numerical-only, no categorical input" (needs a quick spike to confirm before committing).

## 2. Evaluation scope

**Decision (confirmed):** 1 seed per method (not 5), evaluated on the final held-out test set after the last task (`test_seen_all` in current code) — same simplification logic as prior single-seed ablations (e.g. the memory-per-class 150/300/500 sweep before v5). Full 5-seed reruns are reserved for whichever methods look promising after this pass.

Note: the per-task result matrix (`result_matrix[task_id, eval_task_id]`, already computed every task in the current `train_cil.py` loop) is **kept** even under "1 seed only" — it's needed for the new BWT/FWT/Average-Incremental-Accuracy metrics (§5) and costs no extra runs, just extra bookkeeping already present in the codebase.

## 3. Scenario (task-construction protocol)

`scripts/build_cil_tasks.py` supports three protocols today:

| Protocol | Status | Note |
|---|---|---|
| `random` | **Run** — primary, comparable to all v1–v5 results | 38 base + 7×20 incremental, random class order |
| `frequency_balanced` | **Run** | Existing code, not yet exercised in any prior round |
| `long_tail` | **Run** | Existing code, not yet exercised in any prior round |
| `temporal` (approval-date ordered) | **Blocked** | No approval-date metadata for any drug in `train/validation/test_extracted.parquet` (only `drugid`, `drugname`, `drugsmiles`, descriptors, `class`). Would need an external DrugBank/FDA date lookup per drug — not started. |
| `mechanism-based` (PK/PD semantic grouping) | **Blocked** | Paper's Supplementary Table 4 (178 types → 30 semantic groups) is not present in `reference/tddi_npj_digit_medicine.pdf` as fetched (only main text, no supplementary file) and not elsewhere in the repo. No DDI-type text descriptions survive preprocessing (labels are already integer-encoded 0–177 with no lookup table). Would need either the paper's supplementary file or to hand-build a mapping — not started. |

**Open item:** if the user has the paper's supplementary materials or a DrugBank export with approval dates, both blocked scenarios become tractable — flag if available.

## 4. Method list — organized into parallel execution streams

All 23 methods surveyed across three literature-search passes (vision-CIL lightweight/exemplar-free techniques; newest 2025–2026 follow-ups; cross-domain non-vision techniques). Grouped by shared infrastructure so streams can be built and run independently.

### Stream A — Gradient-based, shared training loop (extends current `train_cil.py`)

Reuses the existing per-task train/eval loop; differs only in loss terms / buffer contents.

| # | Method | Mechanism | Status here |
|---|---|---|---|
| 1 | `sequential` | No anti-forgetting (floor) | Already implemented |
| 2 | `joint_seen` | Full-data oracle (ceiling) | Already implemented |
| 3 | `replay` | Herding buffer + balanced sampling | Already implemented |
| 4 | `replay_distill` | + logit/feature KD from previous model | Already implemented |
| 5 | `ewc` | Online EWC, Fisher-weighted penalty | Already implemented |
| 6 | `iCaRL` (proper) | Herding buffer (already have) **+ Nearest-Mean-of-Exemplars classifier at inference** — the actual distinguishing piece vs. our current `replay` | New: NME inference head |
| 7 | `LwF` | Distillation from previous-task model on **new-task data only**, no buffer | New: KD without buffer |
| 8 | `DER` (Dark Experience Replay) | Reservoir buffer storing **logits at write-time** (not labels); loss = CE(current task) + MSE(stored logits vs. current-model logits on buffer) | New: logit-storing buffer, reservoir sampling |
| 9 | `BiC` (Bias Correction) | Post-task linear bias-correction layer fit on a held-out balanced old/new validation slice | New: bias-correction layer + validation-slice split (reuse `validation_extracted.parquet`) |

### Stream B — Analytic / closed-form classifier heads (frozen backbone after task 1, no backprop)

Shared infra: one frozen feature extractor + a running matrix (covariance or correlation) updated per task in closed form.

| # | Method | Mechanism |
|---|---|---|
| 10 | `SLDA` | Shared covariance + running per-class mean, Gaussian LDA decision rule, O(1) update/sample — simplest, build first |
| 11 | `FeCAM` | Per-class covariance (with shrinkage + power-transform), Mahalanobis-distance NCM classifier |
| 12 | `ACIL` / `G-ACIL` / `F-OAL` | Classifier head = recursive closed-form ridge regression, provably ≡ joint retraining |
| 13 | `AOCIL` | Same analytic-learning family, tuned for per-sample (streaming) updates |
| 14 | `AnaCP` | ACIL + a contrastive-projection step before the analytic head, explicitly targets near-`joint_seen` accuracy |
| 15 | `Class-balanced analytic ridge regression` | Bakes per-class reweighting into the closed-form ridge solve itself — most directly aimed at this project's core imbalance problem |

### Stream C — Prototype / pseudo-rehearsal methods

Shared infra: store per-class prototype statistics (not raw exemplars), synthesize pseudo-samples for replay.

| # | Method | Mechanism |
|---|---|---|
| 16 | `PASS` | 1 prototype/old class, Gaussian-jitter pseudo-features + self-supervised auxiliary task (rotation-prediction in the original paper — **needs a tabular substitute**, e.g. feature-masking/reconstruction pretext) |
| 17 | `CEFCIL` | Ensemble of NCM/Mahalanobis classifiers over diversified backbone "views" (image-augmentation views → **needs tabular substitute**, e.g. feature-subset/dropout-mask views) |
| 18 | `Manifold-aware boundary sampling` (arxiv 2606.05695) | PASS-family; samples pseudo-exemplars near decision boundaries on the feature manifold + adaptive class-balanced loss |
| 19 | `Exemplar-free discriminative-structure preservation` (CVPR 2026) | Regularizes feature-space discriminability directly instead of storing prototypes — mechanism only known at abstract level, needs more reading before implementation |
| 20 | `Prototype Latent World Model Replay` (arxiv 2606.29465) | Per-class prototype distributions (mean+variance) in MLP latent space, sampled for replay — no frozen ImageNet encoder needed since our input is already numerical |
| 21 | **`TRIL3`** (arxiv 2407.09039) | **Tabular-native** (only technique in the entire survey validated on tabular, not vision, data). XuILVQ prototype-based generative model synthesizes pseudo-samples for old classes; DNDF (differentiable decision forest) as incremental classifier. **Highest-confidence port** — build early as a sanity check that the whole "prototype/pseudo-rehearsal" family works on this data before investing in the vision-derived variants (#16–20) |

### Stream D — Reservoir / random-feature methods

| # | Method | Mechanism |
|---|---|---|
| 22 | `CIRCLE` | Fixed random-feature reservoir (never trained) + ensembled SLDA heads across multiple reservoir instantiations |

### Reference-only (not a method to run, a benchmark to mine)

| # | Item | Use |
|---|---|---|
| 23 | `TSCIL` (arxiv 2402.12035, `zqiao11/TSCIL`) | Open time-series CIL benchmark suite — mine its baseline implementations for additional architecture-agnostic method ideas / sanity-check our own metric code against a known-good implementation |

**Not pursued further:** In-Context Large Tabular Models (needs a pretrained tabular foundation model we don't have) and NLP continual-relation-extraction methods (mechanistically redundant with PASS/FeCAM already in the list).

## 5. New CL-specific metrics

In addition to the existing macro-F1 / balanced accuracy / forgetting:

- **BWT (Backward Transfer)** — signed effect of learning new tasks on old-task accuracy (Lopez-Paz & Ranzato, GEM, 2017). Computable directly from the existing per-task result matrix.
- **FWT (Forward Transfer)** — how much prior learning accelerates learning a new task vs. a random-init reference. Requires one additional reference run per scenario (random-init baseline) to subtract against.
- **Average Incremental Accuracy** — mean of the "accuracy right after task i" curve across all i, not just the final number. Also derivable from the existing result matrix.

Implementation note: BWT and Average Incremental Accuracy need no new runs (derivable from data already being collected). FWT needs one extra reference model per scenario.

## 6. Open questions before implementation starts

1. Confirm `tab-transformer-pytorch` numerical-only path works as expected (quick spike) before committing Stream A/B/C/D to it as the shared backbone.
2. Streams B/C/D assume a "frozen backbone after task 1" — decide whether that frozen backbone is (a) the TabTransformer trained during task 0 only and then frozen, or (b) a separately pretrained backbone (e.g. from `static`). This changes what "task 1" means for methods 10–22.
3. Confirm build order across streams — suggested lowest-risk-first sequence: Stream B method `SLDA` (simplest, validates the frozen-backbone infra) → Stream C method `TRIL3` (only tabular-native technique, validates prototype infra) → Stream A methods 6–9 (extends already-working training loop) → remaining Stream B/C/D methods.
4. Scenario execution order: `random` first (directly comparable to v1–v5), then `frequency_balanced`/`long_tail`.
5. `temporal`/`mechanism-based` scenarios stay blocked pending data — confirm whether to drop them or whether the user has access to the missing data.

## 7. Execution flow — scenario → backbone → benchmark methods → predicted weaknesses/pending → post-run CIL-metric reporting

### 7.1 Scenario (task-construction protocol) — run list

| # | Scenario name | Status |
|---|---|---|
| S1 | `random` | Run — primary |
| S2 | `frequency_balanced` | Run |
| S3 | `long_tail` | Run |
| S4 | `temporal` | Blocked — no approval-date data |
| S5 | `mechanism-based` | Blocked — no PK/PD semantic-group mapping available |

### 7.2 Backbone

**TabTransformer, numerical-only** (self-attention over the 3,780 QSAR descriptors as tokens, no categorical/SMILES/CLS branch) — see §1. Applies to every method below; Stream B/C/D methods freeze it after an initial training phase (open question §6.2), Stream A methods keep training it every task.

### 7.3 Benchmark methods — full run list (22 methods, 4 streams)

| Stream | Method codenames |
|---|---|
| A (gradient, trainable backbone) | `sequential`, `joint_seen`, `replay`, `replay_distill`, `ewc`, `iCaRL`, `LwF`, `DER`, `BiC` |
| B (analytic/closed-form, frozen backbone) | `SLDA`, `FeCAM`, `ACIL`, `AOCIL`, `AnaCP`, `class_balanced_ridge` |
| C (prototype/pseudo-rehearsal) | `PASS`, `CEFCIL`, `manifold_boundary`, `discriminative_structure`, `proto_latent_replay`, `TRIL3` |
| D (reservoir) | `CIRCLE` |

22 methods × up to 3 runnable scenarios (S1–S3) × 1 seed = up to 66 runs for the full matrix. `sequential`/`joint_seen` don't need re-running per backbone if the TabTransformer swap is confirmed not to change their role as floor/ceiling references — TBD once §6.1/§6.2 spikes land.

### 7.4 Predicted weaknesses & pending items per method (before running anything)

Predictions below are hypotheses to check against actual results, not settled conclusions — flagged explicitly as "predicted" since none of Stream B/C/D has been run on this data yet.

| Method | Predicted weak point on this data | Pending before it can run |
|---|---|---|
| `sequential` | None new — already-confirmed floor from v1–v5 | None — reference only |
| `joint_seen` | None new — already-confirmed ceiling from v1–v5 | None — reference only |
| `replay` | Already characterized (v1–v5): trades macro-F1 for bal-acc/forgetting via herding | Re-run only needed to isolate TabTransformer-vs-MLP backbone effect |
| `replay_distill` | Same as above | Same as above |
| `ewc` | **Predicted to stay weak** — regularization-only, doesn't address 178-class recency bias, consistent with v1–v5 finding and general Class-IL literature consensus (§ Part 2 of `CIL_TECHNIQUES_REFERENCE.md`) | None — infra already exists |
| `iCaRL` (NME) | NME may underperform if any DDI class is multi-modal in feature space (e.g. a class covering several distinct interaction mechanisms lumped under one label) — unverified until per-class analysis is run | Implement NME inference head |
| `LwF` | No buffer at all — predicted to forget faster than `replay` as task count grows (8 tasks is a relatively long chain for buffer-free distillation) | Implement no-buffer KD loop |
| `DER` | Buffer must store logits at write-time; logit vector width changes as classes are added — remapping bug risk similar to the EWC head-growth bug hit in v2 | Implement logit-storing reservoir buffer + remap logic |
| `BiC` | Needs a held-out balanced old/new validation slice distinct from the training buffer — carving this out of an already-small held-out validation split risks starving both the main validation loop and the bias-correction fit | Decide validation-split carve-out strategy before implementing |
| `SLDA` | Shared-covariance assumption may not hold across 178 pharmacologically heterogeneous interaction classes — **primary open question this method is meant to answer** | Confirm frozen-backbone definition (§6.2) |
| `FeCAM` | Per-class covariance estimation may be unstable for the ~121 DDI types with <0.1% of samples (per README's own imbalance note) even with shrinkage | Same as `SLDA` |
| `ACIL`/`AOCIL` | Closed-form ridge assumes a linear relationship between frozen features and class posterior — may underfit if TabTransformer's attention features need a nonlinear head to separate 178 classes well | Same as `SLDA`, plus recursive-update implementation |
| `AnaCP` | Least-verified mechanism in the survey (only abstract-level detail) — real implementation risk of misreading the contrastive-projection step | Needs full-paper read before implementation, not just abstract |
| `class_balanced_ridge` | Same epistemic caveat as `AnaCP` | Needs full-paper read |
| `PASS` | Single-Gaussian-per-class prototype may be a poor fit for the same multi-modal-class risk flagged under `iCaRL`; SSL auxiliary task needs a tabular substitute with no validated precedent here | Design + validate tabular SSL pretext task before implementing |
| `CEFCIL` | "Diversified views" mechanism is image-augmentation-native; tabular substitute (feature-subset/dropout views) is unverified to produce useful diversity | Design tabular view-diversification scheme |
| `manifold_boundary` | Manifold/boundary estimation cost may be nontrivial at 3,780-dim (pre-attention) or attention-output feature width — could be the slowest method to run per task | Needs full-paper read; feasibility spike on estimation cost |
| `discriminative_structure` | Mechanism understood only at abstract level — real implementation risk highest in this list | Needs full-paper read before any implementation attempt |
| `proto_latent_replay` | Same multi-modal-class risk as `PASS`, mitigated somewhat by storing variance not just mean | None beyond standard prototype-buffer infra |
| `TRIL3` | Two nonstandard components (XuILVQ generator, DNDF classifier) — highest implementation effort in Stream C despite best modality match | Implement XuILVQ + DNDF from the source paper's spec |
| `CIRCLE` | Untrained random-feature backbone is a strictly weaker representation than the trained TabTransformer used everywhere else — predicted to be the weakest-accuracy method in the whole matrix, included as a compute-efficiency reference point, not an accuracy contender | Confirm ensemble size / reservoir dimensionality before running |

### 7.5 Post-run comparison & CIL-metric reporting

Once scenarios × methods have run, report per (scenario, method) cell on:

- **Existing metrics:** macro-F1, balanced accuracy, forgetting (as in all v1–v5 reports)
- **New CIL-specific metrics (§5):** BWT, FWT, Average Incremental Accuracy
- **Cross-cutting comparison:** backbone effect (TabTransformer vs. current MLP, on the `random` scenario methods that overlap with v1–v5) isolated separately from method-choice effect, so the two don't get conflated in the final writeup
- **Predicted-vs-actual:** explicitly revisit every prediction in §7.4 against the real result — this is the payoff of writing predictions down before running anything
