# Class-Incremental Learning for Multi-Class Tabular Data — Technical Reference

> **Trạng thái:** tài liệu nền về CIL, không phải cấu hình chạy hiện tại. Study chính
> dùng `tddi_paper_member` × `replay_distill_fixed_budget_uniform` × P3 × ba member;
> xem `README.md` và `docs/TDDI_PAPER_REPLAY_DISTILL_P3_8TASK_GPU_RUNBOOK.md`.

**Scope of this document.** This is written as a self-contained study reference. Assume the reader knows only this: *we have multi-class tabular data (fixed-length numerical feature vectors, no images/text/graphs), and new classes arrive over time — the model must learn them without forgetting old ones, and without unbounded growth in stored data.* No other project context is assumed here. Part 1 is a neutral technical survey of 23 techniques, organized into 4 implementation "streams" plus one reference benchmark. Part 2 is my own opinionated technical recommendation, reasoned from that same bare assumption — not from any specific dataset's quirks.

---

## Problem framing

Class-incremental learning (CIL) is the hardest of the three standard continual-learning scenarios:

- **Task-incremental (Task-IL):** the model is told *which* task/class-subset it's being evaluated on at inference time. Easiest — effectively reduces to training separate small classifiers.
- **Domain-incremental (Domain-IL):** the class set is fixed, but the input distribution shifts over time.
- **Class-incremental (Class-IL / CIL):** the class set grows over time, and at inference the model must discriminate among **all classes seen so far, without being told which task a sample came from**. This is what's assumed throughout this document, since it's the realistic setting for "new categories appear over time."

The central failure mode is **catastrophic forgetting**: gradient updates for new classes overwrite the weights that encoded old classes, and — specific to CIL — the growing output layer also develops a **recency bias** (new-class logits are systematically larger, because they were trained more recently and more often relative to old classes sitting in a bounded or absent replay buffer).

Techniques for CIL split along two independent axes:

1. **Exemplar-based vs. exemplar-free.** Does the method store any real samples (or approximations of them) from old classes?
2. **Trainable vs. frozen backbone after early tasks.** Does the feature extractor keep being updated by gradient descent every task, or is it frozen (often after the first task) with only a lightweight classifier head updated afterward?

Every technique below is placed in one of four streams based on shared implementation infrastructure, which follows mostly from these two axes.

---

# Part 1 — Technique-by-technique technical analysis

## Stream A — Gradient-based methods with a trainable backbone

These all share one shape: a standard train-loop where the backbone gets gradient updates every task; they differ in what extra loss terms or buffer contents are used to resist forgetting.

### A1. Sequential fine-tuning ("Finetune")
**Mechanism:** Train on each new task's data only, with no anti-forgetting mechanism at all — plain cross-entropy on the current task, gradient descent as usual.
**Requirements:** Nothing beyond a standard classifier.
**Strengths:** Trivial to implement; establishes the empirical floor every other method should beat.
**Weaknesses:** Catastrophic forgetting is severe and expected — old-class accuracy typically collapses toward zero within 1-2 subsequent tasks in class-IL settings.
**Role:** Always include as the lower-bound reference, never as a candidate "real" method.

### A2. Joint / "Joint-seen" training (oracle)
**Mechanism:** At every task boundary, retrain (or continue training) on the **full real dataset** for every class seen so far, not just the new task's data.
**Requirements:** Unbounded storage/access to all historical raw data — which is precisely what CIL is trying to avoid needing.
**Strengths:** Establishes the empirical ceiling — what accuracy would be achievable with zero memory constraint.
**Weaknesses:** Not a CIL method by any reasonable definition (it doesn't solve the problem, it sidesteps it) — only useful as a reference bound.
**Role:** Always include as the upper-bound reference, paired with A1 as the floor.

### A3. Replay with herding (iCaRL-style buffer)
**Mechanism:** Rebuffi et al., *iCaRL*, CVPR 2017. Maintain a fixed-size buffer of exemplars per old class. When a class must be pruned down to the buffer budget, **herding selection** picks the exemplars whose feature-space mean is closest to the true class mean (nearest-to-class-mean via iterative greedy selection), rather than random sampling — this keeps the buffer's centroid faithful to the true class centroid even under heavy compression. At train time, buffer exemplars are mixed with new-task data (often with class-balanced sampling, since new classes vastly outnumber old-class buffer samples).
**Requirements:** O(buffer_size × num_classes × feature_dim) storage for raw exemplar features; a class-mean computation pass per task.
**Strengths:** Simple, robust, one of the most consistently strong rehearsal baselines across CIL benchmarks generally.
**Weaknesses:** Buffer size scales linearly with number of classes seen — eventually becomes a real memory bottleneck. Herding selection favors "typical" samples over rare/edge-case ones, which can hurt worst-class performance even while helping average performance.

### A4. Replay + Distillation
**Mechanism:** A3's buffer, plus knowledge distillation (Hinton et al. 2015) from the previous task's frozen model — typically a KL term matching old-class logit distributions between student and teacher, sometimes with an added feature-space MSE term matching intermediate representations. Rooted in Li & Hoiem's *Learning without Forgetting* (TPAMI 2018) for the distillation piece.
**Requirements:** Keep the previous task's model in memory (or on disk) as a frozen teacher.
**Strengths:** Distillation regularizes the decision boundary shape, not just the raw logit magnitudes — often reduces forgetting further than replay alone, especially on balanced-accuracy-style metrics.
**Weaknesses:** Extra hyperparameters (distillation temperature, loss weight) that need tuning; diminishing returns once the buffer is already large enough to anchor old classes well.

### A5. EWC (Elastic Weight Consolidation) — Online variant
**Mechanism:** Kirkpatrick et al., PNAS 2017 (original); Schwarz et al., *Progress & Compress*, ICML 2018 (Online EWC, the practical variant used in almost all modern implementations). Computes the **Fisher information matrix diagonal** — an estimate of how sensitive the loss is to each parameter — after each task, and adds a quadratic penalty `λ · Σ F_i (θ_i − θ*_i)²` to the loss for subsequent tasks, discouraging drift in parameters that mattered for old tasks. "Online" means a single running Fisher estimate is maintained and updated additively across tasks, rather than storing a separate Fisher matrix per task (which would grow unboundedly).
**Requirements:** No exemplar storage at all — purely a regularization term. Needs an extra backward-pass-equivalent computation (squared gradients) per task to estimate Fisher.
**Strengths:** Zero raw data retention — attractive when storage or privacy is a hard constraint.
**Weaknesses:** **Well-documented in the literature to underperform badly in true class-incremental settings with many classes** — regularizing weights alone cannot prevent the output layer's recency bias, since nothing in EWC's objective corrects for the old-vs-new logit-scale imbalance. Consistently one of the weakest method families in class-IL benchmarks (though it does much better in task-IL, where task-ID at inference sidesteps the recency-bias problem entirely).

### A6. iCaRL, proper (with Nearest-Mean-of-Exemplars classification)
**Mechanism:** Identical buffer/herding mechanics to A3, but the actual iCaRL paper's distinguishing contribution is **not** herding — it's classifying via **nearest-mean-of-exemplars (NME)** at inference: compute each class's exemplar-set mean in feature space, and classify a new sample by nearest mean (Euclidean or cosine), rather than using the trained softmax head at all. The softmax head is only used to drive representation learning during training; at test time it's discarded in favor of NME.
**Requirements:** Same as A3, plus recomputing exemplar-set means after each task.
**Strengths:** NME classification is inherently immune to the softmax head's recency-bias problem, since it never uses the head's output layer weights at inference.
**Weaknesses:** Throws away the (often useful) softmax head's discriminative calibration; NME can underperform when class-conditional feature distributions are not well-approximated by a single mean (multi-modal or high-variance classes).

### A7. LwF (Learning without Forgetting)
**Mechanism:** Li & Hoiem, TPAMI 2018. The purest distillation-only method: **no buffer at all**. When training on new-task data, additionally compute the previous model's soft predictions on that *same new-task data* for old classes, and add a distillation loss encouraging the current model to reproduce those old-class predictions — using only new-task inputs, never storing any old-task samples.
**Requirements:** Keep the previous task's model as a frozen teacher; no exemplar storage.
**Strengths:** Zero raw-data retention (privacy-friendly, like EWC), but empirically stronger than pure regularization methods in many benchmarks since it directly regularizes output behavior, not just weight values.
**Weaknesses:** Relies on new-task inputs being distributionally similar enough to old-task inputs that distilling on them transfers meaningfully to old-class boundaries — this assumption weakens as more tasks accumulate and the input distribution drifts further from early tasks.

### A8. DER (Dark Experience Replay)
**Mechanism:** Buzzega et al., NeurIPS 2020. Maintains a **reservoir-sampled** buffer (uniform-probability sampling over the entire stream seen so far, not per-class herding) that stores each buffered sample's **logits at the moment it was written** (the "dark knowledge," hence the name), not just its label. Loss combines: (a) cross-entropy on the current task's real data, and (b) an MSE term matching the *current* model's logits on buffer samples to the *stored* logits from write-time. DER++ (a common extension) adds a further replay-based CE term on the buffer's true labels alongside the logit-matching term.
**Requirements:** Buffer must store logits (a vector of size num_classes-at-write-time per sample) alongside features — larger per-sample memory footprint than A3's feature-only buffer, and the stored logit vector's dimensionality changes as classes are added, requiring padding/remapping logic.
**Strengths:** Reservoir sampling gives a buffer whose class composition automatically tracks the true stream frequency, without needing an explicit herding/class-balancing step. Matching logits (not just labels) preserves more information about the old model's decision boundary shape.
**Weaknesses:** Implementation complexity higher than A3/A4 (logit storage + remapping across growing output layer); reservoir sampling can still under-represent very rare classes if the buffer is small relative to total stream length.

### A9. BiC (Bias Correction)
**Mechanism:** Wu et al., *Large Scale Incremental Learning*, CVPR 2019. Diagnoses the recency-bias problem directly: after training each task normally (e.g. with a replay buffer), fit a **small linear correction layer** (just 2 scalar parameters, a scale and a shift applied to the new classes' logits) on a **held-out validation slice that's balanced between old and new classes**, using a separate optimization pass after the main training loop for that task.
**Requirements:** A dedicated held-out balanced validation split, kept separate from the main train/val data (reusing the training buffer for this purpose contaminates the very recency-bias signal you're trying to measure).
**Strengths:** Directly and interpretably targets the exact failure mode (logit-scale imbalance) rather than hoping a general-purpose regularizer fixes it as a side effect.
**Weaknesses:** Sensitive to how the held-out balancing split is constructed — too small and the 2-parameter fit is noisy; needs its own data-splitting discipline.

---

## Stream B — Analytic / closed-form classifier heads (frozen backbone after early tasks)

Shared infra: a feature extractor that is **not** updated by gradient descent after an initial phase; a lightweight classifier head updated via closed-form linear algebra (no backprop) as new classes arrive. This family trades "trainable-backbone flexibility" for "provable non-forgetting" — several of these methods are **mathematically equivalent** to retraining from scratch on all data, not just empirically resistant to forgetting.

### B1. SLDA (Streaming Linear Discriminant Analysis)
**Mechanism:** Hayes & Kanan, 2020 (the "REMIND" lineage). Maintain one **shared covariance matrix** (pooled across all classes) and a **running per-class mean vector**, both updated incrementally in O(1) per new sample (simple running-average update rules — no matrix inversion needed per sample, only once at classification time). Classification uses the standard LDA decision rule: assign to the class whose Gaussian (shared covariance, per-class mean) gives highest likelihood, equivalent to a linear discriminant.
**Requirements:** A frozen feature extractor; storage for one D×D covariance matrix and C mean vectors (D = feature dim, C = classes) — no raw exemplar storage.
**Strengths:** The simplest and cheapest method in this entire survey to implement and run. Despite its simplicity, it's a genuinely strong baseline in streaming/class-IL settings when the frozen feature space is reasonably linearly separable — often surprisingly competitive with much more complex methods.
**Weaknesses:** Assumes each class is roughly Gaussian and that a *shared* covariance across all classes is a reasonable approximation (heteroscedastic class distributions hurt it) — this is exactly the limitation FeCAM (B2) was designed to fix. Entirely dependent on the frozen feature extractor already being good; if the backbone can't linearly separate classes, no amount of classifier-head cleverness recovers that.

### B2. FeCAM (Feature Covariance-Aware Mean classifier)
**Mechanism:** Goswami et al., NeurIPS 2023. Same frozen-backbone philosophy as SLDA, but replaces the shared-covariance assumption with **per-class covariance matrices** (each with shrinkage regularization to stabilize estimation from limited samples, plus a Tukey power-transform on features to make class-conditional distributions closer to Gaussian). Classification is via **Mahalanobis distance** to each class's mean, using that class's own covariance — a strictly more expressive decision rule than SLDA's shared-covariance LDA.
**Requirements:** Same frozen-backbone assumption as SLDA, but storage for C separate D×D covariance matrices instead of one shared matrix — heavier memory footprint that grows with class count and feature dimensionality.
**Strengths:** Handles heterogeneous class shapes (some classes tight/compact, others diffuse) far better than SLDA's shared-covariance assumption.
**Weaknesses:** Per-class covariance estimation needs enough samples per class to be reliable — degrades on very rare classes without careful shrinkage tuning; higher memory cost than SLDA.

### B3. ACIL / G-ACIL / F-OAL (Analytic Class-Incremental Learning family)
**Mechanism:** Zhuang et al., *ACIL*, 2022; G-ACIL (arxiv 2403.15706); F-OAL, NeurIPS 2024. Reformulates the classifier head entirely as a **recursive closed-form ridge regression**: instead of a softmax layer trained by gradient descent, maintain a running correlation/auto-covariance matrix `R = Σ xᵢxᵢᵀ` (updated additively per new task via a rank-update formula, no need to revisit old data) and solve `W = (R + λI)⁻¹ Σ xᵢyᵢᵀ` in closed form. The key theoretical result is that this recursive update is **provably equivalent** to solving the ridge regression on the full concatenated dataset from scratch — i.e. it doesn't just resist forgetting empirically, it mathematically cannot forget (within the linear-model assumption).
**Requirements:** Frozen feature extractor; storage for one running (D×D or D×C, depending on formulation) matrix, updated per task with no raw exemplar storage needed at all once the matrix update is applied.
**Strengths:** The forgetting guarantee is exact, not approximate — a fundamentally different (stronger) kind of robustness than every method above. No backprop needed after the initial backbone-training phase, so per-task compute cost is very low (a matrix solve, not an epoch loop).
**Weaknesses:** The guarantee only holds within the fixed frozen-feature-space's expressive power — if the true class boundaries need representation changes the frozen backbone can't provide, no amount of closed-form classifier cleverness fixes that (same caveat as SLDA/FeCAM, but especially binding here since there's no gradient-based way to adapt).

### B4. AOCIL (Analytic Online CIL)
**Mechanism:** Same analytic-ridge-regression family as B3 (arxiv 2403.15751), specifically engineered for **per-sample** (true streaming, one-example-at-a-time) updates rather than per-task batch updates — relevant if the deployment setting genuinely sees data one row at a time rather than in task-sized batches.
**Requirements:** Same as B3.
**Strengths:** Lower latency per update if the use case is truly streaming (not batched-by-task).
**Weaknesses:** If the actual deployment is task-batched anyway (most CIL benchmarks and most realistic "new categories added periodically" settings are), this buys little over B3 for extra implementation care around per-sample update numerics.

### B5. AnaCP (Analytic Contrastive Projection)
**Mechanism:** Late-2025 paper (arxiv 2511.13880) extending the B3 lineage: adds a **contrastive projection** step (a learned or fixed projection that pulls same-class features together and pushes different-class features apart) applied *before* the closed-form ridge head, explicitly aiming to close the gap to joint-training accuracy (not just avoid forgetting relative to sequential fine-tuning).
**Requirements:** Same frozen-backbone-plus-closed-form infra as B3, plus fitting/storing the contrastive projection.
**Strengths:** Directly targets the "how close can we get to the oracle ceiling" question, which is usually the more interesting number in practice than just "how much better than sequential fine-tuning."
**Weaknesses:** Newest and least battle-tested method in this survey (only found via a single late-2025 arxiv listing at the time of this survey) — mechanism details beyond the abstract-level summary above were not independently verified against the full paper.

### B6. Class-balanced analytic ridge regression
**Mechanism:** A 2026-wave variant of the B3 family that bakes per-class reweighting **directly into the closed-form ridge solve** — instead of a separate balanced-sampler or focal loss applied during a gradient-based training loop (which doesn't exist here anyway, since there's no gradient loop), the reweighting is applied to the second-order feature statistics (`R`) themselves before the matrix solve, so rare classes get proportionally more influence on the closed-form solution.
**Requirements:** Same as B3, plus per-class count tracking to compute the reweighting.
**Strengths:** Attacks class imbalance at the same mathematical layer as the rest of the analytic family — no extra loss-function engineering, no extra hyperparameter sweep over a focal-loss gamma or sampler weights.
**Weaknesses:** Same epistemic caveat as B5 — a very recent variant, mechanism understood at the level of "what it claims to do," not independently verified against a full implementation.

---

## Stream C — Prototype / pseudo-rehearsal methods

Shared infra: store compact **per-class statistics** (not raw exemplars) — typically a mean vector and some notion of spread — and **synthesize** pseudo-samples from those statistics for replay, rather than replaying real stored data. This sits philosophically between "exemplar-based" (still replaying *something*) and "exemplar-free" (no raw data at all) — sometimes called "pseudo-rehearsal."

### C1. PASS (Prototype Augmentation and Self-Supervision)
**Mechanism:** Zhu et al., CVPR 2021 (oral). Stores exactly **one prototype vector per old class** (the class mean in feature space — much cheaper than A3/A6's full exemplar buffers). When training on new tasks, samples **pseudo-features** by adding Gaussian jitter around each stored prototype, and trains the classifier on these synthetic old-class samples alongside real new-class samples. Additionally uses a **self-supervised auxiliary task** during representation learning (the original paper uses rotation-prediction, a vision-specific pretext task) to improve the general transferability of the learned features, which the authors found helps the prototype-jitter approximation stay valid across tasks.
**Requirements:** Just C mean vectors (extremely cheap storage — this is the lightest true exemplar-adjacent method in the survey after B1/SLDA). The self-supervised auxiliary task is the one genuinely vision-specific piece (rotation prediction has no natural equivalent for a raw feature vector) — needs a tabular-appropriate substitute (e.g., a masked-feature-reconstruction pretext task, analogous to how tabular deep-learning papers commonly adapt SSL for non-image data).
**Strengths:** Very cheap storage; the core prototype-jitter mechanism has no vision-specific assumptions at all.
**Weaknesses:** Single-Gaussian-per-class approximation can break down for classes with multi-modal or highly non-Gaussian feature distributions (same underlying assumption risk as SLDA). The auxiliary SSL task needs non-trivial adaptation work to make sense outside images.

### C2. CEFCIL (Comprehensive Ensemble Framework for Exemplar-Free CIL)
**Mechanism:** 2025 work. Ensembles **multiple NCM/Mahalanobis classifiers**, each trained on a different "view" of the backbone features (in the original vision formulation, these views come from different data augmentations of the same image), softmax-averaged at inference. Adds a cached "root model" (an early-task snapshot) used as a distillation anchor, and a "dimensional collapse" regularizer that discourages the feature space from degenerating into a low-rank subspace over many tasks.
**Requirements:** Multiple parallel classifier heads (one per ensemble member) plus the root-model cache.
**Strengths:** Ensembling generally improves robustness/calibration over any single classifier variant in this survey.
**Weaknesses:** The "diversified views" concept is intrinsically tied to image-augmentation techniques (crops, flips, color jitter) — needs a genuine tabular substitute (e.g., random feature subsets, dropout-masked views, or bootstrap resampling of the feature vector) whose effectiveness at producing *useful diversity* is unverified for tabular data specifically.

### C3. Manifold-aware boundary sampling (arxiv 2606.05695, June 2026)
**Mechanism:** A PASS-family variant: instead of sampling pseudo-exemplars via uniform Gaussian jitter around the class mean, it samples pseudo-exemplars specifically **near class decision boundaries** on the estimated data manifold — the intuition being that boundary-adjacent synthetic samples are more informative for keeping decision boundaries sharp than samples drawn from the bulk of the class distribution. Paired with an adaptive class-balanced loss.
**Requirements:** Manifold-distance/boundary estimation in feature space (e.g., via nearest-neighbor graphs or local density estimation) — meaningfully more computation than PASS's simple Gaussian jitter.
**Strengths:** Directly targets the part of the feature space that actually matters for classification decisions, rather than uniformly covering the whole class distribution.
**Weaknesses:** Manifold/boundary estimation is itself a nontrivial sub-problem, more so in higher-dimensional tabular spaces than in the (often heavily downsampled via a CNN) feature spaces this was likely validated on; only found via arxiv listing, mechanism detail beyond the abstract not independently verified.

### C4. Exemplar-free CIL via preserving class-discriminative structure (CVPR 2026)
**Mechanism:** Regularizes the feature space's discriminability directly — a loss term that discourages old-class feature clusters from drifting closer together or overlapping as new classes are learned — rather than storing any prototypes or exemplars at all.
**Requirements:** Unclear beyond the abstract-level description found; would need the full paper before this could be implemented with confidence.
**Strengths:** If it works as described, it's the "purest" exemplar-free approach in this survey (Stream C entries C1–C3 all store *something*, even if just a mean vector; this stores nothing but a regularization signal).
**Weaknesses:** Least-understood entry in this entire survey — flagged here for completeness, not as an actionable near-term candidate.

### C5. Prototype Latent World Model Replay (arxiv 2606.29465)
**Mechanism:** Stores each old class as a **prototype-centered distribution (mean + variance) in a latent space**, encoded via a frozen pretrained encoder in the original (vision) formulation. Samples from these distributions to replay old classes when learning new ones, trains a lightweight adapter + classifier on the mix of sampled-old and real-new latent states, using supervised contrastive learning to sharpen class separation.
**Requirements:** In the original vision formulation, a frozen ImageNet-pretrained encoder — **not needed** for tabular data, since the input is already a numerical feature vector; the "latent space" can simply be the frozen backbone's own feature space (making this mechanically very close to C1/PASS, distinguished mainly by explicitly storing a variance term, not just a mean, and by using supervised contrastive learning rather than a rotation-prediction SSL task).
**Strengths:** Conceptually the same low-storage-cost profile as PASS, and the "no frozen ImageNet encoder needed" caveat is actually a strength here — this method's core mechanism was never vision-specific in the first place, unlike PASS's SSL half.
**Weaknesses:** Storing a full covariance (or even diagonal variance) per class instead of just a mean adds some storage/compute over the plainer PASS approach, for a benefit that's unverified outside the original CIFAR-100 validation.

### C6. TRIL3 — Tabular-native pseudorehearsal (arxiv 2407.09039, 2024)
**Mechanism:** The **only technique in this entire survey validated on tabular data specifically**, not images. Uses **XuILVQ**, a prototype-based *incremental generative model* (an evolution of Learning Vector Quantization), to synthesize artificial pseudo-samples standing in for old-class data — this is a genuinely generative approach (produces new synthetic feature vectors resembling the old class distribution), not just jitter around a stored mean like PASS. Pairs this with **DNDF** (Differentiable Decision Forest), a decision-tree-ensemble classifier reformulated to be trainable via gradient descent and updatable incrementally, as the actual classifier. The paper reports matching or beating other CL baselines using only 50% synthetic (generated) data for the old-class portion of training.
**Requirements:** Training/maintaining the XuILVQ generative prototype model incrementally (nontrivial but well-specified in the source paper), plus a DNDF classifier instead of a standard MLP/softmax head.
**Strengths:** Zero raw-data retention; validated on the exact data modality (tabular) this document assumes; the generative (not just jitter) approach to pseudo-sample synthesis is potentially more faithful to real class distributions than PASS's simple Gaussian assumption.
**Weaknesses:** Requires implementing two nonstandard components (XuILVQ generator, DNDF classifier) rather than reusing a plain MLP — meaningfully higher implementation effort than most other entries in this survey, despite being the best modality match.

---

## Stream D — Reservoir / random-feature methods

### D1. CIRCLE (Data-free reservoir features for cold-start CIL)
**Mechanism:** Uses a **fixed random-feature reservoir** (e.g. randomly initialized, never trained — the "BiRC2D"-style bidirectional random projection) as the entire feature extractor, combined with **Streaming LDA heads** (the same mechanism as B1) — but ensembled across **multiple independent random reservoir instantiations**, averaging softmax outputs across the ensemble members. The defining property: the feature extractor is never trained on any data at all, at any point.
**Requirements:** No training compute for the backbone whatsoever — only the SLDA heads' running statistics need updating; ensemble of reservoirs multiplies storage/compute by the ensemble size, but each member is cheap.
**Strengths:** Extremely low compute cost (no backprop anywhere in the pipeline); explicitly validated at very fine task granularities (hundreds of small tasks), a regime where methods requiring backbone retraining per task become prohibitively expensive.
**Weaknesses:** A random (untrained) feature extractor is a strictly weaker representation than any of the trained/frozen-after-training backbones used in Streams A-C — this method trades accuracy ceiling for extreme compute efficiency. Best understood as a "how cheap can we go" reference point, not a top accuracy candidate.

---

## Reference-only: not a method, a benchmark harness

### R1. TSCIL (arxiv 2402.12035, KDD 2024, `zqiao11/TSCIL` on GitHub)
A standardized, open-source class-incremental benchmark suite built for **time-series** data (not vision) — architecture-agnostic, evaluates feature-extractor-plus-classifier CIL methods on sequential numeric data. Not itself a technique to adopt, but a useful thing to **mine**: (a) its baseline method implementations are a source of already-debugged reference code for several of the methods above (SLDA, EWC, replay-based methods are typically all included in such suites), and (b) its evaluation-metric code is a good sanity-check target for any new BWT/FWT/forgetting-metric implementation, since it's a maintained open-source repo rather than a from-scratch reimplementation from a paper's math alone.

---

# Part 2 — My technical viewpoint

*Assuming only: multi-class tabular data, new classes arrive over time, class-IL setting (no task-ID at inference). No other project-specific detail factored in.*

## Tiered recommendation

**Tier 1 — build first, in this order:**

1. **SLDA (B1).** Almost zero implementation cost, and it answers a question every other method's usefulness depends on: *is the frozen feature space even linearly separable enough for a cheap closed-form classifier to work well?* If SLDA already gets you close to the joint-training ceiling, that's a strong signal the whole analytic family (B2–B6) is worth investing in, and that trainable-backbone methods (Stream A) may be solving a problem you don't actually have. If SLDA badly underperforms, that's equally informative — it tells you the *backbone*, not the classifier head, is the bottleneck, which reframes where effort should go.
2. **Replay with herding (A3), plain — not yet the distillation or NME variants.** This is the most battle-tested rehearsal method that exists, full stop. It's the right second data point because it's a completely different philosophy from SLDA (trainable backbone, real exemplars) — comparing the two immediately tells you whether "keep training the backbone" or "freeze it and use closed-form linear algebra" is the more productive axis for this specific dataset.
3. **TRIL3 (C6), *if* the extra implementation cost (XuILVQ + DNDF) is acceptable.** It's the only technique here actually validated on tabular data, which for a generic "we know it's tabular and nothing else" starting point is a genuinely meaningful prior, not just a convenient checkbox — a method designed and tuned against tabular data's actual statistical properties (lower dimensionality, more linear structure, no spatial locality assumption to exploit or misapply) is more likely to transfer cleanly than a method whose design choices were shaped by CNN/image quirks.

**Tier 2 — build next, once Tier 1 results indicate which axis (frozen-closed-form vs. trainable-gradient vs. generative-pseudorehearsal) is winning:**

- If SLDA (B1) did well: extend along the analytic family — **FeCAM (B2)** first (per-class covariance is a natural refinement if the shared-covariance assumption is even close to working), then **ACIL/AnaCP (B3/B5)** if you specifically care about closing the gap to the joint-training ceiling rather than just beating sequential fine-tuning, and **class-balanced ridge (B6)** specifically if class imbalance turns out to be a dominant error source (checkable directly from per-class accuracy breakdowns after Tier 1).
- If replay (A3) did well: **replay + distillation (A4)** is close to free to add on top and usually helps; **iCaRL's NME variant (A6)** is worth an isolated test since it's a near-zero-cost swap on top of infrastructure you already have.
- If TRIL3 did well: this validates the pseudo-rehearsal family generally — **PASS (C1)** becomes attractive as a much cheaper (single mean vector, no generative model) approximation worth checking whether it captures most of TRIL3's benefit at a fraction of the engineering cost.

**Tier 3 — lower priority, situational:**

- **EWC (A5) and pure-regularization methods generally.** The literature consensus (not specific to this dataset) is that regularization-only methods without any output-layer correction underperform in true class-IL with more than a handful of classes, because nothing in their objective addresses the growing-head recency-bias problem directly. Worth having as a reference point, low expectation of it being competitive.
- **LwF (A7).** Reasonable if storing *any* data (even prototypes) is a hard constraint (e.g. genuine privacy requirement), but if that constraint doesn't actually hold, methods that do store lightweight statistics (SLDA, PASS) tend to outperform it.
- **BiC (A9), DER (A8).** Solid, well-validated methods, but not distinctive enough over A3/A4 to prioritize before the Tier-1/Tier-2 methods above give a clearer signal about which broad approach (frozen-closed-form vs. trainable-replay vs. generative-pseudorehearsal) suits this specific data.
- **CIRCLE (D1).** Only prioritize this if the real deployment constraint is "hundreds of small incremental updates with near-zero compute budget per update" — for a moderate number of tasks (the common case), its untrained-random-backbone tradeoff is unlikely to be worth the accuracy it gives up.
- **CEFCIL (C2), manifold-aware boundary sampling (C3), exemplar-free discriminative-structure preservation (C4).** All three carry either meaningful vision-specific baggage that needs non-trivial tabular adaptation (C2's "diversified views," C3's manifold estimation) or are too thinly specified to implement with confidence yet (C4). Revisit after Tier 1/2 results narrow the search, not before.

## Why this ordering, reasoned from "tabular data" alone

Two structural facts about tabular data (as opposed to images) drive this ranking:

1. **Tabular feature spaces are usually lower-dimensional and closer to linearly structured than raw pixel/CNN-embedding spaces.** This is exactly the regime where closed-form linear methods (Stream B) are strongest relative to their vision performance — their core assumption (a frozen space where a linear or Gaussian classifier is a good fit) is *more* likely to hold for tabular features than for the high-dimensional, highly non-linear manifolds CNNs produce from images. This is why Tier 1 leads with SLDA rather than with the more complex methods.
2. **Pseudo-rehearsal methods that assume roughly-Gaussian class-conditional distributions (PASS, prototype-latent-replay) are betting on an assumption that's also more often true for engineered/measured tabular features than for raw pixels** — pixel-space class distributions are notoriously multi-modal and non-Gaussian, which is part of why the original PASS paper needed a rotation-prediction SSL crutch to make the assumption workable at all. For tabular data, that crutch may be less necessary, and the core prototype-jitter mechanism may transfer more cleanly on its own.

Both observations point the same direction: methods whose core mathematical assumption is "the feature space is well-behaved (linear/Gaussian)" are *better bets specifically because the data is tabular*, not despite it — which is the opposite of how these methods are usually regarded in the vision-CIL literature they were mostly developed for (there, they're often seen as cheap-but-limited alternatives to "real" trainable-backbone methods). That inversion is the single most useful takeaway from this survey for a "tabular, and nothing else known" starting point.
