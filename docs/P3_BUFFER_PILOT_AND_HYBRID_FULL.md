# P3 equal-buffer pilot + Hybrid full run

Run from the repository root on the training server:

```bash
bash scripts/run_p3_experiments.sh run
```

The script runs the full eight-task P3 protocol for member 0 using Focal on all
current and replay examples with no distillation, plus equal-per-seen-class
buffer allocation. It then attempts a fresh three-member Hybrid run (Focal for
current examples, CE for replay examples, no distillation) using the historical
sqrt-quota buffer and runs the pipeline's normal offline OOF/test ensemble and
threshold evaluation after all members complete. The experiments use separate
output roots and execute sequentially.

If automatic input discovery finds more than one fold/preprocessing directory,
set the exact paths first:

```bash
export P3_FOLD_ROOT="$PWD/outputs/fold_preparation_seed42_YYYYMMDD_HHMMSS/folds"
export P3_PREP_ROOT="$PWD/study_assets/preprocessing_p3_seed0_fold42"
bash scripts/run_p3_experiments.sh run
```

The script continues to the second experiment if training, visualization, or
packaging for the first experiment fails. Check `outputs/p3_experiments/` for
per-experiment logs, status files, review ZIPs, and SHA256 sidecars. ZIPs contain
configs, reports, logs, CSV/JSON audits, and generated visualizations; NPZ/NPY
prediction arrays, model checkpoints, and source Parquet files are excluded.

For an input/configuration check without training:

```bash
bash scripts/run_p3_experiments.sh check
```

Equal allocation is capacity-constrained: each seen class is assigned an equal
number of retained slots when possible, with unused slots redistributed if a
class has fewer available examples. Since discarded old descriptors are not
reloaded, a class cannot grow beyond the examples still retained from its prior
task boundary. The policy and exact allocation are recorded per task.
