# P3 Hybrid + equal-class buffer: full run

This experiment starts from `configs/p3_hybrid_full8_e30_mem4_nodistill.json` and changes only:

- Replay buffer allocation: `fold_min_quota_sqrt_capacity_v1` → `fold_equal_class_capacity_v1` (equal class allocation, capped by retained examples).
- Classification loss: explicitly `hybrid` (Focal for current-task rows + CE for replay rows; no logit/feature distillation).

The rest stays historical: P3 tail-to-head, 8 tasks, 30 max epochs, patience 5, AdamW settings, replay fraction 12.5%, repeat cap 3, 4% memory **per member** (27,778 slots each), same fold/preprocessing artifacts, and no post-hoc WA. The equal allocation is a buffer policy, not an evaluation-time class weighting or access to future data.

## Start on the training server

First make sure the updated repository files/config have been synced to the server and the P3 fold/preprocessing inputs remain present. From the repository root:

```bash
conda activate ai_env
bash scripts/run_p3_hybrid_equal_buffer_full.sh
```

The default action is `start`: checks inputs and runs a dry-run first, then detaches the full job with `nohup`. It trains members 0→1→2 sequentially, then the existing full runner creates the offline OOF/test ensemble. A failure in that stage is recorded; postprocessing/package attempts continue where possible.

Check or follow the job:

```bash
bash scripts/run_p3_hybrid_equal_buffer_full.sh status
bash scripts/run_p3_hybrid_equal_buffer_full.sh follow
```

Optional foreground dry-run only:

```bash
bash scripts/run_p3_hybrid_equal_buffer_full.sh check
```

## Results and review ZIP

Each start uses a UTC timestamp. The complete run is kept under:

```text
outputs/p3_hybrid_equal_buffer_full8_e30_mem4_seed0_<timestamp>/
```

When training finishes, the runner generates the standard offline ensemble report, task 6 and 7 member/ensemble figures for validation and test, plus aligned boundary diagnostics under `final_results/boundary_task6_task7/`. The diagnostic includes per-task/per-class sample support, same-ID old-class F1/precision/recall changes, and old→new / new→old error rates for each member and the ensemble. Validation is for diagnosis and next-run decisions; test should remain descriptive, not be tuned against.

The compact archive and SHA256 sidecar are under:

```text
outputs/p3_hybrid_equal_buffer_full8_e30_mem4_seed0_<timestamp>/review/
```

It includes configs, summaries, metrics/audits, logs, and PNGs. NPZ predictions, checkpoints/model weights, and Parquet sources are intentionally excluded; the offline ensemble has already been computed before those NPZ files are omitted from the ZIP.
