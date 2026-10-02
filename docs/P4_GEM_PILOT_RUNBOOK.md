# P4 GEM pilot — member 0

This pilot follows the original GEM memory policy: fixed episodic memory split
equally by task, with the last `m` stream examples retained for each task.
For P4's eight tasks, `M=2048` gives `m=256` per task. The memory is not
mixed into the current training batch; it supplies one gradient constraint per
previous task, followed by the GEM quadratic-program projection.

The pilot keeps the repository's common model/training schedule (CE,
7560-7560 GELU + LayerNorm, AdamW, batch 64/effective 1024, 30 epochs) so the
experiment runs through the existing P4 pipeline. It is a method survey, not a
claim of an exact reproduction of the paper's one-pass SGD benchmark.

```bash
cd "$HOME/DrugDrug/cil-tddi/cil-tddi"
conda activate ai_env
export P4_PREP_ROOT="$PWD/study_assets/preprocessing_p4_seed0_fold42"

bash scripts/run_p4_gem_pilot.sh check
bash scripts/run_p4_gem_pilot.sh start
```

Use `status` or `follow` to monitor. The review ZIP excludes temporary scaler
and `.npz`/`.npy` files.
