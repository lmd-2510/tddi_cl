# P3 Hybrid full run

This run uses the original TDDI paper-size member, P3 `tail_to_head`, eight tasks,
and `loss_variant=hybrid`. The replay buffer is 4% of the full development set:
27,778 stored slots total, split as 9,260/9,259/9,259 across members 0/1/2.
The separate `fraction=0.125` is replay exposure per training batch, not the
buffer size.

On the GPU server, from the repository root:

```bash
conda activate ai_env

# The script auto-discovers a unique fold/preprocessing root. Do not type a
# placeholder timestamp. Only set these variables when more than one candidate
# exists, using the exact paths printed by find.
unset P3_FOLD_ROOT P3_PREP_ROOT

# Optional explicit form:
# export P3_FOLD_ROOT="$PWD/outputs/<real-fold-preparation-dir>"
# export P3_PREP_ROOT="$PWD/study_assets/<real-p3-preprocessing-dir>"

# If P3 preprocessing does not exist yet, create it from the existing fold
# preparation (assignments + manifest):
# export P3_FOLD_ROOT="$PWD/outputs/<real-fold-preparation-dir>"
# bash scripts/prepare_p3_assets.sh

bash scripts/run_p3_hyb.sh check
bash scripts/run_p3_hyb.sh dry-run
bash scripts/run_p3_hyb.sh start 0
```

Follow or inspect member 0:

```bash
bash scripts/run_p3_hyb.sh status
bash scripts/run_p3_hyb.sh follow
```

After member 0 is complete, run members sequentially:

```bash
bash scripts/run_p3_hyb.sh start 1
bash scripts/run_p3_hyb.sh start 2
```

After all three members have task 7 checkpoints and prediction artifacts, run:

```bash
bash scripts/run_p3_hyb.sh evaluate
```

The script never starts two members concurrently. Existing complete members are
skipped and an interrupted member is resumed from its valid task-boundary
checkpoint. Outputs are isolated under `outputs/p3_hyb_full8_seed0`.
