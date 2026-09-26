# Protocol-diverse P2/P3/P4 ensemble — full run

This experiment assigns one full eight-task trajectory to each member:

| Member | Protocol | Order |
|---:|---|---|
| 0 | P2 `head_to_tail` | Frequent classes first |
| 1 | P3 `tail_to_head` | Rare classes first |
| 2 | P4 `constrained_mass_balanced` | Frequency strata spread across tasks |

All members use the selected best-historical recipe with the two agreed changes:
Hybrid loss (Focal on current-task rows, CE on replay rows; no distillation) and
equal-class buffer. Other settings remain the same: 4% development memory per
member (27,778 slots), 12.5% replay, repeat cap 3, up to 30 epochs/task,
patience 5, AdamW, same experiment/fold seeds and fixed 3-fold assignments.

P2 is generated and frozen in the run output from class counts in the training
split; task membership is built once before training. The models do not receive
future examples before their respective task. Each model's scaler is fit only
from that member's task-0 training rows. Since the order differs, task-0
preprocessing and class histories legitimately differ by member.

## Output and interpretation

The runner performs the three trajectories sequentially, then averages the
members' probabilities **only at task 7 on the common test set**, where all
three output spaces contain the same 178 raw classes. It also reports every
member's test metrics for tasks 0–7 according to that member's own protocol,
task-6/task-7 classwise changes, per-protocol task sample counts, task-7
ensemble classwise metrics, confusion counts, disagreement, figures, and a
compact ZIP with SHA256 sidecar.

Task 0–6 scores are protocol-specific: the models have seen different class sets
at the same numbered task. They are not averaged or presented as if they shared
one boundary. Likewise, the current OOF pipeline cannot produce a valid
three-model OOF ensemble here: each fold's held-out member is one protocol, but
the other members trained on that fold. Therefore the report makes no
protocol-diverse OOF threshold claim; task-7 ensemble results are raw,
full-test metrics.

Protocol is assigned one-to-one with member/fold (P2→member 0/fold 0,
P3→member 1/fold 1, P4→member 2/fold 2). Therefore an improvement would support
this complete combined setup, but would not isolate protocol diversity as the
cause. Rotate the protocol-to-member assignment in follow-up runs before making
a general claim about heterogeneous protocols.

The predeclared primary comparison is task-7 common-test ensemble Macro-F1
against the historical P3 Focal-all result (`0.750904`), with Balanced Accuracy
(`0.712910`) as a guardrail. The report marks “exceeds historical on both” only
when Macro-F1 is higher and Balanced Accuracy is not lower. This is descriptive
for one experiment seed and one fixed test split, not a statistical-significance
claim; the historical bundle has no per-example predictions for paired
bootstrap intervals.

## Run on the Linux training server

From the repository root, after activating the same environment used for the
other P3 runs:

```bash
conda activate ai_env
bash scripts/run_protocol_diverse_ensemble.sh check
bash scripts/run_protocol_diverse_ensemble.sh start
```

`start` performs input/protocol preflight and detaches the full run with
`nohup`; the terminal may be closed after it prints `[STARTED]`. Monitor it with:

```bash
bash scripts/run_protocol_diverse_ensemble.sh status
bash scripts/run_protocol_diverse_ensemble.sh follow
```

If the fold directory is ambiguous, set it explicitly before `check`/`start`:

```bash
export PDE_FOLD_ROOT="$PWD/outputs/fold_preparation_seed42_<timestamp>/folds"
```

The run root is `outputs/protocol_diverse_ensemble_seed0_<UTC timestamp>/`.
The human-readable result is
`final_results/PROTOCOL_DIVERSE_ENSEMBLE_RESULTS.md`; the compact ZIP and
`.sha256` are under `review/`. The ZIP includes configurations, task files,
metrics, audit CSV/JSON, logs and figures, but excludes prediction NPZ files,
checkpoints/model weights and source Parquet.
