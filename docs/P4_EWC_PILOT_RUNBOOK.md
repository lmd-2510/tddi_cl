# P4 EWC pilot — member 0

This pilot is deliberately paper-oriented rather than a hybrid of the existing replay pipeline:

- protocol: P4 `constrained_mass_balanced` (38/20/20/20/20/20/20/20 classes);
- method: empirical diagonal EWC;
- task loss: ordinary cross-entropy;
- `ewc_lambda=1000`;
- Fisher: one empirical diagonal pass over the current task's training rows at each task boundary;
- replay and exemplar buffer: none;
- model/training defaults kept from the best historical P4 recipe (T-DDI paper member, 7560-7560 GELU + LayerNorm, AdamW, 64/1024 accumulated batch, 30 epochs, patience 5);
- preprocessing: task-0-only frozen standardization. The shell runner converts the P4 JSON artifact to the legacy scaler payload without refitting.
- member 0 uses the deterministic P4 stratified fold (validation fold 0, seed 42), so its training partition does not include the held-out fold.

Run on the Linux server:

```bash
cd "$HOME/DrugDrug/cil-tddi/cil-tddi"
conda activate ai_env

# Only needed when the automatic search finds more than one P4 preprocessing root.
# export P4_PREP_ROOT="$PWD/study_assets/preprocessing_p4_seed0_fold42"

bash scripts/run_p4_ewc_pilot.sh check
bash scripts/run_p4_ewc_pilot.sh start
bash scripts/run_p4_ewc_pilot.sh status
```

`start` uses `nohup`, so the SSH terminal can be closed. The output is placed under
`outputs/p4_ewc_pilot_member0/<UTC run id>/`; the review ZIP excludes `.npz`/`.npy` and the temporary scaler.

This is a member-0 pilot only; it is not an ensemble. Compare task-6→task-7 forgetting, final seen-all macro-F1, and the EWC penalty audit against the P4 hybrid/equal-buffer baseline before deciding whether to scale to three members.
