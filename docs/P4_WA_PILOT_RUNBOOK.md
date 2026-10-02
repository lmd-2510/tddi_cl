# P4 pure Weight Alignment pilot — member 0

This is the paper-style WA counterpart to the pure EWC pilot:

- P4 `constrained_mass_balanced`, deterministic member 0 / validation fold 0;
- ordinary cross-entropy, no replay and no exemplar buffer;
- post-task `new_class_mean_norm_v1`: scale only newly introduced classifier weight rows to the mean L2 norm of old rows; bias is unchanged;
- same T-DDI paper-member architecture and training settings as EWC (7560-7560 GELU + LayerNorm, AdamW, batch 64/effective 1024, 30 epochs, patience 5);
- task-0 frozen standardization; no scaler refit during the run.

On the server:

```bash
cd "$HOME/DrugDrug/cil-tddi/cil-tddi"
conda activate ai_env
export P4_PREP_ROOT="$PWD/study_assets/preprocessing_p4_seed0_fold42"

bash scripts/run_p4_wa_pilot.sh check
bash scripts/run_p4_wa_pilot.sh start
```

The run is `nohup`-based. Outputs are under
`outputs/p4_wa_pilot_member0/<UTC run id>/`; each task writes a
`weight_alignment_task_<task>.json` audit containing `gamma` and the old/new
classifier norms. The review package excludes temporary scaler and `.npz`/`.npy` files.
