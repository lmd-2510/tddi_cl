# Prepared feature cache

The cache materializes the frozen `task0_standard_frozen` transform once. It
does not fit a new scaler, rebuild folds, or alter labels/sample identities.
The legacy EWC runner uses it automatically under
`outputs/prepared_cache/p4_seed0_fold42/member_0`.

## EWC pilot

Run as before:

```bash
bash scripts/run_p4_ewc_pilot.sh start
```

The first run has a one-time cache step. Later runs reuse the cache and validate
its source/scaler hashes before training. The cache currently contains train
and validation splits; test remains raw and uses the same frozen scaler.

## Reuse from another legacy method

Materialize a member-specific cache once, then pass it to `train_cil.py`:

```bash
python scripts/materialize_prepared_cache.py \
  --train train_extracted.parquet \
  --validation validation_extracted.parquet \
  --feature-cols study_assets/data_schema/feature_columns.json \
  --scaler path/to/task0_frozen_scaler.pkl \
  --output outputs/prepared_cache/<protocol>/<member>

python src/training/train_cil.py ... \
  --prepared-cache-root outputs/prepared_cache/<protocol>/<member>
```

Use one cache per member because each member has its own frozen preprocessing
artifact. Keep the legacy engine and old artifacts until the cached run has
been validated against an uncached run.
