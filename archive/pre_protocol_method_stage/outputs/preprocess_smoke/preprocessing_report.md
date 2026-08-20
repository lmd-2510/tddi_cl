# Preprocessing Report

## Configuration

- scaler: `standard`
- impute_strategy: `zero`
- rows_fitted_on_train: `2048`
- feature_count: `3780`
- partial_scan_mode: `True`

## Explicit Feature Policy

- label column excluded: `class`
- meta columns excluded: `drugid-drug_a, drugid-drug_b, drugname-drug_a, drugname-drug_b, drugsmiles-drug_a, drugsmiles-drug_b`
- feature columns source: `feature_columns.json`

## Schema Validation

- train schema column count: `3787`
- label present in train schema: `True`
- feature count validated: `3780`

## Split Scan Summary

| split | rows_scanned | nonfinite_rows | columns_with_nonfinite | null_count_total | nan_count_total | inf_count_total | partial_scan |
| --- | --- | --- | --- | --- | --- | --- | --- |
| train | 2048 | 0 | 0 | 0 | 0 | 0 | True |
| validation | 2048 | 0 | 0 | 0 | 0 | 0 | True |
| test | 2048 | 0 | 0 | 0 | 0 | 0 | True |

## Notes

- Scaler is fit on train split only.
- This script does not materialize full transformed train/validation/test matrices by default.
- `standard` scaler is exact under the chosen imputation strategy.
- `robust` scaler uses a deterministic sampled approximation to avoid loading the full matrix into RAM.
