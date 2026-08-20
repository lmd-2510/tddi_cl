# Leakage Summary

## Scope

- This audit checks ordered pair overlap, unordered/reverse overlap, drug-level overlap, duplicate pair patterns, and multilabel unordered pairs.
- Exact duplicate row check is performed on the identity subset `(drugid-drug_a, drugid-drug_b, class)` for memory-aware auditing.

## Feature Leakage Guard

- `class` present in schema: `True`
- `class` accidentally in `feature_cols`: `False`
- meta columns accidentally in `feature_cols`: `[]`
- explicit feature count from schema: `3780`

## Pair Overlap Summary

| split_left | split_right | ordered_overlap_count | unordered_overlap_count | reverse_overlap_unordered_count |
| --- | --- | --- | --- | --- |
| train | test | 0 | 0 | 0 |
| train | validation | 0 | 0 | 0 |
| validation | test | 0 | 0 | 0 |

## Drug Overlap Summary

| split_left | split_right | unique_drugs_left | unique_drugs_right | overlap_drug_count | overlap_fraction_left | overlap_fraction_right | jaccard |
| --- | --- | --- | --- | --- | --- | --- | --- |
| train | validation | 2921 | 2864 | 2847 | 0.9746662102019856 | 0.9940642458100558 | 0.9690265486725663 |
| train | test | 2921 | 2862 | 2843 | 0.9732968161588497 | 0.993361285814116 | 0.9670068027210884 |
| validation | test | 2864 | 2862 | 2826 | 0.986731843575419 | 0.9874213836477987 | 0.9744827586206897 |

## Duplicate Pair Summary

| split | duplicate_type | duplicate_groups | duplicate_rows | notes |
| --- | --- | --- | --- | --- |
| train | exact_duplicate_identity_rows |  | 0 | Duplicates on (drugid-drug_a, drugid-drug_b, class). |
| train | ordered_pair_duplicates | 0.0 | 0 | Repeated ordered pairs within the split. |
| train | unordered_pair_duplicates | 0.0 | 0 | Repeated unordered pairs within the split, including A-B/B-A. |
| validation | exact_duplicate_identity_rows |  | 0 | Duplicates on (drugid-drug_a, drugid-drug_b, class). |
| validation | ordered_pair_duplicates | 0.0 | 0 | Repeated ordered pairs within the split. |
| validation | unordered_pair_duplicates | 0.0 | 0 | Repeated unordered pairs within the split, including A-B/B-A. |
| test | exact_duplicate_identity_rows |  | 0 | Duplicates on (drugid-drug_a, drugid-drug_b, class). |
| test | ordered_pair_duplicates | 0.0 | 0 | Repeated ordered pairs within the split. |
| test | unordered_pair_duplicates | 0.0 | 0 | Repeated unordered pairs within the split, including A-B/B-A. |

## Reverse Overlap Preview

_No reverse overlap rows found._

## Multilabel Unordered Pair Preview

_No unordered pair mapped to multiple classes within a split._

## Interpretation Notes

- Ordered overlap indicates the exact same directional pair appears across splits.
- Unordered/reverse overlap indicates the same drug pair may reappear after swapping A/B.
- If train/test overlap is non-trivial, static supervised metrics can be optimistic.
- This audit does not silently redefine the original split; pair-disjoint variants remain future ablations.
