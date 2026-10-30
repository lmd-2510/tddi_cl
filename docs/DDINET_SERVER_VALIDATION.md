# DDINet server validation

DDINet hiện đã có backbone, Morgan preprocessing/cache và unit test. Backbone
**chưa được nối vào training launcher**, vì vậy các lệnh dưới đây chỉ dùng để
kiểm tra dữ liệu và smoke-test trước khi hoàn thiện workflow train.

## 1. Chuẩn bị môi trường

```bash
cd "$HOME/DrugDrug/cil-tddi/cil-tddi"
conda activate ai_env
git pull
python -m pip install -r requirements.txt
python -c "import rdkit; print('RDKit:', rdkit.__version__)"
```

## 2. Audit SMILES và tạo fingerprint cache

Lệnh này đọc bốn cột metadata, kiểm tra SMILES thiếu/sai và một drug ID ánh xạ
đến nhiều cấu trúc. Cache chỉ được tạo khi toàn bộ audit hợp lệ.

```bash
python - <<'PY'
from collections import defaultdict
from pathlib import Path
import pyarrow.parquet as pq

from src.data.ddinet_fingerprints import (
    build_ddinet_fingerprint_cache,
    canonicalize_smiles,
)

files = [
    Path("train_extracted.parquet"),
    Path("validation_extracted.parquet"),
    Path("test_extracted.parquet"),
]
columns = [
    "drugid-drug_a", "drugsmiles-drug_a",
    "drugid-drug_b", "drugsmiles-drug_b",
]
raw_smiles = defaultdict(set)
missing = 0

for path in files:
    parquet = pq.ParquetFile(path)
    if any(column not in parquet.schema_arrow.names for column in columns):
        raise SystemExit(f"[STOP] Missing DDINet metadata columns in {path}")
    for batch in parquet.iter_batches(columns=columns, batch_size=65536):
        values = {
            name: batch.column(batch.schema.get_field_index(name)).to_pylist()
            for name in columns
        }
        for side in ("a", "b"):
            for drug_id, smiles in zip(
                values[f"drugid-drug_{side}"],
                values[f"drugsmiles-drug_{side}"],
                strict=True,
            ):
                if (
                    drug_id is None
                    or smiles is None
                    or not str(drug_id).strip()
                    or not str(smiles).strip()
                ):
                    missing += 1
                    continue
                raw_smiles[str(drug_id).strip()].add(str(smiles).strip())
    print(f"[READ] {path}: rows={parquet.metadata.num_rows}")

invalid = []
conflicts = []
records = {}
for drug_id, candidates in sorted(raw_smiles.items()):
    canonical = set()
    for smiles in candidates:
        try:
            canonical.add(canonicalize_smiles(smiles))
        except ValueError:
            invalid.append((drug_id, smiles))
    if len(canonical) != 1:
        conflicts.append((drug_id, sorted(canonical)))
    elif not invalid or all(item[0] != drug_id for item in invalid):
        records[drug_id] = next(iter(canonical))

print(f"[AUDIT] unique_drugs={len(raw_smiles)} missing_occurrences={missing}")
print(f"[AUDIT] invalid_smiles={len(invalid)} conflicting_drug_ids={len(conflicts)}")
if missing or invalid or conflicts:
    print("invalid examples:", invalid[:10])
    print("conflict examples:", conflicts[:10])
    raise SystemExit("[STOP] Fix/decide the SMILES policy before DDINet training.")

cache = build_ddinet_fingerprint_cache(
    "outputs/ddinet_preflight/fingerprint_cache",
    records,
)
print(f"[OK] cache={cache.root}")
print(f"[OK] drugs={cache.matrix.shape[0]} bits_per_drug={cache.matrix.shape[1]}")
print(f"[OK] pair_input_dim={cache.input_dim}")
PY
```

Kết quả đạt yêu cầu phải có:

```text
missing_occurrences=0
invalid_smiles=0
conflicting_drug_ids=0
bits_per_drug=1024
pair_input_dim=2048
```

## 3. Smoke-test cache, model và fixed head

```bash
python - <<'PY'
import torch

from src.data.ddinet_fingerprints import load_ddinet_fingerprint_cache
from src.models.ddinet_member import build_fixed_head_ddinet_member
from src.training.class_incremental_policy import ClassIncrementalPolicy

cache = load_ddinet_fingerprint_cache("outputs/ddinet_preflight/fingerprint_cache")
drug_ids = [record["drug_id"] for record in cache.manifest["records"][:2]]
if len(drug_ids) < 2:
    raise SystemExit("[STOP] Need at least two drugs for a DDINet smoke test.")

features = torch.from_numpy(cache.ordered_pairs(drug_ids, drug_ids[::-1]))
model = build_fixed_head_ddinet_member().eval()
with torch.no_grad():
    logits = model(features)

policy = ClassIncrementalPolicy.from_raw_class_order(
    list(range(178)), active_raw_classes=list(range(38))
)
masked = policy.mask_logits(logits)
assert logits.shape == (2, 178)
assert torch.isfinite(masked[:, :38]).all()
assert torch.isneginf(masked[:, 38:]).all()
print("[OK] DDINet forward:", tuple(logits.shape))
print("[OK] fixed-head task-0 masking: 38 active / 140 inactive")
PY
```

## 4. Chạy test

```bash
python -m pytest -q \
  tests/test_ddinet_fingerprints.py \
  tests/test_ddinet_member.py \
  tests/test_tddi_fixed_head_policy.py

# Chạy khi có thời gian; local hiện đã đạt 409 passed.
python -m pytest -q
```

## Sau khi các bước trên đạt

Gửi lại toàn bộ output audit/smoke-test. Khi đó mới nối `backbone=ddinet` vào
dataset loader, optimizer, checkpoint provenance và training launcher; chưa dùng
DDINet để train experiment thật trước bước tích hợp này.
