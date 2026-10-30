# MFP-Siamese server validation

MFP-Siamese đã có backbone, cache Morgan 2048 bit và fixed head 178 lớp nhưng
chưa nối vào training launcher. Chỉ chạy các bước này sau khi DDINet SMILES
audit trong `docs/DDINET_SERVER_VALIDATION.md` đã đạt.

## 1. Tạo cache riêng từ danh sách thuốc đã audit

```bash
cd "$HOME/DrugDrug/cil-tddi/cil-tddi"
conda activate ai_env
python -m pip install -r requirements.txt

python - <<'PY'
from src.data.ddinet_fingerprints import load_ddinet_fingerprint_cache
from src.data.morgan_fingerprints import (
    MorganFingerprintConfig,
    build_morgan_fingerprint_cache,
)

source = load_ddinet_fingerprint_cache(
    "outputs/ddinet_preflight/fingerprint_cache"
)
records = {
    record["drug_id"]: record["canonical_smiles"]
    for record in source.manifest["records"]
}
config = MorganFingerprintConfig(
    radius=2,
    n_bits=2048,
    include_chirality=False,
    consumer="mfp_siamese",
)
cache = build_morgan_fingerprint_cache(
    "outputs/mfp_siamese_preflight/fingerprint_cache",
    records,
    config=config,
)
print("[OK] drugs:", cache.matrix.shape[0])
print("[OK] bits per drug:", cache.matrix.shape[1])
print("[OK] ordered pair dim:", cache.input_dim)
PY
```

Kỳ vọng: `bits per drug: 2048`, `ordered pair dim: 4096`.

## 2. Targeted tests

```bash
python -m pytest -q \
  tests/test_mfp_siamese_member.py \
  tests/test_ddinet_fingerprints.py \
  tests/test_tddi_fixed_head_policy.py
```

Sau khi hai bước đạt, gửi lại output. Khi đó mới nối `mfp_siamese_member` vào
loader, checkpoint và launcher; checkpoint tốt nhất sẽ chọn theo validation
Macro-F1 thay vì accuracy của mã paper.
