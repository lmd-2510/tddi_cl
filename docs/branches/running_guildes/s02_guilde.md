# Hướng dẫn chạy S02 prediction và latent export

S02 là opt-in và chỉ áp dụng cho run directory mới. Không backfill 20 legacy S01 runs,
không chạy lại vào directory đã có nội dung.

## 1. CLI

Bật export cho cả validation và test:

```bash
--export-s02 --s02-splits validation test
```

Khi có `--export-s02` nhưng bỏ `--s02-splits`, mặc định vẫn là `validation test`.
Có thể chỉ export validation bằng:

```bash
--export-s02 --s02-splits validation
```

Không có export train. Test artifacts chỉ dùng reporting; calibration, uncertainty và
replay allocation phải lấy từ validation.

## 2. CPU smoke 8 tasks

Chạy từ thư mục gốc `DDI-CIL` với một output directory mới:

```bash
.venv/bin/python src/training/train_cil.py \
  --train train_extracted.parquet \
  --validation validation_extracted.parquet \
  --test test_extracted.parquet \
  --feature-cols outputs/audit_smoke/feature_columns.json \
  --scaler outputs/preprocess_smoke/scaler.pkl \
  --task-file outputs/tasks_smoke/random_seed0_tasks.json \
  --outdir /tmp/ddi-s02-smoke \
  --method replay_distill \
  --variant small \
  --batch-size 64 \
  --epochs 1 \
  --patience 1 \
  --seed 0 \
  --device cpu \
  --max-train-rows-per-task 64 \
  --max-validation-rows-per-task 64 \
  --max-test-rows-per-task 64 \
  --export-s02 \
  --s02-splits validation test
```

Smoke chỉ xác minh pipeline. Không dùng artifact bị giới hạn rows để phân tích nghiên cứu.

## 3. Layout và schema

```text
<run>/
├── checkpoints/
├── memory/
├── class_trajectory.csv
└── s02/
    ├── manifest.json
    ├── task_0/
    │   ├── validation/
    │   │   ├── predictions.parquet
    │   │   └── latent_features.npz
    │   └── test/
    │       ├── predictions.parquet
    │       └── latent_features.npz
    └── task_7/...
```

Với 8 tasks và hai splits, một run hoàn chỉnh có:

```text
16 predictions.parquet
16 latent_features.npz
1 manifest.json
```

`manifest.json` dùng `schema_version=1` và lưu run provenance, checkpoint hash,
task/split inventory, class order, row counts và latent dimension.

O03 gồm:

```text
run_id, seed, method, method_protocol, train_task, split,
sample_id, drug_id_a, drug_id_b, label, prediction,
confidence, entropy, class_ids, logits, probabilities
```

O04 gồm các arrays/scalars:

```text
latent_features, sample_ids, labels, drug_ids_a, drug_ids_b, class_ids,
run_id, seed, method, method_protocol, train_task, split,
checkpoint_sha256, schema_version
```

Khóa join là `(run_id, train_task, split, sample_id)` và
`sample_id = <drug_id_a>|<drug_id_b>`.

## 4. Kiểm tra artifacts

Chạy unit tests trước:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

Đếm data files:

```bash
find /tmp/ddi-s02-smoke/s02 -name predictions.parquet | wc -l
find /tmp/ddi-s02-smoke/s02 -name latent_features.npz | wc -l
```

Kiểm tra mọi O03/O04 pair trong manifest:

```bash
.venv/bin/python - <<'PY'
import json
from pathlib import Path

from src.eval.s02_artifacts import validate_s02_artifacts

root = Path("/tmp/ddi-s02-smoke/s02")
manifest = json.loads((root / "manifest.json").read_text())
if manifest["schema_version"] != 1 or len(manifest["exports"]) != 16:
    raise SystemExit("Invalid S02 manifest inventory")

for export in manifest["exports"]:
    validate_s02_artifacts(
        root / export["predictions"],
        root / export["latent_features"],
    )

print("All S02 shards are valid.")
PY
```

Một shard chỉ được ghi event `s02_exported` sau khi O03, O04 và manifest đã qua
validation. Export từ chối duplicate drug pair, provenance sai, NaN/Inf, sai dtype,
sai vector width, O03/O04 lệch hàng hoặc shard đã tồn tại.

## 5. Clean research runs

Không thêm S02 vào các directory `outputs/runs_s01_*` hiện có. Khi chạy clean S04,
giữ cùng training protocol và thêm:

```bash
--export-s02 --s02-splits validation test
```

Như vậy O01–O04 được sinh trong cùng execution và cùng `run_id`, tránh ghép artifact
từ checkpoint hoặc protocol khác nhau. Theo dõi dung lượng trước khi mở rộng full
dataset vì logits/probabilities tăng theo số seen classes; sharding theo task/split chỉ
giới hạn peak RAM, không làm giảm tổng dung lượng.
