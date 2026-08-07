# Hướng dẫn chạy S03 calibration

S03 là post-hoc calibration trên O03 của S02, không train lại classifier. Mỗi
`(run_id, train_task)` fit một scalar temperature bằng validation NLL. Test artifacts
chỉ được đọc sau khi temperature đã khóa và chỉ dùng để báo cáo.

## Chạy full S03

Chạy từ thư mục gốc `DDI-CIL` sau khi đủ 20 S02 runs:

```bash
.venv/bin/python src/eval/run_s03_calibration.py \
  --runs-root outputs/runs_s02 \
  --outdir outputs/s03
```

Mặc định:

- 15 equal-width ECE bins;
- high-confidence threshold 0.9;
- temperature nằm trong `[0.05, 100]`;
- xử lý batch 8.192 hàng;
- từ chối output directory không rỗng.

Chỉ dùng `--overwrite` khi chủ đích tái tạo O05 từ đầu.

## Output

```text
outputs/s03/
├── calibration_by_task.csv
├── manifest.json
└── run.log
```

Full matrix có 640 hàng:

```text
20 runs × 8 tasks × 2 splits × 2 stages = 640
```

Hai stages là `raw` và `temperature_scaled`; hai splits là `validation` và `test`.
Các metric gồm accuracy, ECE, Brier, NLL, mean confidence và high-confidence error
rate. Accuracy trước/sau scaling phải giống hệt. Nếu không có mẫu confidence ≥ 0.9,
error rate được ghi rỗng và `high_confidence_count=0`.

Manifest khóa source S02 manifest hashes, implementation hashes, fit split, ECE bins,
threshold và output hash. Không cần đọc O04 latent features khi chạy S03.

## Kiểm tra

```bash
.venv/bin/python -m unittest discover -s tests -v
wc -l outputs/s03/calibration_by_task.csv
jq '{run_count,row_count,temperature_fit_split,test_usage}' outputs/s03/manifest.json
```

Kỳ vọng: 29 tests pass, CSV có 641 dòng gồm header, `run_count=20`,
`row_count=640`, fit split là validation và test usage là reporting only.
