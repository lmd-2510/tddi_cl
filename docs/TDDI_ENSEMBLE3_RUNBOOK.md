# T-DDI Ensemble3 EWC runner

Study này dùng config `configs/tddi_ensemble3_ewc_p4_seed0.json`. Entrypoint mặc định
chỉ in kế hoạch, không khởi chạy job tám task:

```powershell
python src/training/tddi_ensemble3_study.py `
  --config configs/tddi_ensemble3_ewc_p4_seed0.json
```

Để chạy hoặc resume riêng một trajectory trên máy GPU:

```powershell
python src/training/tddi_ensemble3_study.py `
  --config configs/tddi_ensemble3_ewc_p4_seed0.json `
  --execute --member-id 0
```

Chạy member `0`, `1`, `2` lần lượt. Nếu member có
`checkpoints/latest_ewc_state.pt`, runner tự thêm `--resume-ewc-checkpoint`. Nếu run đã
có đủ `run_summary.md`, `metrics.csv`, `forgetting.csv` và prediction artifacts, runner
skip và không ghi đè. Directory không hoàn tất mà không có checkpoint sẽ bị từ chối.

Khi cả ba member hoàn tất, lần gọi `--execute` cuối cùng kiểm tra sample IDs, labels,
protocol, task, seed và `raw_class_ids`, sau đó gọi offline ensemble lần lượt cho từng
task/split và tạo `study_manifest.json` liên kết cả ba artifact. Có thể bỏ
`--member-id` để chủ động chạy cả ba trajectory tuần tự; đây là thao tác tốn nhiều thời
gian và chỉ nên thực hiện sau smoke test trên máy GPU riêng.
