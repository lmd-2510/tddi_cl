# Runbook hiện hành

Study chính của repository hiện là:

```text
replay_distill_fixed_budget_uniform
× tddi_paper_member ensemble 3
× P3 tail-to-head
× experiment seed 0
```

Không còn variant MLP cũ mang tên `tddi`.

Để setup môi trường, kiểm tra P3, chạy từng member bằng `nohup`, resume, ensemble, UE
và threshold, dùng tài liệu:

```text
docs/TDDI_PAPER_REPLAY_DISTILL_P3_8TASK_GPU_RUNBOOK.md
```

Dry-run nhanh:

```bash
python src/training/tddi_ensemble3_study.py \
  --config configs/tddi_ensemble3_replay_distill_p3_seed0.json \
  --member-id 0
```

Lệnh trên không train. Training chỉ bắt đầu khi thêm `--execute`.
