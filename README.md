# DDI-CIL — T-DDI Paper Member Ensemble

Repository hiện tập trung vào study:

```text
replay_distill_fixed_budget_uniform
× tddi_paper_member ensemble 3
× P3 tail-to-head
× experiment seed 0
```

Backbone T-DDI numerical duy nhất dùng cho study chính:

```text
LayerNorm(3780)
-> Linear(7560) -> GELU -> Dropout
-> Linear(7560) -> GELU -> Dropout
-> expandable class head
```

Alias MLP T-DDI cũ đã bị loại bỏ. Các preset `small`, `base`, `large` còn lại chỉ là
MLP baseline generic và không được gọi là T-DDI.

## Config chính

```text
configs/tddi_ensemble3_replay_distill_p3_seed0.json
```

Task schedule:

```text
outputs/tasks/tail_to_head_tasks.json
```

P3 có 8 task với layout `[38,20,20,20,20,20,20,20]`, tổng cộng 178 raw class.
Experiment seed 0 giữ study metadata; P3 task order là deterministic và task-file có
`seed: null` vì schedule tail-to-head không dùng random permutation.

## Dry-run

Dry-run không tạo model và không train:

```bash
python src/training/tddi_ensemble3_study.py \
  --config configs/tddi_ensemble3_replay_distill_p3_seed0.json \
  --member-id 0
```

Chỉ thêm `--execute` khi thực sự muốn chạy trên máy GPU.

## Tài liệu nên đọc

1. `docs/TDDI_PAPER_REPLAY_DISTILL_P3_8TASK_GPU_RUNBOOK.md` — lệnh full run từng member.
2. `docs/TDDI_REPLAY_DISTILL_ENSEMBLE3_FILES_GUIDE.md` — tác dụng của từng file.
3. `docs/TDDI_ENSEMBLE3_EWC_PLAN.md` — lịch sử thiết kế paper-size member/EWC.

Ba member phải chạy tuần tự. Sau khi đủ ba prediction artifact, offline ensemble lấy
mean probabilities và UE audit báo cáo entropy, mutual information, variance,
disagreement, error-detection AUROC và AURC.
