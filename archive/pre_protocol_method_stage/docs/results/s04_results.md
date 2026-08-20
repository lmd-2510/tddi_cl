# Kết quả S04 fixed-budget replay

## 1. Protocol và phạm vi

S04 chạy `replay_distill_fixed_budget_uniform` trên random 8-task protocol, seeds
0–4, MLP-base, focal loss, distillation, batch 1.024, tối đa 20 epochs và patience
5. Cả năm runs được tạo từ clean detached worktree tại commit
`181328e69679b772c475db8c7fc62be016a82411` với `git.dirty=false`.

Hai budget được khóa độc lập:

- total memory: đúng 6.800 unique train exemplars sau mỗi task;
- replay exposure: task 0 bằng 0; task 1–7 đúng 6.800 old-sample draws mỗi epoch.

Validation và test chỉ dùng cho early stopping/evaluation/export. Run config ghi
`buffer_source_policy=current_task_train_split_only`; allocation không dùng
calibration, validation hoặc test.

## 2. Kiểm tra acceptance

- 5/5 runs có `run_completed`, mỗi run đủ 8 tasks;
- 5 O06 files, mỗi file 864 data rows, tổng 4.320 rows và không trùng khóa
  `(run_id, task, raw_class_id)`;
- memory sau mọi task là 6.800; replay thực tế của mọi trained epoch từ task 1 là
  6.800;
- 5 S02 manifests, tổng 80 exports: 40 validation và 40 test;
- không có NaN/Inf trong O07, memory/replay audit hoặc final metrics;
- 39 unit/regression tests pass; CPU smoke 8 tasks đạt memory/replay invariants và
  xuất đủ O01–O06 cùng S02;
- full outputs chiếm khoảng 25 GiB; trước khi chạy còn 115 GiB và sau khi hoàn tất
  còn 87 GiB.

## 3. O07 — final metrics

Kết quả là mean ± sample SD trên năm seeds:

| Metric | S04 fixed budget | Legacy replay-distill |
| --- | ---: | ---: |
| Accuracy | 0.2180 ± 0.0596 | 0.2242 ± 0.0401 |
| Macro-F1 | 0.3761 ± 0.0191 | 0.2632 ± 0.0257 |
| Weighted-F1 | 0.1481 ± 0.0425 | 0.1618 ± 0.0245 |
| Balanced accuracy | 0.4994 ± 0.0433 | 0.6166 ± 0.0170 |
| Task forgetting | 0.3273 ± 0.0578 | 0.2077 ± 0.0338 |
| Final class forgetting | 0.2375 ± 0.0248 | 0.1994 ± 0.0320 |
| Số class F1 = 0 | 6.2 ± 2.2 | 1.8 ± 0.4 |

Paired mean difference S04 − legacy là +0.1129 Macro-F1, −0.1172 balanced
accuracy, +0.1196 task forgetting và +4.4 zero-F1 classes. S04 vì vậy cải thiện
Macro-F1 rõ trong năm runs này nhưng không chi phối legacy trên các metric còn lại.

Đây không phải so sánh cùng budget. Legacy dùng per-class cap 50 và
inverse-class-frequency replacement sampler: storage tăng từ khoảng 1.253–1.563
exemplars ở task 0 lên 6.801 ở task 7, còn expected replay draws thay đổi từ 7.458
đến 144.562 tùy task/seed. S04 giữ 6.800 storage ngay từ task 0 và đúng 6.800 replay
draws từ task 1. Bảng chỉ mô tả chênh lệch thực nghiệm, không cô lập causal effect
của storage hoặc exposure.

## 4. Calibration lại bằng S03

S03 tạo 160 rows:

```text
5 runs × 8 tasks × 2 splits × 2 stages = 160
```

Toàn bộ 40 temperature fits hội tụ, không fit nào chạm boundary. Trên tất cả test
tasks, temperature scaling giảm mean ECE từ 0.3176 xuống 0.0700 và mean NLL từ
3.8059 xuống 2.4984. Riêng task 7 test, mean ECE giảm từ 0.4187 xuống 0.1119 và
mean NLL từ 7.1889 xuống 4.1915. Final temperatures theo seeds nằm trong
`[2.8969, 4.2484]`.

Sau calibration task 7 không còn sample đạt confidence 0.9, nên
high-confidence error rate để trống theo schema; accuracy/prediction argmax không
thay đổi.

## 5. Artifacts

- Runs: `outputs/runs_s04/random_seed<seed>_replay_distill_fixed_budget_uniform_mlpbase`
- O06: `replay_budget_audit.csv` trong từng run
- O07: `outputs/s04/fixed_budget_baseline.csv`
- O07 provenance: `outputs/s04/manifest.json`
- Calibration: `outputs/s04/calibration/calibration_by_task.csv`
- Calibration provenance: `outputs/s04/calibration/manifest.json`

S04 hoàn thành baseline kiểm soát budget cần thiết để triển khai các diagnostics
E01–E03. Các thí nghiệm sau phải dùng S04 làm baseline chính khi câu hỏi nghiên cứu
cần cố định storage và replay exposure.
