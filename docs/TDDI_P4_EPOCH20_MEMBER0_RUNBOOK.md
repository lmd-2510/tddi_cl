# Chạy P4 Ensemble3 bằng một lệnh

Study dùng P4 `constrained_mass_balanced`, 20 epoch/task và baseline
`replay_distill_fixed_budget_uniform × tddi_paper_member`.

```text
member 0 (task 0–7)
→ member 1 (task 0–7)
→ member 2 (task 0–7)
→ ensemble + UE + threshold
→ final report
```

Mỗi thời điểm chỉ có một member trên GPU. Nếu job bị ngắt, gọi lại `start`: controller
skip member đã xong và resume member dở từ task boundary gần nhất.

## 1. Cập nhật repo

```bash
cd "$HOME/DrugDrug/cil-tddi/cil-tddi"
conda activate ai_env
git pull
python -m pytest tests/test_fold_ensemble3_full.py tests/test_build_final_report.py -q
```

## 2. Kiểm tra input và tạo scaler

```bash
bash scripts/run_p4_ensemble3.sh check
bash scripts/run_p4_ensemble3.sh prepare
```

`prepare` tạo scaler B riêng cho ba member và tự skip file đã hoàn tất. Nếu script tìm
thấy nhiều fold root, chỉ định đúng thư mục:

```bash
export P4_FOLD_ROOT="$PWD/outputs/fold_preparation_seed42_20260912_161002/folds"
```

## 3. Dry-run và bắt đầu

```bash
bash scripts/run_p4_ensemble3.sh dry-run
bash scripts/run_p4_ensemble3.sh start
```

`start` dùng CUDA 0, chạy member 0 → 1 → 2 tuần tự, sau đó tự chạy ensemble/UE,
threshold và tạo final report. Có thể ngắt SSH hoặc tắt laptop; server GPU phải còn
bật. Muốn dùng GPU khác:

```bash
P4_GPU_ID=1 bash scripts/run_p4_ensemble3.sh start
```

## 4. Theo dõi

```bash
bash scripts/run_p4_ensemble3.sh status
bash scripts/run_p4_ensemble3.sh follow
```

`Ctrl+C` chỉ thoát `follow`. Log controller nằm trong các thư mục dễ đọc `run_001`,
`run_002`,... tại:

```text
outputs/stratified_ensemble3_monitor/full_p4_seed0_8tasks/
```

Log riêng từng member nằm trong `$FULL_ROOT/member_0/stdout.log`, `member_1/stdout.log`
và `member_2/stdout.log`, với `FULL_ROOT` là:

```text
outputs/stratified_ensemble3/full_p4_seed0_8tasks
```

## 5. Resume, kiểm tra và tạo lại kết quả cuối

Nếu job đã dừng trước khi hoàn tất, sửa nguyên nhân rồi gọi lại:

```bash
bash scripts/run_p4_ensemble3.sh start
```

Không xóa output. Khi `status` báo `exit_code=0`, kiểm tra artifacts:

```bash
bash scripts/run_p4_ensemble3.sh verify
```

`start` đã tự tạo final report. Nếu chỉ muốn tạo lại kết quả cuối từ artifacts mà
không train và không inference, dùng lệnh riêng:

```bash
bash scripts/run_p4_ensemble3.sh report
```

Lệnh Python tương đương là:

```bash
python src/eval/report.py \
  --full-root outputs/stratified_ensemble3/full_p4_seed0_8tasks \
  --task-file study_assets/task_protocols/constrained_mass_balanced_seed0_tasks.json \
  --overwrite
```

Kết quả cuối:

```text
outputs/stratified_ensemble3/full_p4_seed0_8tasks/final_results/
  ensemble3_p4_final_report.md
  ensemble3_p4_task_summary.csv
  ensemble3_p4_paper_table.csv
  ensemble3_p4_task_matrix.csv
  ensemble3_p4_forgetting_summary.csv
  ensemble3_p4_member_forgetting_summary.csv
  ensemble3_p4_diversity_summary.csv
```
