# Loss pilots

Hai pilot này chỉ thay đổi cách tính loss; không đổi protocol P4, backbone,
buffer, sampler, seed hay replay fraction. Cả hai chạy tuần tự trên `member_0`
và chỉ đến task 1.

| Pilot | Loss | Output |
|---|---|---|
| ER | Cross-Entropy trên current + replay; không distillation | `outputs/loss_pilot_er_p4_t01` |
| Hybrid | Focal trên current, Cross-Entropy trên replay; không distillation | `outputs/loss_pilot_hybrid_p4_t01` |

Baseline cũ không bị sửa. `loss_variant=baseline` vẫn là Focal + logit
distillation + feature distillation.

Trên máy GPU:

```bash
conda activate ai_env
export FOLD_ROOT="$PWD/outputs/fold_preparation_seed42_20260916_230549/folds"
export PREP_ROOT="$PWD/study_assets/preprocessing_p4_seed0_fold42_old"

# Dry-run cả hai pilot, không tạo model:
bash scripts/run_loss_pilots.sh dry-run

# Chạy tuần tự ER rồi Hybrid:
nohup env CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
  bash scripts/run_loss_pilots.sh --execute \
  > outputs/loss_pilots.nohup.log 2>&1 < /dev/null &
echo $! | tee outputs/loss_pilots.pid
```

Theo dõi:

```bash
tail -f outputs/loss_pilots.nohup.log
```

Sau khi PID kết thúc, xem `run_summary.md`, `metrics.csv`,
`training_audit.csv` trong từng thư mục `member_0`. Không chạy ensemble ở bước
này vì đây là pilot loss trên một member; chỉ chạy full 3 member sau khi chọn
được variant tốt hơn baseline.
