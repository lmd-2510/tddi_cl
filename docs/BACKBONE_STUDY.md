# Selected-Protocol Backbone Study

Đây là study mở rộng riêng để kiểm tra khả năng khái quát của các protocol đã chọn trên `TabM` và `DDI-GCN`. Không gộp kết quả của study này vào bảng T-DDI P0–P8 trước khi ghi rõ backbone.

## Câu hỏi và ma trận

- Protocol theo thứ tự chạy: P4, P6, P0, P2, P3.
- Backbone: TabM và DDI-GCN.
- Pilot: seed 0, tổng cộng 10 run.
- Full study sau khi pilot hợp lệ: seed 0–4, tổng cộng 50 run; runner tự bỏ qua seed 0 đã xong.
- Nguồn cấu hình máy đọc được: `configs/selected_backbone_protocol_study.json`.

P4 là ứng viên chính trung lập. P0 là reference; P2/P3 là stress-test đối xứng. P6 được giữ để kiểm tra khả năng chuyển giao nhưng phải báo cáo là **model-informed từ static T-DDI**, nên không dùng riêng P6 để tuyên bố một protocol độc lập với backbone.

## Phần giữ cố định

- Split train/validation/test: 520,841 / 173,614 / 173,614 dòng, không trùng drug pair.
- 178 lớp và 8 task `[38, 20, 20, 20, 20, 20, 20, 20]`; dùng nguyên task JSON hiện có.
- Method: `replay_distill_fixed_budget_uniform`.
- AdamW, learning rate `0.001`, weight decay `0.0001`.
- Tối đa 20 epoch/task, patience 5 theo macro-F1 validation trên toàn bộ lớp đã thấy.
- Focal loss gamma 1; logit distillation alpha 1, temperature 2; feature MSE weight 0.5.
- Replay memory 6,800 exemplar và 6,800 replay draw/epoch sau task 0.
- Microbatch 256, tích lũy gradient 4 bước, effective batch 1,024, float32.
- Năm training seed: 0–4.

Replay của DDI-GCN lưu cặp chỉ số graph để model forward. Tuy nhiên exemplar được xếp hạng bằng khoảng cách tới class mean trong cùng không gian 3,780 descriptor đã chuẩn hóa như TabM/T-DDI. Vì vậy cùng task file và seed sẽ giữ cùng chính sách chọn exemplar; chỉ đầu vào backbone thay đổi.

## Backbone

### TabM

Adapter dùng package chính thức [`tabm`](https://github.com/yandex-research/tabm), input 3,780 descriptor, `k=32`, 3 block, width 512, dropout 0.1. Khi train, loss được tính độc lập cho mọi member. Khi validation/test, xác suất của 32 member được lấy trung bình. Head `LinearEnsemble` được mở rộng theo các lớp đã thấy và sao chép nguyên các cột lớp cũ.

### DDI-GCN

Implementation là bản port PyTorch của categorical model từ [`LabWeng/DDI-GCN`](https://github.com/LabWeng/DDI-GCN): atom feature 51, bond feature 10, tối đa 65 atom, degree 0–5, 8 graph-convolution layer width 128, BiGRU qua layer states, co-attention dimension 65 và merger `100→100→100`. Output gốc được thay bằng head mở rộng đến 178 lớp để chạy CIL.

Graph cache được tạo chỉ từ drug ID/SMILES có trong ba split. Dataset hiện có 2,957 drug, không có SMILES lỗi và không có molecule vượt 65 atom.

## Tạo graph cache

```bash
.venv/bin/python scripts/build_molecular_graph_cache.py \
  --splits train_extracted.parquet validation_extracted.parquet test_extracted.parquet \
  --outdir outputs/graph_cache/ddi_gcn \
  --max-atoms 65
```

Script từ chối ghi đè cache đã có. Cache là artifact cục bộ và không được đưa lên git.

## Chạy an toàn bằng tmux

Pilot seed 0:

```bash
tmux new-session -d -s backbone_pilot \
  "cd '$PWD' && mkdir -p outputs/runs_selected_backbones && BACKBONE_SEEDS=0 bash scripts/run_selected_backbone_protocols.sh all all 2>&1 | tee outputs/runs_selected_backbones/pilot_driver.log"
```

Theo dõi:

```bash
tmux attach -t backbone_pilot
tail -f outputs/runs_selected_backbones/pilot_driver.log
```

Sau khi kiểm tra đủ 10 run seed 0, chạy phần còn lại:

```bash
tmux new-session -d -s backbone_full \
  "cd '$PWD' && mkdir -p outputs/runs_selected_backbones && BACKBONE_SEEDS='1 2 3 4' bash scripts/run_selected_backbone_protocols.sh all all 2>&1 | tee outputs/runs_selected_backbones/full_driver.log"
```

Runner chạy tuần tự, yêu cầu tối thiểu 12 GiB RAM khả dụng, 50 GiB ổ đĩa trống và 10,000 MiB VRAM trống trước mỗi run. Run hoàn tất được bỏ qua; thư mục run dở không bị ghi đè.

## Điều kiện qua pilot

Chỉ mở seed 1–4 khi cả 10 run seed 0 có đủ `run_summary.md`, `metrics.csv`, `forgetting.csv`, không có NaN/Inf trong metric chính và không có lỗi resource. So sánh protocol cuối cùng phải dùng mean ± sample standard deviation trên đủ 5 seed, không chọn từ seed 0.
