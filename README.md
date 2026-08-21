# DDI2025-CIL — T-DDI Protocol Study

Study kết quả gốc vẫn được khóa ở **giai đoạn thiết kế và so sánh protocol** cho dữ liệu DDI2025 mất cân bằng. Biến độc lập duy nhất của bảng P0–P8 bên dưới là cách phân 178 lớp thành 8 task; backbone T-DDI và phương pháp học được giữ cố định. Một study mở rộng riêng cho TabM/DDI-GCN được mô tả ở phần sau và không được trộn số liệu với bảng này.

## Phạm vi đang khóa

- Backbone duy nhất: `tddi`, MLP `3780 → 1024 → 512 → head mở rộng`.
- Method duy nhất để so protocol: `replay_distill_fixed_budget_uniform`.
- Task layout duy nhất: `[38, 20, 20, 20, 20, 20, 20, 20]`.
- Protocol đang xét: P0–P8.
- Mỗi protocol chạy 5 training seed: `0, 1, 2, 3, 4`.
- Protocol chính để báo cáo: **P4 — constrained mass-balanced**.
- P0 là random reference; P2/P3 là cặp stress test; P1 là baseline cân bằng đơn giản; P5–P8 là các thiết kế nâng cao.
- Không phát triển method mới. So sánh backbone chỉ được mở trong ma trận riêng P4/P6/P0/P2/P3, dùng cùng method và ngân sách của study gốc.

Mọi thay đổi làm biến thiên backbone, method, loss, memory, optimizer, task layout hoặc split phải được coi là **một study khác**, không được gộp vào kết quả P0–P8 hiện tại.

## Kết quả protocol hiện tại

Kết quả tại task cuối, biểu diễn bằng mean ± sample standard deviation trên 5 seed. Task forgetting lấy đúng dòng `mean_old_tasks` trong `forgetting.csv`.

| Protocol | Macro-F1 ↑ | Balanced Acc. ↑ | Task Forgetting ↓ | Vai trò |
|---|---:|---:|---:|---|
| P0 Random | 0.3769 ± 0.0194 | 0.5030 ± 0.0390 | 0.3213 ± 0.0418 | Reference |
| P1 Frequency-balanced | 0.3585 ± 0.0081 | 0.5189 ± 0.0153 | 0.3236 ± 0.0126 | Baseline |
| **P2 Head→tail** | 0.2719 ± 0.0050 | **0.7642 ± 0.0170** | **0.0441 ± 0.0011** | Tốt nhất về balanced accuracy và forgetting |
| **P3 Tail→head** | **0.4935 ± 0.0090** | 0.4530 ± 0.0132 | 0.5003 ± 0.0107 | Macro-F1 cao nhất nhưng forgetting tệ nhất |
| *P4 Mass-balanced* | *0.3825 ± 0.0246* | *0.4795 ± 0.0250* | *0.3562 ± 0.0283* | *Protocol chính nên lựa chọn* |
| P5 Multi-factor | 0.3791 ± 0.0167 | 0.4785 ± 0.0170 | 0.3570 ± 0.0270 | Nâng cao |
| **P6 Difficulty-balanced** | **0.3831 ± 0.0184** | **0.4947 ± 0.0191** | **0.3438 ± 0.0324** | Tốt nhất nhóm P4–P8, nhưng model-informed |
| P7 Confusion-spread | 0.3805 ± 0.0206 | 0.4792 ± 0.0140 | 0.3668 ± 0.0143 | Nâng cao |
| P8 Rarity-drift | 0.3759 ± 0.0238 | 0.4788 ± 0.0136 | 0.3529 ± 0.0127 | Stress test rarity |

**In đậm** đánh dấu kết quả nổi bật; *in nghiêng* đánh dấu protocol chính được lựa chọn. P2/P3 chỉ là stress test cực đoan. P6 là lựa chọn model-informed tốt nhất, còn P4 vẫn là protocol chính trung lập vì không phụ thuộc checkpoint/model để chia task.

## Thứ tự đọc bắt buộc

Để không lệch định hướng, đọc theo đúng thứ tự sau:

1. `README.md` — phạm vi và quy tắc khóa hiện tại.
2. `docs/PROTOCOL_STUDY.md` — câu hỏi nghiên cứu, vai trò từng protocol và tiêu chí chọn protocol chính.
3. `configs/protocol_study_tddi.json` — nguồn sự thật dạng máy cho toàn bộ cấu hình khóa.
4. `docs/EXPERIMENTS.md` — diễn giải đầy đủ dữ liệu, preprocessing, model, training, replay, protocol và metric.
5. `docs/RESULTS.md` — kết quả 5 seed và kết luận hiện tại.
6. `docs/BACKBONE_STUDY.md` — study mở rộng riêng cho TabM/DDI-GCN và giới hạn diễn giải P6.
7. `configs/selected_backbone_protocol_study.json` — cấu hình máy đọc được của study backbone.
8. `docs/RUNBOOK.md` — cách kiểm tra môi trường, tạo task và chạy bằng tmux.
9. `CIL.md` — chỉ đọc khi cần nền tảng kỹ thuật về CIL; tài liệu này **không được dùng để tự ý mở rộng ma trận hiện tại**.

Không dùng nội dung trong `archive/` để quyết định cấu hình hiện tại. Archive chỉ phục vụ truy vết lịch sử.

## Chạy study

Chạy một protocol:

```bash
tmux new-session -d -s ddi_p4 \
  "cd '$PWD' && bash scripts/run_protocol_study.sh P4 2>&1 | tee outputs/runs_backbones/p4_tddi_driver.log"
```

Chạy toàn bộ P0–P8 (runner tự bỏ qua run đã hoàn tất):

```bash
tmux new-session -d -s ddi_protocols \
  "cd '$PWD' && bash scripts/run_protocol_study.sh all 2>&1 | tee outputs/runs_backbones/protocol_study_tddi_driver.log"
```

Theo dõi:

```bash
tmux attach -t ddi_protocols
tail -f outputs/runs_backbones/protocol_study_tddi_driver.log
```

Runner có chặn thiếu RAM/ổ đĩa, không ghi đè run dở và không nhận lựa chọn backbone. Xem toàn bộ quy trình tại `docs/RUNBOOK.md`.

## Study TabM và DDI-GCN

Ma trận mở rộng dùng đúng P4, P6, P0, P2, P3 trên hai backbone. Tất cả cấu hình CIL chung được giữ nguyên; microbatch là 256 và tích lũy 4 bước để effective batch vẫn là 1,024, tránh tràn GPU 16 GiB. P6 vẫn kế thừa lịch chia task được tạo từ static T-DDI nên chỉ là nhánh model-informed, không phải phép so sánh backbone-neutral.

Pilot seed 0 đã hoàn tất đủ 10/10 run. Bảng dưới là kết quả **sơ bộ của một seed**, chỉ dùng để xác nhận pipeline và định hướng; không dùng để kết luận protocol trước khi hoàn thành seed 1–4.

| Backbone | Protocol | Macro-F1 ↑ | Balanced Acc. ↑ | Task Forgetting ↓ |
|---|---|---:|---:|---:|
| TabM | P4 Mass-balanced | 0.3792 | 0.4692 | 0.3222 |
| TabM | P6 Difficulty-balanced | 0.3702 | 0.5121 | 0.3174 |
| TabM | P0 Random | 0.3601 | 0.5886 | 0.2318 |
| TabM | **P2 Head→tail** | 0.1775 | **0.6965** | **0.0808** |
| TabM | **P3 Tail→head** | **0.4192** | 0.3932 | 0.5746 |
| DDI-GCN | P4 Mass-balanced | 0.3669 | 0.4819 | 0.1081 |
| DDI-GCN | P6 Difficulty-balanced | 0.4603 | 0.5645 | 0.0844 |
| DDI-GCN | P0 Random | 0.4640 | 0.6335 | 0.1049 |
| DDI-GCN | **P2 Head→tail** | 0.2862 | **0.7606** | **0.0479** |
| DDI-GCN | **P3 Tail→head** | **0.6057** | 0.5608 | 0.3777 |

Ở seed 0, P3 có Macro-F1 cao nhất và P2 có balanced accuracy cao nhất/forgetting thấp nhất trên cả hai backbone. Đây vẫn là hai stress test cực đoan; quyết định protocol chính phải dựa trên mean ± sample standard deviation đủ 5 seed và giữ riêng lưu ý model-informed của P6.

Chạy bốn seed còn lại; runner tự bỏ qua mọi run hoàn tất:

```bash
tmux new-session -d -s backbone_full \
  "cd '$PWD' && mkdir -p outputs/runs_selected_backbones && BACKBONE_SEEDS='1 2 3 4' bash scripts/run_selected_backbone_protocols.sh all all 2>&1 | tee outputs/runs_selected_backbones/full_driver.log"
```

Xem contract, cấu hình hai model, resource guard và lệnh chạy seed 1–4 tại `docs/BACKBONE_STUDY.md`.

## Cấu trúc đang dùng

```text
configs/protocol_study_tddi.json   cấu hình khóa của study
docs/PROTOCOL_STUDY.md             định hướng nghiên cứu
docs/EXPERIMENTS.md                mô tả đầy đủ thử nghiệm
docs/RESULTS.md                    bảng kết quả và quyết định
docs/RUNBOOK.md                    lệnh tái lập
docs/BACKBONE_STUDY.md             study TabM/DDI-GCN riêng
scripts/run_protocol_study.sh      runner P0–P8, chỉ T-DDI
scripts/run_selected_backbone_protocols.sh runner P4/P6/P0/P2/P3, TabM/DDI-GCN
scripts/build_molecular_graph_cache.py tạo graph cache DDI-GCN
scripts/build_cil_tasks.py         tạo P0–P4
scripts/prepare_advanced_protocol_signals.py
scripts/build_advanced_protocols.py tạo P5–P8
src/training/train_cil.py          training engine
archive/pre_protocol_method_stage/ tài liệu/entrypoint/output lịch sử
```

## Dữ liệu và môi trường

Ba split parquet không được track bởi git: `train_extracted.parquet`, `validation_extracted.parquet`, `test_extracted.parquet`. Môi trường Python nằm tại `.venv`; dependency được ghi trong `requirements.txt`.

Để kiểm tra nhanh contract của study:

```bash
.venv/bin/python -m pytest -q
bash -n scripts/run_protocol_study.sh
bash -n scripts/run_selected_backbone_protocols.sh
```
