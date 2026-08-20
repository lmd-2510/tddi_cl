# DDI2025-CIL — T-DDI Protocol Study

Repo hiện được khóa ở **giai đoạn thiết kế và so sánh protocol** cho dữ liệu DDI2025 mất cân bằng. Biến độc lập duy nhất là cách phân 178 lớp thành 8 task; backbone và phương pháp học được giữ cố định.

## Phạm vi đang khóa

- Backbone duy nhất: `tddi`, MLP `3780 → 1024 → 512 → head mở rộng`.
- Method duy nhất để so protocol: `replay_distill_fixed_budget_uniform`.
- Task layout duy nhất: `[38, 20, 20, 20, 20, 20, 20, 20]`.
- Protocol đang xét: P0–P8.
- Mỗi protocol chạy 5 training seed: `0, 1, 2, 3, 4`.
- Protocol chính để báo cáo: **P4 — constrained mass-balanced**.
- P0 là random reference; P2/P3 là cặp stress test; P1 là baseline cân bằng đơn giản; P5–P8 là các thiết kế nâng cao.
- Chưa mở lại so sánh backbone hoặc phát triển method. Các implementation cũ vẫn tồn tại để tái lập lịch sử nhưng không thuộc ma trận hiện tại.

Mọi thay đổi làm biến thiên backbone, method, loss, memory, optimizer, task layout hoặc split phải được coi là **một study khác**, không được gộp vào kết quả P0–P8 hiện tại.

## Thứ tự đọc bắt buộc

Để không lệch định hướng, đọc theo đúng thứ tự sau:

1. `README.md` — phạm vi và quy tắc khóa hiện tại.
2. `docs/PROTOCOL_STUDY.md` — câu hỏi nghiên cứu, vai trò từng protocol và tiêu chí chọn protocol chính.
3. `configs/protocol_study_tddi.json` — nguồn sự thật dạng máy cho toàn bộ cấu hình khóa.
4. `docs/EXPERIMENTS.md` — diễn giải đầy đủ dữ liệu, preprocessing, model, training, replay, protocol và metric.
5. `docs/RESULTS.md` — kết quả 5 seed và kết luận hiện tại.
6. `docs/RUNBOOK.md` — cách kiểm tra môi trường, tạo task và chạy bằng tmux.
7. `CIL.md` — chỉ đọc khi cần nền tảng kỹ thuật về CIL; tài liệu này **không được dùng để tự ý mở rộng ma trận hiện tại**.

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

## Cấu trúc đang dùng

```text
configs/protocol_study_tddi.json   cấu hình khóa của study
docs/PROTOCOL_STUDY.md             định hướng nghiên cứu
docs/EXPERIMENTS.md                mô tả đầy đủ thử nghiệm
docs/RESULTS.md                    bảng kết quả và quyết định
docs/RUNBOOK.md                    lệnh tái lập
scripts/run_protocol_study.sh      runner P0–P8, chỉ T-DDI
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
.venv/bin/python -m unittest discover -s tests -p 'test_*.py' -v
bash -n scripts/run_protocol_study.sh
```
