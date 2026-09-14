# Prompt 10 — Checkpoint/resume cho replay 3-fold A/B

Chỉ áp dụng khi bật `--fold-replay-policy stratified_fraction_v1`.
Không thay schema/CLI của EWC hoặc fixed-budget replay legacy. Không train dữ liệu
thật trong lần triển khai này, không tự push GitHub.

## Hiểu đơn giản

Sau khi học xong một task, trainer lưu một “điểm tiếp tục”. Nếu dừng sau task 0,
khi chạy lại sẽ bắt đầu task 1 với đúng weights, exemplar, preprocessing và RNG
đã có. Nếu gián đoạn giữa task 1, **học lại task 1 từ đầu** từ checkpoint task 0;
không giả định tiếp tục được từ batch/epoch đang dở.

## File thay đổi

| File | Tác dụng |
| --- | --- |
| `src/training/replay_checkpoint.py` | Thêm API `save_fold_replay_checkpoint` / `load_fold_replay_checkpoint`, schema riêng, checksum state, atomic no-overwrite publication, validation artifact và restore buffer. Các API legacy không đổi. |
| `src/training/fold_pilot_training.py` | Lưu boundary sau best-model evaluation và cập nhật buffer; resume đúng next task, dựng teacher từ best model, khôi phục RNG sau dựng model; giữ nguyên artifact đã hoàn tất. |
| `src/training/train_cil.py` | Thêm `--resume-fold-checkpoint`, chỉ cho phép với policy frozen-fold. |
| `tests/test_fold_replay_checkpoint.py` | Synthetic A/B ba task, so continuous/resume, gián đoạn giữa epoch, mismatch, giữ file cũ, scope completion và atomic write. |
| `docs/TDDI_STRATIFIED_3FOLD_TRAINER.md` | Cập nhật giới hạn/trạng thái trainer sau Prompt 10. |

## Checkpoint chứa gì?

File: `RUN_ROOT/checkpoints/task_N.pt`.

- Kind `ddi_cil_frozen_fold_replay_task_boundary`, schema version 1.
- Run ID, completed/next task, `full_trajectory_complete` (chỉ true ở task 7).
- Best `model_state`, dense class map và thứ tự raw class IDs của các cột head.
- Contract: method/backbone/hyper, seeds/member/fold, assignment/manifest/source/
  task-file hashes, feature order, preprocessing artifact hash **và state frozen**,
  allocation/ranking/budget/sampler versions cùng implementation hashes.
- Buffer: raw float64 descriptors, raw labels, sample/source IDs, rank priorities/
  distances, observed counts và lịch sử feasible capacities/quota; không lưu dữ
  liệu task cũ đã loại. Số slot tính từ 4% **toàn development**, không tính từ pilot prefix.
- RNG Python/NumPy/Torch CPU/CUDA tại task boundary.
- Sampler state task vừa hoàn tất (queues/cursors/epoch), quy tắc reset task kế
  tiếp ở epoch 0, member/task keyed RNG và seed DataLoader task mới.
- Toàn bộ metric rows, epoch audit rows, task summaries mà trainer hiện có.
  Trainer này chưa có một classwise tracker riêng để serialize; không bịa tracker state.
- SHA256 của các artifact task hoàn tất và `run_config.json`; checksum nội dung
  toàn checkpoint, gồm cả tensor và NumPy/RNG state.

Không lưu teacher riêng; sau resume dựng model đúng head cũ, load best weights và
freeze/eval để làm teacher. Model mới được mở rộng/copy theo raw ID như continuous run.
Không lưu optimizer: AdamW mới ở mỗi task đúng thiết kế. Restore RNG **sau** khi dựng
model cũ, để bước khởi tạo phục hồi không làm lệch RNG của student task tiếp theo.

## Lệnh resume trên server (khi đã được phép chạy pilot)

Giữ nguyên mọi tham số của lệnh A hoặc B trước đó, cùng member, preprocessing artifact,
method/hyper và `--outdir`. Chỉ thêm cờ resume và đặt phạm vi thực thi cần thiết:

```bash
# Phần bổ sung vào lệnh train_cil.py của run hiện tại:
--resume-fold-checkpoint "$RUN_ROOT/checkpoints/task_0.pt" \
--stop-after-task 1
```

Đây là **đoạn tham số**, không phải một lệnh bash độc lập. Runbook A/B đầy đủ thuộc
Prompt 11–12. Không dùng `--resume-replay-checkpoint` hoặc `--resume-ewc-checkpoint`
cho checkpoint mới; `task_N/best_model.pt` chỉ chứa inference weights, không đủ resume.

Muốn mở rộng pilot task 0–1 sang task 2: dùng checkpoint task 1 và stop-after-task 2,
nhưng vẫn dùng P3 tám task gốc và hash cũ. Không tạo task-file rút gọn, không đổi
epochs/lr/budget giữa chừng. Chỉ chạy thêm task khi người dùng cho phép.

## Không ghi đè và xử lý gián đoạn

1. Run mới vẫn yêu cầu output chưa tồn tại; không tự xóa một run dở.
2. Mỗi task có checkpoint riêng. Ghi tạm cùng thư mục → flush/fsync → publish
   bằng hard link không replace. Nếu đích đã tồn tại, fail; chỉ dọn file tạm do
   lần ghi này tạo. Filesystem không hỗ trợ hard link sẽ báo lỗi, không fallback ghi đè.
3. Completion chỉ hợp lệ khi checkpoint và **tất cả artifact cần thiết khớp hash**.
   `completed_task.json` hoặc thư mục tồn tại đơn lẻ không đủ để skip.
4. Task đã hoàn tất không được train/evaluate/export lại. Nếu chọn checkpoint cũ
   trong khi có boundary mới hơn, fail và yêu cầu dùng checkpoint mới nhất.
5. Task kế tiếp có thư mục dở được giữ nguyên. Lần thử mới dùng
   `attempts/task_N_<token>/`; đường dẫn thật lưu trong `artifact_directory` của
   task summary/checkpoint. Không mặc định kết quả mới luôn ở `task_N/`.
6. Báo cáo root của lần chạy trước giữ nguyên. Sau resume, tổng hợp mới nằm tại
   `reports/through_task_N_<token>/`; log ghi đường dẫn. Không dùng summary cũ
   ở root để quyết định full trajectory đã xong.
7. Resume vào phạm vi đã hoàn tất: validate checkpoint/artifacts rồi skip, không
   tạo model hoặc ghi file. Xong pilot task 1 **không** đồng nghĩa xong full task 7.

## Validation và giới hạn

Fail nếu khác A/B hoặc scaler state/hash, member/fold/seed, source/assignment/task
hash, feature order, class map, backbone/hyper, budget/quota/ranking/sampler policy;
checkpoint thiếu/sai schema/RNG/model/buffer/metric progress cũng fail rõ.

Mỗi run/resume validate nguồn từ đầu, không reuse context cũ để bỏ qua hash. Scaler B
được **load**, không fit lại; folds được **load**, không dựng lại. Có thể chuyển toàn
run directory hoặc đổi path nguồn khi giữ nguyên byte hashes và cùng contract.
Preprocessing artifact cũng phải còn có thể load và khớp hash; state trong checkpoint
không phải lý do tự bỏ qua file nguồn hay kiểm tra provenance.

Checkpoint là file Torch có Python/NumPy RNG state: chỉ load checkpoint tin cậy của
mình (`weights_only=False`), không load file lạ. Checksum phục vụ phát hiện hỏng/sai
state, không phải chữ ký xác thực chống người cố tình làm giả.

Reproducibility chính xác được test trên cùng môi trường CPU synthetic. Không cam
kết bitwise giữa hardware/software khác nhau; CUDA RNG phải tương thích số thiết bị
visible. Không có test VRAM hoặc full dataset trong bước này.

## Kiểm thử

```bash
python -m pytest tests/test_fold_replay_checkpoint.py tests/test_fold_pilot_training.py tests/test_replay_checkpoint.py tests/test_fold_replay_buffer.py tests/test_fold_replay_sampler.py -q
```

Các đối chứng so model/head, teacher, raw buffer/IDs/ranking/counts, class map,
next task, RNG, sampler order và metrics. Runtime/VRAM không được yêu cầu bằng
nhau giữa các lần chạy; chúng là telemetry, không phải kết quả số của model.

Kết quả kiểm chứng ngày 2026-09-14:

- Nhóm checkpoint/trainer/buffer/sampler/legacy: **156 passed**, 94,71 giây.
- `python -m pytest tests -q`: **425 passed**, 290,38 giây; không có failure.
- `train_cil.py --help` hiển thị `--resume-fold-checkpoint`; compileall và
  `git diff --check` thành công.
- Không chạy dataset thật, không commit/push trong Prompt 10.

Log cuối task ghi `checkpoint=... checkpoint_bytes=...` cho boundary checkpoint.
Trường `checkpoint_size_bytes` trong task summary vẫn là kích thước `best_model.pt`
để giữ rõ ý nghĩa telemetry đã có; hai file có dung lượng khác nhau vì boundary
còn chứa buffer và reporting/RNG state.
