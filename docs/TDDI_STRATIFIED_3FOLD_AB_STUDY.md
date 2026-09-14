# Prompt 11 — Config và điều phối smoke/pilot preprocessing A/B

Ngày: 2026-09-14. Theo [decision record](TDDI_STRATIFIED_3FOLD_DECISION_RECORD.md).

Đây là công cụ **chuẩn bị/chạy có chủ đích hai đối chứng**, chưa phải full study.
Mặc định chỉ member 0, task 0–1, validation-only. Không tự chọn phương án thắng,
không chạy test evaluation, ensemble, OOF hoặc threshold. Prompt 12 đã bổ sung
[report so sánh và runbook GPU](TDDI_PREPROCESSING_AB_PILOT_RUNBOOK.md). Không có training dữ liệu thật trong lượt
triển khai Prompt 11.

## 1. Những file liên quan

| File | Công dụng |
| --- | --- |
| `src/training/fold_ab_study.py` | Đọc/validate config mới, in dry-run, kiểm tra checkpoint, gọi trainer tuần tự, ghi manifest điều phối. |
| `src/training/fold_pilot_training.py` | Tách `prepare_fold_run()` dùng chung giữa trainer và bộ điều phối; validate nguồn/fold/preprocessing/contract mà chưa tạo model. Training và resume vẫn dùng pipeline Prompt 9–10. |
| `src/training/train_cil.py` | `parse_args(argv=None)` cho phép kiểm tra command bằng danh sách tham số; CLI cũ vẫn đọc `sys.argv` như trước. |
| `configs/smoke_tddi_p3_fold_ab_A_raw_seed0.json` | Smoke A, raw descriptors, tối đa 3 epoch/task. |
| `configs/smoke_tddi_p3_fold_ab_B_task0_scaler_seed0.json` | Smoke B, scaler task 0 đóng băng, tối đa 3 epoch/task. |
| `configs/pilot_tddi_p3_fold_ab_A_raw_seed0.json` | Pilot A, tối đa 20 epoch/task. |
| `configs/pilot_tddi_p3_fold_ab_B_task0_scaler_seed0.json` | Pilot B, tối đa 20 epoch/task. |
| `tests/test_fold_ab_study.py` | Fake runner kiểm tra thứ tự/skip/resume/dry-run; synthetic nhỏ kiểm chứng checkpoint thật và alignment A/B. |
| `docs/TDDI_STRATIFIED_3FOLD_AB_STUDY.md` | Tài liệu này: config, CLI, path override, namespace và giới hạn của pilot. |
| `docs/TDDI_STRATIFIED_3FOLD_IMPLEMENTATION_PROMPTS.md` | Cập nhật tiến độ: Prompt 11 đã triển khai, bước tiếp theo là Prompt 12. |

Không sửa config cũ hoặc logic study EWC/replay legacy. Entry mới chỉ mượn helper
chạy subprocess đồng bộ `_default_runner` từ `tddi_ensemble3_study.py`, không dùng
quy tắc legacy budget 6800, legacy checkpoint hoặc tự động ensemble của entry cũ.

## 2. Những giá trị được khóa để so sánh công bằng

- Method CLI: `replay_distill_fixed_budget_uniform`; **policy mới opt-in**:
  `stratified_fraction_v1`. Không coi kết quả là recipe uniform 6800/6800 cũ.
- Paper-size member: 3780 → 7560 → 7560 → expandable head, input LayerNorm,
  GELU/dropout 0.2. Không tuyên bố toàn bộ training recipe giống chính xác paper.
- P3 full protocol: `study_assets/task_protocols/tail_to_head_tasks.json`, đủ 178
  class, layout `[38,20,20,20,20,20,20,20]`; chỉ **dừng thực thi sau task 1**.
- SHA256 task-file:
  `0d64c465b0c4bd34f66e6c76088b6b73fd60839ade3e56017b5fd36c21a26e79`.
  Đổi path được, đổi byte/nội dung không được. Nếu hash khác, kiểm tra revision,
  file protocol và line endings; không tự sửa hash chỉ để vượt validation.
- Experiment seed 0; fold seed 42; mặc định `[0]`. Member ID cũng là held-out fold.
  Member seeds dùng derivation hiện có, giống nhau giữa A/B của cùng member.
- Budget tính từ **toàn bộ development 694455**, không từ task 0–1 và không gán
  cả 27778 slots cho member 0: tổng 27778, member 0/1/2 là 9260/9259/9259.
  Đây là sức chứa tối đa, không bắt buộc lấp đầy buffer ở task sớm.
- Quota 10 + sqrt tần suất training, capacity-aware; ranking chung tạm thời
  `raw_sample_normalized_class_mean_control_v1` giữ exemplar IDs chung giữa A/B.
- Current đầy đủ mỗi epoch; replay mục tiêu 12.5%, class-uniform có giới hạn,
  repeat cap 3; giảm replay khi thiếu capacity. Task 0 không replay.
- Microbatch 64, effective batch mục tiêu 1024, accumulation 16. Batch tích lũy
  cuối có thể nhỏ hơn; không bỏ current samples để ép đủ 1024.
  Prompt 12 cho phép config OOM riêng 32/16/8 với accumulation 32/64/128;
  effective vẫn 1024. Áp cùng điều kiện cho A/B, namespace mới, không auto fallback.
- AdamW mới mỗi task, lr 0.001, weight decay 0.0001, patience 5, focal gamma 1,
  distillation alpha 1/temperature 2, feature MSE weight 0.5. Không thêm scheduler.

A/B cùng phase chỉ được khác case, preprocessing artifact/policy và output.
Smoke dùng 3 epoch/task, pilot 20; không ghép smoke A với pilot B. Early stopping
có thể làm số epoch thực chạy khác nhau, nên so sánh sampling ở các epoch chung.

## 3. Input cần có trên server

Config ghi riêng `train`, `validation`, `test`, `feature_cols`, `fold_assignments`,
`fold_manifest`, task-file và preprocessing artifact template. Không dựng lại
StratifiedKFold khi train; không copy thành sáu feature Parquet.

Preprocessing phải chuẩn bị trước theo [Prompt 6](TDDI_STRATIFIED_3FOLD_PREPROCESSING.md).
A cũng cần artifact `raw_identity` ghi provenance, không chỉ B có scaler.
Trainer/bộ điều phối **không tự fit** scaler khi chạy hoặc resume.

Default preprocessing layout tương thích tài liệu Prompt 6:

```text
study_assets/fold_preprocessing_member0/A/fold_preprocessing.json
study_assets/fold_preprocessing_member0/B/fold_preprocessing.json
```

Giữ cả sidecar của preprocessing trong cùng thư mục, không chỉ file JSON. Test
nguồn vẫn cần cho integrity/hash và kiểm tra ID overlap; validation-only không
dùng test để tính metric hoặc chọn preprocessing.

## 4. CLI thực tế — mặc định chỉ dry-run

Chạy từ repository root. Các lệnh dưới **không training**, chưa cần `nohup`:

```bash
python src/training/fold_ab_study.py --help

python src/training/fold_ab_study.py \
  --config configs/smoke_tddi_p3_fold_ab_A_raw_seed0.json \
           configs/smoke_tddi_p3_fold_ab_B_task0_scaler_seed0.json

python src/training/fold_ab_study.py \
  --config configs/pilot_tddi_p3_fold_ab_A_raw_seed0.json \
           configs/pilot_tddi_p3_fold_ab_B_task0_scaler_seed0.json \
  --member-id 0
```

Thứ tự luôn A rồi B; member tăng dần trong mỗi case. Có thể truyền một config
để chỉ chạy A hoặc B. Không có `--member-id` thì dùng danh sách config `[0]`.
`--member-id 1` hoặc `--member-id 2` chỉ chọn member đó; có thể lặp flag để chọn
nhiều member khi người dùng chủ động mở rộng pilot, không cần đủ ba member.

Nếu máy code không có dữ liệu, dry-run in `unverified_missing_inputs`/`UNVERIFIED`
và liệt kê path thiếu: chỉ xác nhận config/command, **không phải đủ điều kiện train**.
Nếu đủ file, dry-run hash/validate inputs và checkpoint nên có thể mất thời gian
đọc dữ liệu; không tạo model hoặc output. Run/resume thật luôn validate lại,
không coi kết quả dry-run cũ là chứng nhận integrity vĩnh viễn.

### Override đường dẫn mà không sửa config gốc

Ví dụ dưới là path minh họa; thay bằng path thật trên server trước khi dùng:

```bash
python src/training/fold_ab_study.py \
  --config configs/smoke_tddi_p3_fold_ab_A_raw_seed0.json \
           configs/smoke_tddi_p3_fold_ab_B_task0_scaler_seed0.json \
  --train /path/to/train_extracted.parquet \
  --validation /path/to/validation_extracted.parquet \
  --test /path/to/test_extracted.parquet \
  --feature-cols /path/to/feature_columns.json \
  --fold-assignments /path/to/folds/fold_assignments.parquet \
  --fold-manifest /path/to/folds/fold_manifest.json \
  --task-file /path/to/tail_to_head_tasks.json \
  --preprocessing-root /path/to/preprocessing \
  --output-root /path/to/new_fold_ab_outputs \
  --device cuda
```

Quy ước riêng của `--preprocessing-root` là
`<root>/member_0/A/fold_preprocessing.json` và `<root>/member_0/B/fold_preprocessing.json`.
Nếu đang dùng default `fold_preprocessing_member0/A|B`, **bỏ flag này**; hoặc cấu
hình `preprocessing.artifact_template` theo nơi đã lưu, giữ `{member_id}` trong path.
Không bắt buộc di chuyển hay ghi đè preprocessing có sẵn.

`--output-root` tự nối `<root>/<smoke|pilot>/<A|B>/member_<ID>` để tách namespace.
`--python` chọn Python interpreter của subprocess; mặc định dùng đúng Python đang
chạy entrypoint. Nguồn có thể chuyển path nếu byte hashes vẫn khớp manifest/artifact.

## 5. Chạy có chủ đích, skip và resume

**Chỉ khi người dùng quyết định chạy trên server GPU**, thêm `--execute` vào đúng
lệnh đã dry-run. `CUDA_VISIBLE_DEVICES=0` có thể đặt trước lệnh để chọn GPU 0.
Không có flag này thì không subprocess training; không có flag tự chạy full8.
Hướng dẫn môi trường/GPU, `nohup`, thu thập số liệu và go/no-go thuộc Prompt 12.

- Một child training chạy đồng bộ và thoát trước khi child tiếp theo bắt đầu:
  không giữ model A/B hoặc hai member đồng thời trong bộ điều phối.
- Output chưa tồn tại và input hợp lệ: bắt đầu mới.
- Có checkpoint task 0 hợp lệ: tự thêm `--resume-fold-checkpoint .../checkpoints/task_0.pt`.
- Có checkpoint task 1 hợp lệ cùng toàn bộ artifact/hash/contract: skip. Hoàn tất
  **scope 0–1**, không có nghĩa hoàn tất full8; `full_trajectory_complete=false`.
- Output đã tồn tại nhưng không có boundary checkpoint, checkpoint hỏng hoặc
  thiếu artifact: fail rõ, không tự restart, xóa hoặc ghi đè. Giữ lại log để điều tra.
- Chạy lại cùng command sẽ inspect lại và resume/skip như trên. Task đang dở có
  thể phải học lại từ đầu task; không resume giữa epoch. Artifact task đã xong giữ nguyên.
- Không chạy hai lệnh `--execute` cùng namespace đồng thời. Kiểm tra PID trước
  khi chạy lại; checkpoint hợp lệ không chứng minh process cũ đã dừng.

Contract resume khóa cả hash implementation từ Prompt 10. Không trộn checkpoint
của revision/code khác rồi bỏ validation để tiếp tục. Namespace mới ở đây dùng
cho smoke/pilot mới; muốn resume run đang dở phải giữ đúng code, dữ liệu và hyper
đã tạo checkpoint. Checkpoint EWC/legacy không dùng với entry này.

## 6. Output và manifest

```text
outputs/fold_ab/
  smoke/
    A/member_0/...
    B/member_0/...
    manifests/<dispatch_id>/planned.json
    manifests/<dispatch_id>/completed_0.json
    manifests/<dispatch_id>/completed_1.json
    manifests/<dispatch_id>/complete.json
  pilot/
    A/member_0/...
    B/member_0/...
    manifests/<dispatch_id>/...
```

Snapshot JSON được ghi atomic, tên dispatch duy nhất, không thay file cũ. Khi lỗi
sau planning, có `failed_<index>.json`; lỗi preflight trước dispatch chưa ghi output.
Manifest chứa config gốc và config đã resolve path, policy/scope, budget, seeds,
run IDs, checkpoint/run_config paths và hashes, fold/source/preprocessing/task hashes,
summary task, hashes ID/class map/exemplar và sampling audit/order hash mỗi epoch.

`order_proofs` giúp kiểm tra A/B không khác ngầm current rows, validation rows,
exemplar đã giữ, class map hoặc sampler order ở epoch chung. Không so loss/model
weights vì đó là kết quả có quyền khác giữa hai preprocessing. Thiếu một case
hoặc chưa hoàn tất task thì chưa kết luận alignment của phần chưa chạy.

`checkpoints/task_1.pt` cùng các artifact được checkpoint xác nhận là nguồn trạng
thái hoàn tất. Sau resume, report tổng hợp mới có thể ở `reports/through_task_1_<id>/`
thay vì ghi đè report root cũ; manifest trỏ task artifact thực tế, kể cả attempt mới.

## 7. Kiểm chứng trên máy code

Kết quả ngày 2026-09-14, sau các chỉnh sửa cuối:

- `python -m pytest tests/test_fold_ab_study.py -q`: **27 passed**.
- `python -m pytest -q --ignore=tests/test_fold_ab_study.py`: **425 passed**.
- Tổng 452 test trong hai lượt trên; không có failure còn lại/failure cũ.
- Dry-run CLI thật cho cặp smoke A/B: exit 0; báo `unverified_missing_inputs`
  đúng vì máy code không có Parquet/assignment/preprocessing server, không train.
- `git diff --check`: pass. Không commit/push hoặc thay artifact lịch sử.

Có thể chạy riêng nhóm liên quan bằng:

```bash
python -m pytest tests/test_fold_ab_study.py \
  tests/test_fold_pilot_training.py tests/test_fold_replay_checkpoint.py \
  tests/test_tddi_ensemble3_replay_study.py -q
```

Các test dùng runner giả và dữ liệu synthetic nhỏ; model trong integration test
chỉ 2 → 8 → 4 trên CPU, không cấp phát model paper-size hoặc train dataset thật.
Không dùng kết quả synthetic để đánh giá A tốt hơn B. Prompt 12 đã có
[runbook GPU](TDDI_PREPROCESSING_AB_PILOT_RUNBOOK.md); dừng chờ kết quả thật trước Prompt 13.
