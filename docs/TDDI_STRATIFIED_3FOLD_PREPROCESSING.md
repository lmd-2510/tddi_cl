# Preprocessing A/B từ frozen folds — Prompt 6

Trạng thái: đã có API/CLI và tests synthetic. **Chưa nối trainer, chưa chạy dataset
thật, chưa chọn A hoặc B làm phương án cuối.** Không thay scaler/config legacy.
Theo [decision record](TDDI_STRATIFIED_3FOLD_DECISION_RECORD.md), bước này chỉ chuẩn
bị hai phương án để pilot sau; không chạy training hoặc tự tạo full study.

## Hai policy

| Policy/version | Làm gì | Dữ liệu được dùng fit |
| --- | --- | --- |
| `raw_identity` / 1 | Giữ descriptor gốc; LayerNorm vẫn nằm trong model | Không fit; `fit=null`, `statistics=null` |
| `task0_standard_frozen` / 1 | `(x - mean) / scale`, đóng băng xuyên task | Chỉ class task 0 **và** hai training folds của member |

Member 0/1/2 giữ validation fold 0/1/2. Task 0 lấy từ task-file P3
`protocol=tail_to_head`, không mặc định raw class IDs liên tục. Không nhận tham số
fit task 1–7, không fit toàn bộ training folds. API nhận task-file nhỏ để test
synthetic; kiểm tra đủ tám task/178 class sẽ nằm ở bước study config sau.

Chuẩn bị A cũng scan các raw rows task-0-train để kiểm tra input/feature schema,
nhưng không tính/lưu mean/variance. Cả A/B ghi cùng reference selection để đối chiếu.
Parquet đọc theo batch, lọc trước khi thống kê; không giữ toàn bộ feature matrix.
Test chỉ được loader đọc ID và hash để kiểm tra integrity, không dùng fit.

## Quy ước số học và dữ liệu lỗi

- Tính toán và output **float64**. Trainer sau này chịu trách nhiệm chuyển sang
  dtype model, không thay dtype scaler đã freeze.
- Variance tổng thể: `ddof=0`; gộp thống kê batch theo Chan, không cộng bình phương
  rồi trừ hai số lớn. Không random sampling, không fit lại khi transform.
- `scale=sqrt(variance)`; variance đúng bằng 0 thì `scale=1`.
- Không thêm epsilon/floor vào variance gần 0. Không cam kết bitwise giống mọi
  phiên bản sklearn ở cột gần hằng; batch size khác có thể sai khác làm tròn nhỏ.
- **Cả A/B đều fail khi input có null/NaN/Inf** tại scan task0 hoặc transform.
  Loader raw chuyển null thành NaN; preprocessing không impute, clip hay bỏ dòng.
  Nếu thống kê/output overflow float64 cũng fail. Khi gặp lỗi trên server, cần
  xem cột/dòng và quyết định policy riêng, không tự lấy thống kê validation/future.
- Không scan/đánh giá toàn bộ future descriptors để chọn preprocessing. Values
  held-out/future không tham gia statistics; khi transform chúng về sau, vẫn áp
  quy tắc kiểm tra nonfinite như trên.

## File và API

- `src/data/ddi_dataset.py`: thêm `iter_development_fold_arrays` để đọc raw batches
  theo đúng frozen assignment. `load_development_fold_arrays` vẫn trả cùng arrays,
  IDs/provenance/thứ tự như Prompt 5. Loader/scaler legacy không đổi.
- `src/data/fold_preprocessing.py`:
  - `prepare_fold_preprocessing`: tạo raw artifact hoặc fit scaler task0-train.
  - `save_fold_preprocessing`: tạo **outdir mới**, ghi staging + fsync rồi publish
    JSON bằng hard link atomic, không clobber file cũ. Filesystem phải hỗ trợ hard
    link (NTFS/ext4); không âm thầm fallback sang ghi đè. Lỗi có thể để lại thư mục
    rỗng; retry chọn namespace mới sau khi kiểm tra.
  - `load_fold_preprocessing`: kiểm tra schema, payload checksum, policy, member,
    seeds, assignment/source/task/feature order và đúng reference rows; **không fit**.
  - `FoldPreprocessing.transform`: áp cùng frozen transform cho train, replay,
    validation và test. Không thay đổi input hoặc artifact, không cập nhật statistics.
- `scripts/prepare_fold_preprocessing.py`: CLI chuẩn bị một member/một policy;
  không import trainer/model, không train, không tạo sáu bản feature Parquet.
- `tests/test_fold_preprocessing.py`: thống kê tính tay, exact IDs, A/B alignment,
  future/held-out independence, integrity, round-trip/atomic/CLI và guard nonfinite.

## Artifact và resume

Một file `fold_preprocessing.json` gồm:

- Schema/policy version; timestamp UTC; numeric conventions; scan batch size.
- Experiment/member seed và derivation hiện có; validation fold; fold seed.
- Hash assignment, manifest, cả ba source và task file; source row counts.
- Ordered feature names và hash JSON canonical của **thứ tự feature**, không phải
  hash bytes của file feature JSON (thay whitespace không đổi feature order).
- Reference task0-train rows gồm split/index/ID/raw label/fold và hashes; không
  chỉ ghi số mẫu. Có thể tái dựng/đối chiếu từ frozen assignment + source.
- Với B: fit task/classes/count/ID hashes, mean/variance/scale. Với A: không giả
  vờ đã fit thống kê; reference rows chỉ là provenance kiểm tra input.
- SHA256 nội dung payload canonical để phát hiện sửa/corrupt artifact.

Đổi đường dẫn source được nếu bytes giữ nguyên; path vật lý không dùng làm khóa
compatibility. Manifest/task file đã pin vẫn phải giữ hash tương ứng.

Đầu **mỗi run/resume** phải tạo lại validated context theo Prompt 5, sau đó load
artifact từ disk. Không khôi phục context cache cũ, không gọi prepare để refit.
Checkpoint ở bước sau cần pin thêm hash bytes artifact bằng `expected_sha256`.

```python
# context: vừa tạo qua prepare_development_fold_context ở đầu run/resume.
from src.data.fold_preprocessing import load_fold_preprocessing

frozen = load_fold_preprocessing(
    "study_assets/fold_preprocessing_member0/B/fold_preprocessing.json",
    context=context,
    task_file="study_assets/task_protocols/tail_to_head_tasks.json",
    feature_columns=feature_columns,
    member_id=0, validation_fold=0, experiment_seed=0,
    policy="task0_standard_frozen",
    # expected_sha256=hash_bytes_saved_in_checkpoint,
)
x = frozen.transform(
    raw_features, feature_columns=feature_columns, member_id=0,
    policy="task0_standard_frozen",
)
```

Đầu vào transform phải là raw descriptors theo đúng feature order, không phải
features đã scale. Tích hợp đường dữ liệu train/replay/validation/test thuộc prompt
trainer sau; API này không tự chạy các đường đó.

## CLI tham khảo trên server — chưa cần chạy ngay

Xác minh path fold thực tế trước khi dùng; ví dụ dưới dùng namespace trong tài liệu
artifact, không khẳng định đây là vị trí dataset/assignment hiện tại của server.
Chạy từ repo, môi trường đã cài requirements. Không cần GPU.

```bash
python scripts/prepare_fold_preprocessing.py --help

python scripts/prepare_fold_preprocessing.py \
  --assignments study_assets/stratified_3fold_seed42/fold_assignments.parquet \
  --manifest study_assets/stratified_3fold_seed42/fold_manifest.json \
  --train train_extracted.parquet \
  --validation validation_extracted.parquet \
  --test test_extracted.parquet \
  --task-file study_assets/task_protocols/tail_to_head_tasks.json \
  --feature-cols study_assets/data_schema/feature_columns.json \
  --member-id 0 --validation-fold 0 --experiment-seed 0 --fold-seed 42 \
  --policy task0_standard_frozen --batch-size 2048 \
  --outdir study_assets/fold_preprocessing_member0/B
```

Để chuẩn bị A, thay `--policy raw_identity` và
`--outdir study_assets/fold_preprocessing_member0/A`; giữ các đối số khác.
Lệnh chỉ tạo preprocessing artifact; không chạy smoke/pilot/full training.

Tests synthetic trên máy code:

```bash
python -m pytest tests/test_fold_preprocessing.py tests/test_development_fold_loader.py -q
```

Tiếp theo: Prompt 7 (buffer/ranking), chưa phải bước chạy hai pilot thật.
