# Fold artifact — kết quả Prompt 1–3

Implementation nằm trong `src/data/stratified_folds.py`. Prompt 1 bổ sung định dạng
và API lưu/đọc; Prompt 2 thêm CLI build `scripts/build_stratified_3fold_assignments.py`.
Prompt 3 thêm CLI audit độc lập `scripts/audit_development_folds.py`.
Chưa nối artifact này vào training.
Script build được đổi tên thành `build_stratified_3fold_assignments.py` để thể hiện
rõ mục đích: gán mỗi mẫu development vào một trong ba stratified folds. Các lệnh
mới bên dưới dùng tên mới; tên file đầu ra, schema và thuật toán không thay đổi.
Không cần build lại folds đã audit chỉ vì đổi tên script. `creation_command` trong
manifest lịch sử vẫn giữ nguyên để bảo toàn provenance; không sửa hash/artifact cũ.
Thuật toán runtime `build_stratified_fold_assignments` và `select_development_fold`
được giữ nguyên. Không có module `development_folds.py` thứ hai.

## Hai file được lưu

```text
<outdir mới>/
├── fold_assignments.parquet
└── fold_manifest.json
```

Schema version: `1`; artifact kind: `ddi_cil_development_folds`.

| Cột Parquet | Dtype Arrow | Ý nghĩa |
| --- | --- | --- |
| `source_split` | string | `train` hoặc `validation`, không có test |
| `source_row_index` | int64 | Index 0-based của dòng trong file nguồn |
| `sample_id` | string | ID ordered pair do `sample_identity.py` tạo |
| `raw_class_id` | int64 | Nhãn gốc, không phải index của expandable head |
| `fold_id` | int16 | 0, 1 hoặc 2 |

Không chấp nhận null, thiếu/thừa cột hoặc tự ép dtype. Thứ tự dòng và dtype được
giữ qua round-trip; `to_lookup()` tạo một lookup sort theo ID riêng biệt. Metadata
`input_order` mô tả thứ tự đầu vào thuật toán chia fold, không bắt buộc thứ tự lưu
Parquet; vị trí nguồn luôn được ghi trong `source_row_index`.

Manifest lưu fold seed, n_splits=3, strategy/shuffle/identity version, mapping
member 0→fold 0, 1→1, 2→2; path/SHA256/row count của train/validation/test;
assignment filename/SHA256/row count; timestamp UTC và creation command.
Writer bổ sung `fold_summary`: số dòng và số mẫu theo raw class ID của mỗi fold.
Seed được truyền explicit, API mới không tự quyết định dùng 0 hay 42.

## API cho Prompt 2–3

- `describe_fold_source(path)`: lấy file hash và Parquet row count; đọc hash theo
  chunk, không load feature matrix. Hash file lớn trên server vẫn cần đọc toàn bộ bytes.
- `save_fold_artifact(outdir, table, sources=..., fold_seed=..., creation_command=...)`:
  nhận bảng theo `FOLD_ASSIGNMENT_SCHEMA` và metadata của ba nguồn. Kiểm tra bảng
  rồi lưu vào thư mục mới. Không tự chia fold hoặc fit scaler.
- `load_fold_artifact(assignment_path, manifest_path, expected_metadata=..., source_paths=...)`:
  xác minh SHA256 assignment, schema, coverage dòng nguồn và metadata được yêu cầu.
- `validate_fold_artifact(table, manifest, expected_metadata=...)`: kiểm tra object
  trong RAM; không tự đọc/hash file.
- `FoldArtifact.to_lookup()`: chuyển về interface `StratifiedFoldAssignments` cũ.

`expected_metadata` đối chiếu các key top-level được cung cấp. Ví dụ
`{"fold_seed": 42, "member_to_validation_fold": {"0": 0, "1": 1, "2": 2}}`.
Nếu truyền object lồng nhau như `sources`, giá trị object được so sánh đầy đủ.

`source_paths` là mapping đủ `train`, `validation`, `test`. Khi cung cấp, loader
kiểm tra file bytes SHA256 và row count. File có thể chuyển sang đường dẫn server
khác miễn nội dung không đổi. Khi bỏ qua `source_paths`, loader chỉ xác minh artifact;
không được diễn giải kết quả load là đã audit dataset thật.

## Phạm vi validation và atomic write

Validation kiểm tra mỗi source row xuất hiện đúng một lần, ID duy nhất, fold hợp lệ,
đủ ba fold, số dòng khớp manifest và các field bắt buộc. Chưa xác minh sample ID/raw
label thực sự tương ứng với dòng Parquet nguồn, mức stratification, hoặc test overlap.
Đây là giới hạn của validator/loader dùng chung. CLI build ở Prompt 2 kiểm tra
trùng ID giữa development/test và dùng builder stratified hiện tại. Việc kiểm tra
độc lập nội dung artifact sau khi lưu vẫn thuộc script audit ở Prompt 3.

Save từ chối mọi output directory đã tồn tại, kể cả directory rỗng. Hai file dùng
staging cùng directory, flush/fsync và atomic replace riêng từng file; manifest được
publish cuối cùng. Đây không phải transaction atomic cho cả directory. Nếu gián đoạn,
có thể còn assignment nhưng thiếu manifest; loader từ chối và lần save tiếp theo
không ghi đè directory đó. Chỉ file staging do lần save đó tạo được dọn dẹp.

## CLI build — Prompt 2

Chạy từ repo trên **server có dataset**, không phải máy code hiện tại. Ví dụ dưới
đây dùng fold seed 42; đây là seed chia fold, không phải quyết định thay experiment
seed hoặc cấu hình buffer. Phải truyền seed rõ ràng, không có default ngầm.

```bash
python scripts/build_stratified_3fold_assignments.py \
  --train train_extracted.parquet \
  --validation validation_extracted.parquet \
  --test test_extracted.parquet \
  --fold-seed 42 \
  --n-splits 3 \
  --outdir study_assets/stratified_3fold_seed42
```

Chỉ dùng một **outdir mới, chưa tồn tại**, kể cả chưa tạo thư mục rỗng trước.
Nếu cần build lại, chọn namespace mới thay vì ghi đè artifact cũ.

Script thực hiện:

1. Hash ba nguồn và đọc ID; chỉ train/validation cần đọc thêm nhãn.
2. Ghép logic theo thứ tự train rồi validation, giữ index dòng trong từng nguồn.
3. Từ chối duplicate ID, overlap development/test theo ordered pair, nhãn không
   nguyên/null hoặc class có ít hơn ba mẫu development. Không tự bỏ mẫu hay đổi split.
4. Gọi builder `StratifiedKFold` dùng chung, với shuffle và seed đã chỉ định.
5. Hash lại nguồn để phát hiện file bị thay đổi trong lúc build, rồi lưu hai artifact.

Chỉ đọc các cột ID/nhãn vào bảng xử lý; không load/copy 3.780 cột feature, không
tạo sáu bản dataset cho ba member. Việc SHA256 vẫn đọc toàn bộ bytes của ba file
hai lần nên cần thời gian I/O; không cần GPU. Test giữ nguyên, không tham gia chia
fold. Guard ordered pair không đồng nghĩa đã kiểm tra leakage theo unordered pair
hoặc drug-disjoint; không tự áp dụng chính sách split khác trong bước này.

Sau khi audit đạt yêu cầu, mapping dự kiến là member 0 train folds 1+2/val fold 0,
member 1 train 0+2/val 1, member 2 train 0+1/val 2. CLI này **chỉ tạo artifact**;
chưa làm training hiện tại chuyển sang đọc artifact, chưa fit scaler hay chọn buffer.

## Tests và bước tiếp theo

`tests/test_fold_artifact.py` kiểm tra round-trip, compatibility lookup cũ, schema,
duplicate/source-row coverage, invalid fold, null/dtype, hash/metadata mismatch,
nguồn đổi nội dung, đường dẫn nguồn thay đổi, không overwrite và save gián đoạn.

`tests/test_build_stratified_3fold_assignments.py` kiểm tra assignment deterministic, seed khác,
coverage/disjointness và cân bằng từng class; tương thích builder cũ; không đọc feature
hoặc test label; input lỗi, nguồn bị đổi, không overwrite và CLI chạy ngoài repo.

```bash
python -m pytest tests/test_build_stratified_3fold_assignments.py tests/test_fold_artifact.py tests/test_stratified_ensemble_mode.py -q
```

Kết quả kiểm tra local ngày 2026-09-12: **63 passed**, 85,99 giây, chỉ dữ liệu giả.

`tests/test_audit_development_folds.py` kiểm tra báo cáo tốt/xấu, CLI exit code,
hash/count, row identity/label, duplicate/coverage, stratification, member mapping,
test leakage, cảnh báo unordered pair, output preservation và nguồn đổi trong lúc audit.

Kiểm tra gộp sau Prompt 3 (local, chỉ dữ liệu giả):

```bash
python -m pytest tests/test_audit_development_folds.py tests/test_build_stratified_3fold_assignments.py tests/test_fold_artifact.py tests/test_stratified_ensemble_mode.py -q
```

Kết quả ngày 2026-09-12: **84 passed**, 81,39 giây (21 tests audit mới).
CLI `python scripts/audit_development_folds.py --help` cũng đã chạy thành công.

## Audit độc lập — Prompt 3

`scripts/audit_development_folds.py` chỉ đọc folds và ba Parquet nguồn, không gọi
builder để chia lại, không sửa nguồn/folds, không sửa loader/model/training.
Script đối chiếu hash/count, schema và mỗi source row đúng một lần, ID/nhãn khớp
dòng nguồn, ID duy nhất, ba fold không giao nhau, mapping member, mỗi class có mặt
trong cả ba folds và lệch tối đa một mẫu giữa folds, tổng kích thước folds lệch tối
đa một mẫu, không trùng ordered pair với test. Summary trong manifest (nếu có) được
đối chiếu lại, không tin số liệu summary có sẵn.

Audit không chứng minh fold seed đã tạo đúng assignment bằng cách chạy lại thuật
toán; nó kiểm tra partition đã lưu và metadata. Test không được đọc nhãn/feature.
Hash các nguồn và artifact được kiểm tra lại cuối lượt để phát hiện thay đổi khi audit.

Output nằm trong một thư mục audit **mới**:

| File | Ý nghĩa |
| --- | --- |
| `fold_audit.json` | PASS/FAIL từng check, hash/path thực tế, fold seed, mapping, cảnh báo và số liệu |
| `fold_class_counts.csv` | Mỗi raw class: tổng development, số mẫu fold 0/1/2, độ lệch và số train của mỗi member |
| `fold_member_views.csv` | Member 0/1/2: train folds, validation fold, số mẫu train/val/test, fraction và số class |
| `fold_audit.md` | Bản dễ đọc: kết luận, checks, kích thước member và cảnh báo |

Exit code **0** = invariant bắt buộc đạt; **1** = audit fail; **2** = lỗi CLI/output.
Input không hợp lệ vẫn xuất báo cáo FAILED khi output ghi được; nếu không thể kiểm
tra tiếp thì CSV có thể chỉ còn header. Không coi check chưa chạy là pass. Directory
output tồn tại bị từ chối; ghi atomic từng file, JSON publish cuối. Chỉ dùng báo cáo
khi JSON hoàn chỉnh tồn tại. Không tự dọn hoặc ghi đè lần chạy gián đoạn.

`PASSED` chỉ bảo đảm chính sách ordered-pair/row-level đã khai báo, **không phải**
cam kết drug-disjoint hay lệnh cho phép train ngay. Cặp A–B/B–A giữa development/test
hoặc giữa folds được báo warning và đếm theo unordered pair để quyết định ở Prompt 4.
`unordered_crossfold_pair_intersections` là tổng giao của ba cặp folds; cùng một cặp
có thể đóng góp ở nhiều giao, không phải số cặp duy nhất toàn cục.

### Lệnh trên server sau khi build xong

Từ thư mục repo, activate môi trường đã cài requirements. Thay đường dẫn input nếu
dataset nằm nơi khác; hash được đối chiếu theo nội dung, không bắt buộc cùng path
với máy đã tạo manifest. Ví dụ seed 42 phải khớp lệnh build phía trên:

```bash
python scripts/audit_development_folds.py \
  --assignments study_assets/stratified_3fold_seed42/fold_assignments.parquet \
  --manifest study_assets/stratified_3fold_seed42/fold_manifest.json \
  --train train_extracted.parquet \
  --validation validation_extracted.parquet \
  --test test_extracted.parquet \
  --outdir outputs/fold_audit_seed42
```

Hoặc dùng nohup (chọn **một** trong hai cách chạy, không chạy trùng):

```bash
mkdir -p outputs/fold_audit_logs
AUDIT_LOG="outputs/fold_audit_logs/audit_$(date +%Y%m%d_%H%M%S).log"
nohup env PYTHONUNBUFFERED=1 python scripts/audit_development_folds.py \
  --assignments study_assets/stratified_3fold_seed42/fold_assignments.parquet \
  --manifest study_assets/stratified_3fold_seed42/fold_manifest.json \
  --train train_extracted.parquet \
  --validation validation_extracted.parquet \
  --test test_extracted.parquet \
  --outdir outputs/fold_audit_seed42 \
  > "$AUDIT_LOG" 2>&1 < /dev/null &
echo $! | tee "${AUDIT_LOG}.pid"
tail -f "$AUDIT_LOG"
```

Không cần GPU. `Ctrl+C` khi đang `tail -f` chỉ thoát xem log, không dừng audit nohup.
Đợi log ghi PASSED/FAILED rồi đọc `outputs/fold_audit_seed42/fold_audit.md`.
Nếu chạy lại, chọn outdir audit mới; không xóa artifact cũ để vượt guard.

**Dừng ở đây trước training.** Gửi `fold_manifest.json` và bốn file audit để làm
Prompt 4: chốt split policy, preprocessing, buffer/replay và config dựa trên số liệu
thật. Chưa quyết định các giá trị đó trong Prompt 3; chưa chạy dataset thật ở máy code.
