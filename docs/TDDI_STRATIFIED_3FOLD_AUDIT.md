# Audit Prompt 0 — T-DDI stratified 3-fold

Ngày audit: 2026-09-12. Commit được kiểm tra: `cc325990b7a62c7a2eceb8e0034865bb85839374`.
Working tree sạch khi bắt đầu. Phạm vi: đọc code/config/test, kiểm tra synthetic và tạo tài liệu này; chưa triển khai Prompt 1.

## 1. Kết luận

Repo đã có implementation chia train + validation thành ba fold và tích hợp vào training, prediction export, orchestration và OOF evaluation. Nên mở rộng `src/data/stratified_folds.py`, không tạo thêm module chia fold song song.

Implementation hiện tại chưa đáp ứng thiết kế artifact được lưu cố định rồi audit trước training. Các thiếu sót chính: chưa có assignment/source hashes; scaler vẫn thuộc train split cũ; kiểm tra fold khi resume/aggregate chưa đồng nhất; OOF đang được dùng chọn threshold cho ensemble test dù mỗi mẫu OOF chỉ có một member dự đoán.

Ba Parquet nguồn theo config không tồn tại trên máy audit. Các kết luận dưới đây dựa trên code, metadata preprocessing đã lưu và dữ liệu giả; chưa xác nhận chất lượng phân chia hoặc leakage thực tế trên máy GPU. Audit không xác minh recipe của paper gốc.

## 2. Luồng hiện tại và nơi chịu trách nhiệm

| File/function | Đang làm gì | Trạng thái so với kế hoạch |
| --- | --- | --- |
| `src/data/sample_identity.py::build_stable_sample_ids` | Tạo ID có thứ tự `drug_a\|drug_b`; kiểm tra kiểu, số dòng, ID rỗng và duplicate | Tái sử dụng làm quy tắc identity chung |
| `src/data/stratified_folds.py::build_stratified_fold_assignments` | Đọc label + hai drug ID từ train rồi validation; `StratifiedKFold(shuffle=True)`; lưu lookup trong RAM | Đã có thuật toán; thiếu schema/save/load/manifest |
| `StratifiedFoldAssignments.lookup` | Tra fold theo ID; fail nếu mẫu không có trong assignment | Có lookup; chưa đối chiếu raw label/source hash |
| `select_development_fold` | Ghép arrays, chọn held-out fold hoặc phần bù; giới hạn `max_rows` sau chọn fold | Có lựa chọn fold; cần gắn artifact provenance |
| `src/training/train_cil.py::main` | Dựng assignment khi khởi chạy từng member, cả lúc resume; dùng lại qua các task trong process đó | Chưa đọc assignment cố định từ disk |
| `train_cil.py::load_development_fold_split` | Load cả hai nguồn theo class filter, áp dụng scaler rồi chọn fold | Có train/validation integration; còn tốn RAM do ghép feature arrays |
| `src/training/tddi_ensemble3_study.py::_training_command` | Truyền `--ensemble-mode`, `--fold-id=member_id`, fold count và seed | Đã có mapping và chạy tuần tự |
| `src/eval/predictions.py::MemberPredictionContext` | Lưu run/member seed, mode, fold ID/count/seed và raw class order | Schema 2; chưa có assignment/source/task-file hash |
| `src/eval/ensemble_ue.py::aggregate_member_predictions` | Nối held-out validation thành OOF; mean probabilities trên test chung | Có hai nhánh, nhưng cần tăng kiểm tra provenance/coverage |
| `src/eval/threshold.py` | Chọn threshold từ OOF hoặc validation, freeze/load trước đánh giá test | Chạy được; policy OOF-to-ensemble cần quyết định lại |

Với A=fold 0, B=fold 1, C=fold 2:

| Member | Current training và nguồn exemplar | Validation/early stopping | Test |
| --- | --- | --- | --- |
| 0 | B+C | A | Test gốc |
| 1 | A+C | B | Test gốc |
| 2 | A+B | C | Test gốc |

Trong mỗi task, current training lọc class mới; held-out validation lọc toàn bộ class đã thấy. Buffer cập nhật từ `current_train` đã lọc fold (`train_cil.py`, phần `replay_buffer.update`). Test không được truyền vào builder fold.

CLI trực tiếp chỉ kiểm tra `fold_id` trong 0–2, không bắt buộc bằng `member_id`; orchestrator mới là nơi áp mapping đó. Cần giữ mapping explicit trong manifest và kiểm tra thống nhất.

## 3. Identity, thứ tự và seed

- Duplicate ordered pair bị chặn trong mỗi nguồn và giữa train/validation. `A|B` và `B|A` vẫn là hai ID khác nhau. Đây không phải group-aware split theo drug hoặc unordered pair; chưa kiểm tra development/test overlap. Chính sách group cần người dùng chốt ở Prompt 4.
- Cùng input có cùng thứ tự dòng, cùng seed và môi trường thư viện cho cùng assignment. Lookup được sort theo sample ID **sau khi** chia fold; việc sort đó không làm thuật toán bất biến với hoán vị dòng nguồn.
- Loader giữ thứ tự train rồi validation sau class filter, không sort feature rows theo ID. `max_rows` lấy phần đầu sau lọc nên không đại diện ngẫu nhiên cho toàn bộ fold.
- `fold_seed=42` điều khiển phân chia A/B/C. Prompt hiện minh họa `--fold-seed 0`: hai giá trị tạo study khác nhau, cần lựa chọn explicit và namespace riêng nếu đổi.
- `experiment_seed=0` điều khiển Python/NumPy shared RNG; task order thực tế lấy từ task JSON P3 đã cố định, không được seed này tự xây lại khi train.
- `src/utils/seed.py::derive_member_seed` dùng `numpy_seedsequence_v1`: member 0/1/2 nhận lần lượt `409845317`, `215626784`, `3041879697`. Torch initialization/dropout và `FixedReplaySampler` dùng member seed. Sampler còn kết hợp task ID và epoch.
- Buffer ghi `random_seed=experiment_seed`, nhưng `FixedBudgetReplayBuffer.update` hiện xếp exemplar theo khoảng cách tới class mean, tie-break bằng vị trí dòng; không lấy mẫu ngẫu nhiên trong bước chọn exemplar. Hai member có nguồn fold khác nhau có thể giữ exemplar/count khác nhau dù cùng experiment seed. Comment nói buffer “shared” chỉ phù hợp khi đầu vào thực sự giống nhau.
- Buffer hiện lưu features/labels và counts, chưa lưu retained sample IDs để audit trực tiếp exemplar thuộc training fold nào.

## 4. Artifact và preprocessing

Đã có: `run_config.json` (arguments có các tham số fold), checkpoint có `config.ensemble`, member NPZ schema 2, ensemble NPZ schema 3, study manifest có mode/count/seed, run IDs và hash của config/task file/prediction artifacts. Những hash này không thay thế hash dữ liệu nguồn hoặc assignment.

Chưa có:

- `fold_assignments.parquet`, `fold_manifest.json`, schema partition và assignment SHA256;
- `source_split`, `source_row_index`, raw label được lưu kèm assignment;
- hash/row count nguồn gắn với assignment, identity/algorithm version và mapping cố định;
- `scripts/build_stratified_3fold_assignments.py`, `scripts/audit_development_folds.py` và báo cáo audit fold (tên script build đã cập nhật sau audit);
- scaler theo member có fold provenance.

Config 3-fold vẫn trỏ tới `study_assets/preprocessing/scaler.pkl`. `scaler_config.json` ghi standard scaler, zero imputation, `rows_fitted=520841`; `preprocessing_report.md` ghi fit trên toàn bộ train gốc. `load_development_fold_split` áp cùng scaler cho cả hai nguồn trước khi chọn fold. Vì held-out mới chứa các mẫu từ train gốc, statistics của scaler có thể chứa chính held-out samples. Đây là vấn đề preprocessing leakage, dù các hàng đó không được đưa vào gradient training.

Hướng xử lý: fit scaler chỉ trên hai training folds của member, gắn assignment hash/member ID và kiểm tra khi load. Cần chốt thêm có cho phép preprocessing nhìn feature của các class tương lai hay không: fit toàn bộ hai training folds vẫn nhìn feature của các task sau. Không tự đổi policy trong audit này.

## 5. Checkpoint/resume và validation provenance

`train_cil.py::build_replay_checkpoint_config` (dòng 415) lưu `ensemble_mode`, `fold_id`, `fold_count`, `fold_seed` trong `config.ensemble`, cùng seed metadata, task-file/scaler/feature-column hashes. `load_replay_checkpoint(expected_config=...)` so sánh cả dictionary, nên direct trainer có thể từ chối khi những tham số fold này đổi.

Nhưng train/validation/test chỉ được ghi bằng đường dẫn, không có hash nội dung. Assignment được dựng lại trước khi load checkpoint: đổi dữ liệu hoặc thứ tự dòng ở cùng đường dẫn có thể đổi fold mà contract không phát hiện.

`tddi_ensemble3_study.py::_validate_replay_resume_checkpoint` (dòng 557) chỉ đối chiếu một tập field; thiếu `ensemble` và source/scaler/feature hashes. Vì vậy dry-run có thể báo resume hợp lệ rồi trainer mới từ chối. `_member_status` nhánh complete dựa vào completion/prediction files; `_validate_member_group` kiểm tra method/seed/task nhưng chưa so fold seed/ID với config hoặc raw class order với task file tại bước đó. Aggregate chỉ so các member với nhau.

Hướng xử lý tại Prompt 8–9: dùng một contract fold thống nhất cho trainer, orchestrator, skip-complete và ensemble. Fail trước khi train/aggregate khi assignment, mapping hoặc source hashes không khớp. Giữ behavior seeded và EWC hiện tại; EWC cũng đang dùng `fold_provenance`, tránh sửa helper chung làm đổi contract ngoài phạm vi.

## 6. OOF và UE: điều đã đúng, giới hạn còn lại

- OOF nối prediction của A/B/C và sort sample IDs; không mean các hàng khác nhau. Code kiểm tra đủ ba fold ID và không overlap, nhưng không có expected assignment để phát hiện một số mẫu bị thiếu hoặc mẫu gán sai fold.
- Với mỗi mẫu OOF chỉ có một phân phối dự đoán. Entropy, normalized entropy, entropy confidence và max probability mô tả prediction của member tương ứng.
- MI, normalized MI, member-normalized MI, hai variance và pairwise disagreement ở OOF hiện được gán 0. Phải diễn giải là **không đo được từ OOF này**, không phải ensemble chắc chắn hay ba member đồng ý. Metadata `member_count=3` mô tả ba model nguồn; số model dự đoán trên mỗi mẫu chỉ là 1.
- Test chung có đủ ba phân phối cho mỗi mẫu, được kiểm tra sample IDs/labels/class-column order rồi mean probabilities và tính UE. Tuy nhiên nhánh test chưa đối chiếu fold seed giữa các member; cả hai nhánh thiếu assignment hash.
- Threshold hiện chọn trên OOF rồi áp vào mean ensemble test; có validation/test guard và frozen-load guard. Nhưng confidence từ một member và mean của ba member có thể có phân phối khác nhau; đạt 95% trên OOF không bảo đảm 95% trên test. Held-out còn được dùng early stopping nên OOF cũng chịu ảnh hưởng chọn checkpoint.
- Prompt thiết kế yêu cầu common calibration split nếu muốn validation riêng cho ensemble threshold. Cần chốt common split hoặc chấp nhận OOF như proxy có ghi rõ giới hạn; chưa tự đổi policy/threshold trong Prompt 0.

## 7. Config đang bật

| Config trong `configs/` | Mode/chức năng |
| --- | --- |
| `train_tddi_p3_replay_distill_ensemble3_stratified_3fold_seed0.json` | P3, experiment seed 0, fold seed 42; output `_3fold_v2` |
| `train_tddi_p3_replay_distill_ensemble3_seeded_seed0.json` | Giữ train/validation gốc; output `_seeded_v2` |
| `eval_tddi_p3_ensemble_entropy_threshold.json` | Grid entropy confidence 0.50–0.99; threshold nhỏ nhất đạt selected accuracy 0.95; chưa áp minimum coverage |

Config 3-fold hiện giữ 6800 memory/member và 6800 replay draws/epoch; `load_study_config` còn khóa hai giá trị này bằng validation. Thiết kế tổng buffer 4%, quota sqrt hoặc replay fraction/repeat cap chưa được triển khai/chốt. Không sửa JSON rồi mặc định rằng các policy này đã hoạt động.

## 8. Kiểm chứng và độ bao phủ test

Lệnh baseline đã chạy:

```bash
python -m pytest tests/test_stratified_ensemble_mode.py tests/test_ensemble_seed.py tests/test_replay_checkpoint.py tests/test_tddi_ensemble3_replay_study.py tests/test_member_prediction_export.py tests/test_offline_ensemble.py tests/test_confidence_threshold.py -q
```

Kết quả: **58 passed in 42.85s**, không có failure trong nhóm này. Có tiny CPU training trong test checkpoint; không train dataset thật hoặc chạy full job. Chưa chạy toàn bộ test suite.

| Nhóm test | Đã bảo vệ | Chưa chứng minh |
| --- | --- | --- |
| `test_stratified_ensemble_mode.py` (3 tests) | Cùng seed/input cho cùng lookup; size và stratification ở fold 1; ghép OOF/threshold; chặn overlap OOF | Mọi member mapping, đủ source coverage, test leakage, per-member scaler |
| `test_ensemble_seed.py` | Derivation, sampler RNG và behavior seed legacy | Frozen fold/source provenance |
| `test_replay_checkpoint.py` | Round-trip, mismatch contract, teacher/sampler và continuous/resume tiny 2 task | Two-task resume với stratified mode; nguồn đổi nhưng path giữ nguyên |
| `test_tddi_ensemble3_replay_study.py` | Dry-run/selected member, sequential/skip/resume, config 3-fold được parse | Execution fixture chủ yếu seeded và fake runner; chưa end-to-end train 3-fold |
| Prediction/ensemble/threshold tests | Shape/ID/class order, numerical UE, compatibility, frozen threshold và test guard | Binding prediction rows về assignment nguồn và OOF applicability |

Kiểm tra bổ sung bằng dữ liệu giả trong thư mục tạm (không tạo thêm test/code trong repo):

- 27 mẫu, đảo thứ tự dòng train rồi dựng lại với seed 42: **18/27 sample IDs đổi fold**. Đây là tính phụ thuộc thứ tự input, cần ghi nhận và bảo vệ bằng source hash, không tự đổi thứ tự của implementation cũ.
- Ba OOF artifacts chỉ chứa tổng 5 hàng rời nhau vẫn được nhận: đủ fold ID không chứng minh đủ development rows.
- Ba test artifacts có cùng sample/class order nhưng fold seed 42/43/44 vẫn aggregate được: thiếu guard fold seed ở nhánh test.
- OOF nhận `member_count=3`, MI toàn 0 như code đang thiết kế.

Dry-run config 3-fold với `--member-id 0` được kiểm tra riêng; chỉ dựng command, không tạo model. Dry-run không xác minh dataset/fold coverage khi Parquet vắng mặt.

## 9. Checklist refactor đề xuất trước Prompt 1

- [x] Audit implementation, seed, loader, checkpoint, config và OOF/test semantics.
- [ ] Prompt 1: bổ sung schema/save/load/validation trong `src/data/stratified_folds.py`; giữ API runtime cũ trong lúc bổ sung. Acceptance: round-trip, source/hash/label validation, reject malformed assignment.
- [ ] Prompt 2–3: hai script build/audit mỏng gọi module đó; acceptance: một builder duy nhất, assignment coverage/rarity/test overlap và source hashes kiểm chứng được.
- [ ] Sửa tham chiếu còn sót trong **Prompt 2** từ `src/data/development_folds.py` sang schema của `stratified_folds.py` đã thống nhất. Tài liệu prompt gốc chưa bị sửa trong lượt audit này.
- [ ] Prompt 4: chốt fold seed 0 hay 42, group policy, preprocessing/future-class policy, buffer/replay và common calibration; số liệu phải lấy từ audit thật.
- [ ] Prompt 5–7: loader đọc assignment đã lưu, scaler đúng member và replay chỉ lấy training folds; kiểm tra bằng sample IDs, không chỉ counts.
- [ ] Prompt 8–9: contract partition thống nhất trên resume/skip/export/ensemble; bổ sung synthetic tests đổi nguồn, fold seed, member mapping và save/resume stratified.
- [ ] Prompt 10–11: ghi OOF metric không khả dụng rõ ràng; kiểm tra full coverage; audit common-test fold provenance; runbook mới theo decision record.

Phân công nguồn dữ liệu chuẩn: `sample_identity.py` giữ quy tắc ID; `stratified_folds.py` giữ thuật toán và schema; assignment + manifest đã audit là nguồn partition được training/resume/eval cùng tham chiếu. Hai script build/audit không chứa thuật toán chia fold thứ hai.

**Điểm dừng:** Prompt 0 hoàn thành. Đề xuất người dùng xác nhận hướng mở rộng `stratified_folds.py` trước Prompt 1; buffer/replay/calibration vẫn để quyết định sau audit dữ liệu ở Prompt 4.
