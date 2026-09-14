# Các prompt tiếp theo — T-DDI stratified 3-fold và pilot A/B

Cập nhật: 2026-09-15. Nguồn quyết định:
[TDDI_STRATIFIED_3FOLD_DECISION_RECORD.md](TDDI_STRATIFIED_3FOLD_DECISION_RECORD.md).

**Bước tiếp theo: Prompt 16 — threshold chính/fallback/frozen evaluation.**
Prompt 5–15 đã được triển khai và các quyết định preprocessing/ranking đã được duyệt;
giữ nội dung bên dưới
để tra phạm vi và acceptance, không thực hiện lại. Các bước audit code, fold schema,
build/audit và decision record trước đó đã hoàn thành. Giữ số prompt để nối lịch sử;
từ Prompt 7 trở đi là thứ tự mới, không dùng nội dung của prompt cũ cùng số.

Đã có assignment fold seed 42, 694.455 development samples/178 class, audit 19 PASS.
Giữ hai script riêng: `scripts/build_stratified_3fold_assignments.py` tạo assignment,
`scripts/audit_development_folds.py` kiểm tra. Không tạo thuật toán chia fold thứ hai.
Xem [hướng dẫn artifact/build/audit](TDDI_STRATIFIED_3FOLD_ARTIFACT.md) nếu cần tra lệnh
hoặc phục hồi backup; không bắt buộc chạy lại trước từng prompt.

## Cách sử dụng và giới hạn chung

- Mỗi lần giao **một prompt** trong khối text, kèm yêu cầu đọc decision record và
  giới hạn chung này. Đọc code hiện có, chỉ triển khai phần thiếu, không tạo module trùng.
- Mỗi lượt báo file đã sửa, lệnh/tests, kết quả, failure có sẵn và giới hạn còn lại.
- Chỉ unit/synthetic tests trên máy code. Dataset thật ở server; không tự chạy
  preprocessing, smoke, pilot hoặc full training thật trong các task viết code.
- Giữ EWC, replay legacy, loader API, config seeded/3-fold cũ khi không bật policy
  mới. Không đổi artifact/result cũ, không xóa output, không tự commit/push.
- Giữ P3, experiment seed 0, fold seed 42; member 0/1/2 có validation fold 0/1/2.
  Test không dùng chọn preprocessing/ranking/hyper/threshold.
- Pilot A: raw descriptors + input LayerNorm. Pilot B: scaler chỉ fit trên
  training task 0 của member, đóng băng; không fit toàn bộ hai training folds.
- Buffer/replay mới phải opt-in, có policy/version rõ; không gọi kết quả là recipe
  uniform cũ mà không mô tả thay đổi.
- Không bắt buộc common calibration split: đã chốt OOF threshold và giới hạn
  một prediction/mẫu. Chỉ mean probabilities trên common test.
- Rounding, tie-break, epsilon, variance convention, dtype và xử lí nonfinite phải
  được quy định/test/báo rõ; không âm thầm chọn hyper nghiên cứu khác.
- Các module/CLI mới dưới đây là vị trí dự kiến; tái sử dụng helper chung nếu phù
  hợp. Kiểm tra tên CLI thực tế bằng --help/tests trước khi viết runbook.
- Có mâu thuẫn cần quyết định nghiên cứu mới thì dừng hỏi, không suy ra phương án thắng.

## Lộ trình và điểm dừng

| Prompt | Công việc |
| --- | --- |
| 5 | Loader từ assignment đã lưu, raw features và IDs |
| 6 | Preprocessing A raw / B scaler-task0-frozen |
| 7 | Buffer quota sqrt, IDs và ranking chung tạm thời |
| 8 | Sampler replay fraction/cap/luân phiên |
| 9 | Tích hợp trainer, validation và audit |
| 10 | Checkpoint/resume cho policy mới |
| 11 | Config/orchestrator smoke và pilot A/B |
| 12 | Công cụ so sánh validation + runbook GPU |
| **Dừng** | Người dùng chạy smoke/pilot trên server và gửi artifacts |
| 13 | Đọc kết quả preprocessing, chờ người dùng duyệt |
| 14 | Pilot cách chọn exemplar sau khi preprocessing được duyệt |
| **Dừng** | Chờ kết quả và người dùng chốt ranking cuối |
| 15 | Prediction/OOF/ensemble UE và provenance |
| 16 | Threshold chính/fallback/frozen evaluation |
| 17 | Pilot ba member và điều kiện trước full tám task |

Prompt 5–12 triển khai hai phương án đã được phép thử, không cần chốt một
preprocessing cuối. Không thực hiện phân tích kết quả giả ở Prompt 13 khi chưa
có artifact thật. Full run chưa được tự động cho phép.

Prompt 13 đã đọc bundle thật ngày 2026-09-14 và người dùng đã duyệt preprocessing
B `task0_standard_frozen`; xem
[báo cáo pilot preprocessing](TDDI_PREPROCESSING_AB_PILOT_RESULTS.md). Prompt 14
đã mở rộng đối chứng validation member 0 đến task 7 và người dùng chốt
`raw_sample_normalized_class_mean_control_v1` ngày 2026-09-15. Các pilot này không
phải kết quả test hoặc official ensemble study.

## Prompt 5 — Loader fold-aware từ artifact đã audit

```text
Đọc decision record và giới hạn chung của file prompts. Chỉ triển khai loader;
chưa sửa trainer, preprocessing hoặc config training.

Mở rộng src/data/ddi_dataset.py, dùng src/data/stratified_folds.py và
src/data/sample_identity.py để load hai Parquet nguồn theo assignment/manifest
đã lưu. Không dựng lại StratifiedKFold khi load.

API mới nhận role=train/validation, member_id, validation_fold và raw class_ids.
Validate mapping, assignment/source hashes, row coverage và ID/label agreement.
Cho phép đổi path nguồn nếu byte hashes không đổi. Trả raw descriptors, labels,
sample IDs và source-row provenance; chưa tự áp scaler cũ.
Giữ source order train rồi validation, index nguồn sau lọc. Không copy sáu feature
Parquet. Có thể cache validated context để không hash hàng GB mỗi batch, nhưng
phải validate đầu run/resume, không bỏ kiểm tra integrity.

Tests synthetic: mọi member/role/class filter, row order và metadata alignment,
hash/source/label mismatch, held-out/test không vào train, deterministic A/B rows.
Giữ load_split_arrays và runtime fold API cũ. Không chạy dataset thật.
```

Acceptance: A/B nhận cùng raw rows/IDs; sai partition fail trước training; API cũ
vẫn qua tests. Raw class IDs không được mặc định liên tục 0–177.

## Prompt 6 — Preprocessing A/B, không fit task tương lai

```text
Đọc decision record và giới hạn chung. Chỉ triển khai preprocessing trên loader
Prompt 5; không đổi behavior scripts/preprocess_features.py và scaler legacy.

Thêm API/module và CLI chuẩn bị artifact riêng, ví dụ src/data/fold_preprocessing.py
và scripts/prepare_fold_preprocessing.py. Hai policy explicit/versioned:
A raw_identity: không fit dataset scaler, giữ input LayerNorm.
B task0_standard_frozen: scaler riêng từng member, chỉ fit rows thuộc hai training
folds và classes task 0 trong P3 task file; không refit task 1–7 hoặc resume.

Train/replay/validation/test của B transform bằng cùng scaler. Không fit từ held-out,
test, future classes hoặc dùng lại scaler cũ. Raw mode cũng có metadata nhưng không
giả thống kê như đã fit. Ghi policy/version, member/fold/seed, assignment/source/
task-file/feature-order hashes, fit task/classes/rows, mean/scale, variance/dtype.
Quy định zero-variance/nonfinite policy chung; không âm thầm clip/drop rows hoặc
dùng future/validation statistics để thay thế. Ghi convention rõ trong docs/tests.
Scan theo batch, atomic write vào namespace mới, không overwrite.

Tests tính tay scaler task0-train, raw identity, source-row fit coverage, sửa future/
held-out values không đổi fit statistics (source hash vẫn phải phản ánh file đổi),
transform/frozen round-trip, nhầm member/policy/hash fail, legacy compatibility.
Không chạy preprocessing dataset thật hoặc training.
```

Acceptance: chứng minh đúng tập IDs dùng fit, không chỉ đúng row count; artifact
portable sang server, scaler B cố định xuyên task.

## Prompt 7 — Buffer mới: quota q=10 + sqrt, IDs và ranking chung

```text
Đọc decision record và giới hạn chung. Chỉ triển khai buffer/ranking mới, chưa sửa
sampler/trainer. Tham khảo src/data/fixed_budget_replay.py; giữ allocator/buffer
legacy nguyên behavior. Thêm policy/class có tên/version riêng.

Study thật: floor(4%*694455)=27778 slots, member0/1/2=9260/9259/9259. Cho phép
overlap giữa member, mỗi bản lưu tính một slot; không nhân bản ID trong một buffer.
Thuật toán dùng chung nhận budget/count làm input để test synthetic nhỏ.

Quota nền q10 giới hạn feasible capacity; phần dư chia theo sqrt số training samples
class đó của member khi class xuất hiện. Không dùng future-class/held-out/test counts.
Tách observed count (trọng số) và retained capacity: class cũ chỉ còn buffer, class
mới có current data. Redistribute khi chạm capacity, không lấy discarded old data.
Quy định rounding integer/tie-break deterministic, test budget nhỏ hơn quota nền,
rare class, saturated class, total feasible size và trường hợp quota cũ muốn tăng.

Ranking tạm cho hai pilot: từ raw descriptors normalize từng mẫu không learned affine,
cố định epsilon/variance/dtype; Euclidean tới class mean trong không gian đó, tie-break
sample ID. Không phụ thuộc scaler A/B, model latent hay UE. Class cũ giữ rank prefix.
Lưu sample IDs/raw labels/source provenance/rank priorities/counts. Tách selection
representation khỏi model inputs; A/B phải chọn cùng IDs nhưng model inputs có thể khác.
Không giữ full dữ liệu task cũ ngoài buffer. Chuẩn bị state serialization cho resume.

Tests tính tay allocation/ranking, saturation, no held-out/test/future exemplars,
old trimming, A/B retained-ID equality, overlap/global-slot accounting và legacy.
Không train dataset thật.
```

Acceptance: ranking này chỉ là control cho pilot; không coi là thuật toán cuối
tối ưu. Log rõ policy mới khác uniform buffer cũ.

## Prompt 8 — Sampler fraction/cap và luân phiên

```text
Đọc decision record và giới hạn chung. Chỉ thêm sampler policy mới; giữ
FixedReplaySampler legacy trong src/data/fixed_budget_replay.py. Chưa nối trainer.

Mỗi task/epoch dùng mọi current sample đúng một lần. Replay target f=0.125 tổng draws,
xấp xỉ N_current/7; định nghĩa rounding integer rõ. Task0 replay=0.
Cap3/exemplar/epoch; nếu thiếu capacity thì giảm replay, không bỏ current hoặc vượt cap.
Không ép 56+8 mỗi microbatch và không drop current tail batch.

Ưu tiên đều old classes, cap-aware redistribution. Luân phiên class khi draws ít
hơn classes; trong class luân phiên IDs trước lặp để không luôn dùng cùng prefix.
Buffer lớn chỉ replay số cần, không nhất thiết hết buffer.
RNG member/task/epoch riêng khỏi model/dropout. Hai pilot có early stopping khác
nhau vẫn phải cùng sampler order tại cùng member/task/local epoch nếu memory IDs
giống nhau: số epoch task trước không được làm lệch A/B ở task sau.
Quy định reset state ở task boundary, serialization epoch/cursors/queues/RNG cần thiết
và deterministic resume; không dùng hash() ngẫu nhiên của Python.

Audit target/actual fraction, current/replay draws, per-class exposure, unique IDs,
repeat histogram/max repeat và coverage. Tests current đầy đủ, cap, thiếu capacity,
rare class, task0, tiny N/zero replay, fairness qua epoch, A/B order, state round-trip.
Không đổi effective batch hoặc loss reduction và không train dataset thật.
```

Acceptance: deterministic order và cap được chứng minh bằng tests theo sample IDs;
các state phải rõ để Prompt 10 nối resume.

## Prompt 9 — Tích hợp trainer và validation pilot

```text
Đọc decision record và giới hạn chung. Nối API Prompt 5–8 vào
src/training/train_cil.py bằng opt-in CLI/policy. Không đổi old seeded/stratified/EWC
khi không bật new mode. Chỉ sửa src/methods/replay.py nếu integration thật sự cần.

New mode đọc assignment đã lưu, mapping member=held-out fold, source hashes đúng.
Current chỉ class mới trong hai training folds; validation là all-seen classes
trong held-out. Tách raw ranking features khỏi model inputs A/B. Nối buffer/sampler
mới, kiểm tra runtime quota/cap/IDs/current coverage; không đọc discarded old data.
Giữ model size/head expansion/teacher raw-class mapping và baseline loss formulas.
AdamW mới mỗi task, không thêm scheduler. Log riêng classification, logit distillation,
feature-distillation và total loss, cùng các trọng số tương ứng.

Microbatch64/accumulation16/effective target1024; không bỏ tail batch, log actual steps
và tail effective batch. Early stopping theo validation Macro-F1.
Thêm validation-only pilot mode: không cần chạy/export test để chọn A/B.
Ưu tiên giữ full P3 task file/hash, thêm execution stop-after-task1 rõ ràng thay vì
âm thầm thay class protocol. Kiểm tra đủ layout178 và raw class mapping.

Ghi run_config/audit policy versions, assignment/preprocess/ranking/budget/sampler
seeds/hashes. Synthetic two-task smoke nhỏ cho A/B: current/retained IDs và order,
old/new validation metrics, head expansion và loss finite; legacy regression.
Không chạy training thật. Checkpoint new mode hoàn thiện ở Prompt10.
```

Acceptance: audit đủ dữ liệu kiểm chứng A/B chứ không chỉ console; không chọn
checkpoint bằng test; policy khác được ghi rõ, không trộn result cũ.

## Prompt 10 — Task-boundary checkpoint/resume mới

```text
Đọc decision record và giới hạn chung. Mở rộng src/training/replay_checkpoint.py
và integration trainer cho new policy; không thay EWC hoặc replay legacy contract.

Checkpoint versioned sau task lưu best model_state, raw class map, completed/next task,
fold/source/assignment/task hashes, preprocessing policy/scaler state hoặc reference
hash cần thiết, budget/allocation/ranking/sampler versions, buffer IDs/labels/features/
observed counts/capacities/rank priorities, RNG/scheduling state và metric/tracker/audit.
Không lưu teacher trùng: clone từ boundary model khi resume. Không bắt buộc optimizer
vì task mới tạo optimizer; không claim mid-epoch resume.
Tách pilot-scope complete task1 với full-trajectory complete task7.

Fail rõ A/B/scaler/member/fold/hash/class map/budget mismatch. Không refit scaler B
hoặc rebuild folds khi resume. Atomic writes, không re-export/overwrite task hoàn tất,
không xóa artifact. Tests continuous vs resume 2–3 task cho A/B: model/head/teacher/
buffer/class map/next task/metrics trong tolerance; sampler order đúng và mismatch
guards. Legacy checkpoint vẫn chạy. Không train dataset thật.
```

Acceptance: completion xác minh bằng metadata/artifacts, không chỉ directory;
scope prefix không khiến full run bị skip nhầm.

## Prompt 11 — Config/orchestrator smoke và pilot A/B

Đã triển khai entrypoint riêng `src/training/fold_ab_study.py` và bốn config smoke/pilot.
CLI, namespace, validation và giới hạn:
[Config/orchestrator A/B](TDDI_STRATIFIED_3FOLD_AB_STUDY.md).
Không tự chạy dữ liệu thật. Bước tiếp theo là Prompt 12 (report/runbook GPU).

```text
Đọc decision record và giới hạn chung. Mở rộng src/training/tddi_ensemble3_study.py
hoặc entrypoint pilot nhỏ dùng helpers chung. Tạo config smoke A/B và pilot A/B mới,
không ghi đè config hiện có. Không tự launch training.

P3 study_assets/task_protocols/tail_to_head_tasks.json, seed0/fold_seed42, member0,
execution task0–1, same assignment. A raw và B task0_standard_frozen; paths nguồn/
assignment/manifest/preprocessing explicit và có server override.
Budget dùng full development694455, không tính lại từ pilot prefix: member0=9260,
global27778, planned member1/2=9259. q10+sqrt, f0.125/cap3/class-uniform replay,
same temporary ranking. Smoke2–3 epochs; pilot20/patience5. Giữ microbatch64,
effective1024, accumulation16, AdamW lr0.001/wd0.0001, GELU/dropout0.2/LayerNorm,
focal1, distill alpha1/T2/feature0.5.

Gỡ khóa6800 chỉ cho new policy, giữ validation legacy. Manifest policy/scope/run IDs/
fold/data/order hashes. Pilot mặc định validation-only, không bắt đủ ba member hoặc
test/threshold. Dry-run không tạo model; --execute mới được launch. A/B và member
tuần tự; namespace riêng smoke/pilot/A/B. Skip complete đúng scope, resume valid,
incomplete không checkpoint fail rõ, không overwrite.
Tests runner giả cho sequential/selected member/skip/resume, config equality ngoài
preprocessing/output fields, budget/scope/dry-run và legacy regressions.
```

Acceptance: tên/path config và CLI thực tế được ghi rõ; test bảo vệ không có khác
biệt ẩn về hyper, data hoặc exemplar selection giữa A/B.

## Prompt 12 — Report validation A/B và runbook GPU

Đã có `scripts/compare_preprocessing_pilots.py` (JSON/Markdown, report-only bundle).
Lệnh chuẩn bị, dry-run, nohup, resume/OOM và thu thập:
[Runbook GPU preprocessing A/B](TDDI_PREPROCESSING_AB_PILOT_RUNBOOK.md).
Không tự thực hiện Prompt 13 khi chưa có kết quả thật của người dùng.

```text
Đọc decision record và giới hạn chung. Tạo công cụ so sánh, ví dụ
scripts/compare_preprocessing_pilots.py và docs/TDDI_PREPROCESSING_AB_PILOT_RUNBOOK.md.
Không train và không tự chọn pipeline thắng trong task này.

Report chỉ dùng validation/training audit: seen_all task1 Macro-F1/Balanced Accuracy,
old/new classes theo P3, epoch curves/best epoch, loss components, optimizer steps,
runtime/peak allocated+reserved VRAM/checkpoint size, replay coverage/cap.
Kiểm tra same fold/member/task/hyper, retained IDs và sampler order ở các epoch cả
hai đã chạy. Fail/flag mismatch trước so điểm. Thiếu telemetry ghi unavailable,
không bịa hoặc tự train lại. Xuất JSON/Markdown, không gộp mean stage thành final.

Runbook dùng CLI thực tế: environment/data/P3/fold hashes, chuẩn bị B, dry-run,
smoke A/B rồi pilot A và B bằng nohup CUDA0, PID/log/VRAM, task-boundary resume,
thu thập/so sánh artifacts. Scope chỉ member0 task0–1. Không chạy ba member/threshold
để chọn preprocessing. Namespace riêng; không mkdir outdir gây conflict guard.
Nếu OOM, giảm microbatch/tăng accumulation giữ effective batch, áp cùng điều kiện
cho đối chứng và ghi config; không âm thầm thay hyper hoặc overwrite incomplete run.
Bundle review cần configs/provenance, validation metrics, memory-ID/sampler-order
audit hoặc digests, logs/telemetry. Phân biệt report-only với backup đầy đủ có cả
assignment+manifest; không đóng gói dataset lớn nếu không cần.

Synthetic end-to-end tests nhỏ và CLI help/dry-run. Runbook dừng sau báo cáo A/B,
đợi người dùng gửi kết quả cho Prompt13; không tự chạy full.
```

**Dừng tại đây để chạy trên server.** Prompt 13 cần kết quả thật. Không tự sửa
preprocessing/ranking cuối dựa trên dry-run hoặc synthetic metrics.

## Prompt 13 — Đọc pilot preprocessing và đề xuất, chưa tự chốt

Trạng thái: **đã hoàn thành; người dùng đã duyệt B `task0_standard_frozen`.**

```text
Đọc decision record và A/B artifacts thật người dùng gửi. Kiểm tra cùng data/member/
hyper/exemplar IDs/sampler order ở epoch chung; early stopping có thể khác.
Thiếu hoặc mismatch evidence thì báo rõ, không giả định so sánh hợp lệ.

So validation task1 seen_all Macro-F1/Balanced Accuracy, old/new classes, curves/
loss/telemetry. Không dùng test hoặc threshold để chọn. Kết luận giới hạn ở
member0/task0–1 với ranking tạm; chưa chứng minh khả năng task7.
Nếu sát nhau/trade-off, đề xuất thêm task/member, không ép winner.
Tạo báo cáo docs và bổ sung evidence/proposal vào decision record. Không đánh dấu
preprocessing approved trước xác nhận người dùng, không sửa training/config hoặc
tự chạy thử tiếp. Dừng chờ quyết định trước Prompt14.
```

## Prompt 14 — Pilot không gian exemplar sau khi preprocessing được duyệt

Trạng thái: **đã hoàn thành pilot thật; người dùng đã duyệt sample-normalized.**
Xem [runbook pilot exemplar ranking](TDDI_EXEMPLAR_RANKING_PILOT_RUNBOOK.md).

```text
Chỉ làm khi người dùng đã duyệt preprocessing từ Prompt13. Đọc decision record
cập nhật; nếu chưa chốt thì dừng, không tự chọn A/B.

Giữ preprocessing, quota/replay/seeds/hyper/folds. Tạo hai ranking policy/version:
normalize từng mẫu tạm hiện tại và không gian input của pipeline đã chọn; đều
Euclidean gần class mean, tie-break sample ID. Đây là đối chứng ranking nên IDs
có thể khác, không ép same-ID invariant của pilot preprocessing.
Không dùng learned latent/UE hay policy thứ ba nếu chưa được yêu cầu.
Class cũ chỉ rút gọn retained buffer, không đọc discarded old data.
Nối ranking provenance vào config/checkpoint, thêm mismatch/round-trip tests.

Chuẩn bị config/runbook và validation comparison cho hai ranking pilots riêng
namespace. Không tự train thật hoặc chọn winner. Khi có kết quả, báo trade-off,
cập nhật đề xuất rồi chờ người dùng duyệt ranking cuối, không auto full job.
```

**Dừng sau kết quả exemplar.** Cần người dùng chốt preprocessing/ranking cuối trước
study ba member chính thức. Helpers offline Prompt 15–16 có thể làm sớm nếu được
yêu cầu riêng, nhưng không được tự suy ra policy thắng để train.

Quyết định ngày 2026-09-15: giữ preprocessing B `task0_standard_frozen` và chọn
`raw_sample_normalized_class_mean_control_v1` cho exemplar ranking. Có thể tiếp tục
Prompt 15; không tái sử dụng output pilot làm official run.

## Prompt 15 — Prediction provenance, OOF và ensemble UE

Trạng thái: **đã triển khai bằng schema member v3 và offline ensemble v4.**
OOF có `member_count=3` ở mức nguồn nhưng `prediction_count=1` trên mỗi dòng;
MI/variance/disagreement được đánh dấu không khả dụng bằng metadata và `NaN`, không
được diễn giải như giá trị 0. Common test vẫn mean probabilities của đủ ba member.

```text
Đọc decision record và giới hạn chung. Mở rộng src/eval/predictions.py,
src/eval/ensemble_ue.py và integration cần thiết; không train.

Artifacts giữ IDs/labels/task/member/seeds/raw-class column order, thêm partition/
preprocess/ranking/budget provenance. Đối chiếu IDs/labels với assignment và seen
classes. Scaler hashes giữa member có thể khác hợp lệ: phải đúng member và cùng
policy, không assert hash scaler của ba member bằng nhau.
OOF nối đúng held-out A/B/C, mỗi expected seen-class sample đúng một lần/đúng fold.
Không mean A/B/C như cùng validation set. Ghi source member_count3 nhưng per-sample
prediction_count1; MI/variance/disagreement không khả dụng từ OOF này, không dùng
giá trị0 như bằng chứng ensemble chắc chắn. Version/schema mới, legacy vẫn đọc được.

Common test chỉ aggregate đủ ba member cùng sample/label/task/class order và study
contract; mean probabilities, giữ UE metrics/edge cases. Không dùng test chọn hyper.
Tests coverage/duplicates/misalignment/wrong-fold/provenance, numerical UE và legacy
load; không sửa artifact lịch sử hoặc tạo common calibration split.
```

## Prompt 16 — Threshold chính + fallback, freeze trước test

```text
Đọc decision record và giới hạn chung. Mở rộng src/eval/threshold.py với policy/config
eval mới, giữ legacy behavior; không train hoặc thêm post-hoc temperature calibration.

Mỗi task dùng OOF seen-class để chọn, entropy_confidence/raw probabilities,
grid0.50..0.99 step0.01. Chính: threshold nhỏ nhất selected OOF accuracy>=0.95.
Chỉ nếu không đạt: fallback candidates coverage>=0.50, accuracy cao nhất; hòa thì
coverage cao hơn rồi threshold thấp hơn. Không candidate hợp lệ: threshold=null,
status=no_selection, không lọc. Candidate 0 mẫu không được đạt; fallback không
giả là đạt95%. Không áp minimum coverage fallback lên primary rule.

Freeze full grid/results/rule/status target_met/fallback/no_selection, source split,
task/class/fold metadata/hashes. Test chỉ load frozen artifact, không refit; validate
liên kết OOF với test qua provenance đúng, không đòi OOF/test có cùng sample IDs.
Report full-set, selected metrics/coverage khi có, ECE/NLL/Brier và score semantics.
OOF95% không bảo đảm test95%; entropy confidence không phải max probability.
No-selection vẫn chạy pipeline tiếp; không hard-code0.88.

Tests chính/fallback/ties/no_selection/zero coverage, validation-only guard,
frozen round-trip, mismatch task/class/partition và legacy compatibility.
```

## Prompt 17 — Pilot ba member và điều kiện trước full P3

```text
Chỉ tạo config training chính thức khi người dùng duyệt preprocessing/ranking cuối.
Đọc decision record, tái sử dụng orchestrator/checkpoint/policy mới đã có, không
nhân đôi trainer. Không tự train dataset thật.

Tạo pilot task0–1 ba member chạy tuần tự0→1→2, P3/seed0/fold_seed42,
budgets9260/9259/9259, hyper baseline trừ thay đổi đã được duyệt.
Sau đủ ba member mới OOF → frozen threshold/fallback → ensemble test/UE/report.
Giữ skip/resume đúng scope/provenance, incomplete không checkpoint fail.
Runbook dry-run/nohup CUDA0/PID/log/telemetry/acceptance từng task/member, ensemble/
OOF/threshold commands và review bundle. Synthetic integration tests sequential/
skip/resume/coverage/fallback/no_selection và legacy regressions.

Task0–1 chưa chứng minh hiệu quả task7. Full8 là bước sau cần người dùng cho phép,
không auto launch/chaining. Nếu chuẩn bị template full8, mặc định dry-run, kiểm tra
layout178/hash P3, output tách pilot/cũ. Kết thúc bằng artifact cần gửi để go/no-go,
không tự đổi method/hyper.
```

## Kết thúc

Chỉ sửa phạm vi prompt đang làm, giữ thay đổi người dùng trong worktree. Không xóa
Parquet nguồn, assignment/manifest, scaler cũ, P3 task file, checkpoint hay result cũ.
Nếu code và tài liệu chưa khớp, báo phần thiếu thay vì giả định policy đã hoạt động.
