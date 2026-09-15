# Quyết định thiết kế — T-DDI 3-fold replay ensemble

Ngày tổng hợp: 2026-09-13.

Tài liệu ghi lại các quyết định người dùng đã thống nhất. Đây là **thiết kế cần
triển khai**, không xác nhận code/config hiện tại đã hỗ trợ đầy đủ. Không tự động
khởi chạy training. Không sửa hoặc ghi đè artifact/result lịch sử.

## 1. Nguồn số liệu và trạng thái

Nguồn: gói `fold_backup_full.tar.gz` người dùng gửi, gồm assignment, manifest và bốn
báo cáo audit. Gói đã được kiểm tra đọc được, đủ sáu file; hash assignment khớp
manifest/audit. Dataset nguồn vẫn ở server, chưa được audit lại trên máy code.

- Artifact schema: `ddi_cil_development_folds`, version 1.
- Assignment SHA256: `c84dc034431ed27907cdfd7217fa3460e0a9480bbf8343e5c91e68927de79a43`.
- Train gốc: 520.841 mẫu.
- Validation gốc: 173.614 mẫu.
- Development: **694.455 mẫu**, 178 class.
- Test chung: **173.614 mẫu**.
- Audit: **19 checks PASS**, không có warning.
- Ordered development/test overlap: 0.
- Unordered development/test overlap và tổng giao unordered pairs giữa folds: 0.
- Các kiểm tra trên không bảo đảm drug-disjoint.

Dữ liệu vẫn rất mất cân bằng: class 137 có 98.179 mẫu development, class 111 có
4 mẫu; 48 class có dưới 30 mẫu development. Stratification phân phối từng class
đều giữa folds, không làm các class có số mẫu bằng nhau.

Giữ `fold_assignments.parquet` cùng `fold_manifest.json` trên server và trong backup.
Gói chỉ chứa báo cáo không thay thế assignment; cũng không thay thế ba dataset nguồn.

## 2. Study và partition — đã chốt

- Backbone: `tddi_paper_member`, ba member `0, 1, 2`.
- Tiếp tục replay + distillation; chưa chuyển sang EWC/GEM/A-GEM.
- Protocol **P3 tail-to-head**, tám task, layout `[38,20,20,20,20,20,20,20]`, 178 class.
- Giữ task protocol P3 hiện tại; không tự tạo lại protocol theo fold counts.
- Experiment seed: **0**; member seeds deterministic theo derivation hiện có.
- Giữ partition row-level đã audit: ba stratified folds, **fold seed 42**.
- Không tạo common calibration split trong thiết kế hiện tại; threshold dùng OOF.

| Member | Train folds | Validation fold | Train samples | Validation samples | Test samples |
| --- | --- | --- | ---: | ---: | ---: |
| 0 | 1+2 | 0 | 462.970 | 231.485 | 173.614 |
| 1 | 0+2 | 1 | 462.970 | 231.485 | 173.614 |
| 2 | 0+1 | 2 | 462.970 | 231.485 | 173.614 |

Mỗi member dùng đúng 2/3 development để train, 1/3 để validation. Trong CIL, các
view trên tiếp tục được lọc theo class/task; không đưa toàn bộ task tương lai vào
gradient training. Validation tại giai đoạn t chỉ gồm các class đã thấy.

Partition được chuẩn bị offline bằng nhãn toàn development. Phải phân biệt bước
thiết kế split này với dữ liệu được phép dùng khi training/preprocessing từng task.

## 3. Tổng buffer — đã chốt

- Ba buffer riêng, cho phép exemplar trùng giữa member nếu thuộc training folds hợp lệ.
- Mỗi bản lưu tính một slot. Tổng slots không phải số sample IDs duy nhất của hợp ba buffer.
- Tổng budget tối đa: `floor(0.04 × 694455) = 27778` slots.
- Phân chia: member 0 **9.260**, member 1 **9.259**, member 2 **9.259** slots.
- Đây là giới hạn tối đa, không bắt buộc đầy khi số mẫu hợp lệ chưa đủ.
- Chỉ chọn từ dữ liệu task hiện tại và exemplar cũ đang giữ. Không truy xuất lại
  dữ liệu task cũ đã bỏ để bù quota; không lấy validation/test.
- Cần lưu sample IDs để audit nguồn exemplar, overlap giữa buffer và replay exposure.

## 4. Phân bổ slots theo class — đã chốt

1. Cấp quota nền `q=10` cho mỗi class đã học, giới hạn bởi số mẫu hợp lệ có thể giữ.
2. Phần budget còn lại phân bổ theo **căn bậc hai tần suất class trong training data
   của member**, không dùng validation/test counts.
3. Class hết khả năng nhận thì chuyển slots dư sang class còn khả năng nhận.
4. Class cũ chỉ có thể giữ/rút gọn exemplar đang còn trong buffer.

Class có ít mẫu không được nhân bản để lấp slots. Ví dụ class 111 có 4 mẫu
development, chia 1–2–1; training member 0/1/2 chỉ có 3/2/3 mẫu hợp lệ.

Chi tiết làm tròn quota nguyên và tie-break khi phân bổ phải được quy định
deterministic trong implementation/tests, không để phụ thuộc iteration order.
Đây là policy mới so với capacity-constrained max-min allocation cũ; phải ghi rõ
policy/version trong manifest/audit, không gán toàn bộ kết quả mới cho method cũ
mà không mô tả thay đổi.

## 5. Current sampling và replay — đã chốt

- Current đầy đủ: mỗi epoch dùng mọi training sample thuộc task hiện tại của
  member, mỗi mẫu một lần, thứ tự shuffle deterministic.
- Replay mục tiêu **12,5% tổng lượt học**, tức khoảng `current_draws / 7` lượt replay.
- Mục tiêu theo epoch, **không bắt buộc mỗi microbatch đúng 56 current + 8 replay**.
- Repeat cap: **3 lần/exemplar/epoch**; bộ đếm reset ở epoch mới.
- Khi không đủ khả năng replay: giảm replay draws, không bỏ current samples,
  không vượt cap để ép đạt 12,5%.
- Buffer lớn không có nghĩa phải replay toàn bộ buffer mỗi epoch.
- Task 0 không replay.

Phân phối lượt replay:

1. Ưu tiên đều giữa các class cũ.
2. Class chạm giới hạn exemplar/cap thì chuyển lượt dư sang class còn khả năng replay.
3. Nếu lượt replay ít hơn số class: luân phiên class giữa các epoch.
4. Trong class: luân phiên exemplar, ưu tiên dùng trước khi lặp.
5. RNG theo member seed; trạng thái scheduling cần được tái lập khi resume.

Ghi audit về current/replay draws, fraction thực tế, per-class draws, độ phủ class/
exemplar và số lần lặp. Cách làm tròn replay draws và xử lí batch cuối cần được
quy định/test rõ khi triển khai.

## 6. Preprocessing và exemplar — kiểm chứng hai giai đoạn

**Đã chốt preprocessing B `task0_standard_frozen` ngày 2026-09-14.** Mỗi member
fit scaler riêng chỉ trên training rows task 0 của chính member, sau đó đóng băng.
Không dùng scaler member 0 cho member 1/2; không fit lại bằng task tương lai,
validation hoặc test.

### Giai đoạn A: hai pilot preprocessing

| Thành phần | Pilot A — raw | Pilot B — scaler task 0 |
| --- | --- | --- |
| Đầu vào model | Descriptor gốc → LayerNorm → MLP | Descriptor qua StandardScaler → LayerNorm → MLP |
| Fit scaler | Không | Chỉ training task 0 của member 0 |
| Các task sau | Giữ cách xử lí | Đóng băng scaler task 0, không fit lại |
| Validation/test | Cùng đường xử lí input của member | Transform bằng scaler đã fit, không fit trên validation/test |
| Member/task ban đầu | Member 0, task 0–1 | Giống A |
| Hyper | Giữ hyper cũ làm mốc | Giống A |
| Buffer/replay | Policy mới ở mục 3–5 | Giống A |
| Output | Namespace A mới | Namespace B mới, tách biệt A |

Cùng assignment, member seed, exemplar IDs và sampling order theo epoch. Số epoch
thực chạy có thể khác do early stopping; phải ghi nhận. Không đồng thời đổi activation,
optimizer hoặc các loss weights để rồi quy chênh lệch kết quả riêng cho scaler.

Để có exemplar IDs chung, **tạm dùng quy tắc xếp hạng chung cho đối chứng**:

- Chuẩn hóa descriptor từng mẫu bằng `(x - mean(x)) / sqrt(var(x) + epsilon)`,
  không có tham số học và không fit dataset.
- Tính trung bình class trong không gian này, chọn gần trung bình theo Euclidean.
- Tie-break bằng sample ID; giữ thứ tự ưu tiên khi rút gọn class cũ.
- Cố định epsilon, định nghĩa variance và dtype trong implementation để tái lập.
- Vector này chỉ dùng chọn ID; đầu vào model/replay vẫn theo preprocessing A/B.

**Không xem cách xếp hạng tạm thời này là tối ưu.** Hai pilot trả lời preprocessing
nào tốt hơn *với cùng bộ exemplar đó*, không chứng minh toàn bộ pipeline tốt hơn run cũ.
Chuẩn hóa theo từng mẫu cũng không bảo đảm loại bỏ mọi ảnh hưởng chênh thang đo feature.

### Giai đoạn B: kiểm chứng không gian chọn exemplar

Sau khi có kết quả preprocessing:

- Giữ preprocessing triển vọng cố định.
- So sánh không gian chuẩn hóa từng mẫu với không gian đầu vào của pipeline đã chọn,
  hoặc phương án tiếp theo được người dùng duyệt.
- Giữ quota, replay policy, seed và điều kiện đánh giá để kiểm soát thí nghiệm.
- Nguyên tắc đang theo đuổi: **chọn gần trung bình class**. Không gian khoảng cách
  cuối cùng chờ kết quả, không tự quyết định trong lượt triển khai đầu.
- Nếu hai pilot chưa rõ, mở rộng kiểm tra; không ép chọn phương án thắng.

### Evidence pilot member 0, task 0–1 — 2026-09-14

Hai pilot thật trên server đã hoàn tất và bundle report-only được comparator đọc lại
thành công. Alignment `PASS`: A/B cùng assignment/task hash, current/validation/replay
IDs, retained exemplar IDs và sampler order ở epoch chung. Không dùng test để so sánh.

Tại task 1 `seen_all`, B (`task0_standard_frozen`) đạt Macro-F1 `0,859409` và
Balanced Accuracy `0,853691`; A (`raw_identity`) đạt lần lượt `0,838958` và
`0,830732`. B cũng giữ old classes tốt hơn, trong khi A tốt hơn nhẹ trên current
classes. A đạt best epoch đúng giới hạn 20; dữ liệu tail chỉ tạo một optimizer step
mỗi epoch. Vì vậy **đề xuất tạm thời là B**, nhưng chưa coi là winner cuối cho toàn
pipeline và chưa được phép tự chuyển sang full8.

Trạng thái: **người dùng đã tạm duyệt B ngày 2026-09-14 để thực hiện Prompt 14**.
Prompt 14 chỉ so sánh ranking trên validation; không tự chọn ranking hoặc chạy full8.
Báo cáo đầy đủ:
[Kết quả preprocessing A/B](TDDI_PREPROCESSING_AB_PILOT_RESULTS.md).

### Evidence ranking member 0, task 0–7 — 2026-09-15

Hai trajectory validation-only giữ cố định preprocessing B, folds, seed, quota,
replay và hyperparameter; chỉ thay không gian xếp hạng exemplar. Buffer chưa đầy ở
task 0–3 nên hai run giống nhau. Sau task 4, exemplar bắt đầu khác; Jaccard giữa hai
buffer giảm từ `0,913421` ở task 4 xuống `0,236728` ở task 6.

Tại task 7 `seen_all`, ranking chuẩn hóa từng mẫu đạt Macro-F1 `0,525253` và
Balanced Accuracy `0,440017`; ranking trong không gian input sau scaler đạt lần lượt
`0,504230` và `0,416425`. Trên old classes, chuẩn hóa từng mẫu cũng tốt hơn về
Accuracy (`0,272793` so với `0,229922`), Macro-F1 (`0,495228` so với `0,471755`)
và Balanced Accuracy (`0,382980` so với `0,355739`). Pipeline-input chỉ nhỉnh hơn
nhẹ trên current classes.

Quyết định: **người dùng đã chốt `raw_sample_normalized_class_mean_control_v1` làm
exemplar ranking ngày 2026-09-15**. Đây chỉ là quyết định policy từ validation
member 0; không biến pilot thành kết quả test hoặc official ensemble study. Model
input vẫn dùng preprocessing B `task0_standard_frozen`; “sample-normalized” chỉ mô
tả không gian dùng để chọn exemplar.

## 7. Hyper training tạm giữ — đã chốt làm baseline

| Tham số | Giá trị |
| --- | ---: |
| Optimizer | AdamW |
| Learning rate | 0,001 |
| Weight decay | 0,0001 |
| Microbatch | 64 |
| Gradient accumulation | 16 |
| Effective batch mục tiêu | 1.024 |
| Max epochs | 20/task |
| Early stopping patience | 5 |
| Tiêu chí chọn checkpoint | Validation Macro-F1 |
| Activation | GELU |
| Dropout | 0,2 |
| Input normalization | LayerNorm |
| Focal gamma | 1 |
| Distillation alpha | 1 |
| Distillation temperature | 2 |
| Feature-distillation MSE weight | 0,5 |

Giữ behavior optimizer hiện tại (AdamW tạo mới mỗi task, chưa thêm scheduler).
Đây là hyper của run cũ làm mốc, **không phải toàn bộ recipe chính xác của paper**.
Giữ hyper không có nghĩa quay lại memory/replay draws 6.800/6.800 trong JSON cũ.

## 8. Ensemble và threshold — đã chốt

- Mean probabilities, không mean raw logits.
- Validation ba folds được ghép thành OOF; mỗi mẫu chỉ dùng prediction của member
  không train trên mẫu đó. Không mean ba member trên held-out của một member.
- Chọn riêng theo từng giai đoạn CIL, chỉ dùng class đã thấy.
- Confidence: `1 - entropy / log(C_seen)`; giữ xử lí biên `C_seen <= 1` an toàn.
- Grid cố định 0,50–0,99, bước 0,01.
- Chính: chọn threshold nhỏ nhất đạt OOF selected accuracy >= 95%.
- Không hard-code threshold 0,88 của paper.

Nếu không candidate đạt mục tiêu:

1. Chỉ xét candidate có OOF coverage >= 50%.
2. Chọn accuracy cao nhất; hòa thì coverage cao hơn, rồi threshold thấp hơn.
3. Không candidate đủ điều kiện: không lọc, báo full-set và trạng thái `no_selection`.

Lưu trạng thái `target_met`, `fallback` hoặc `no_selection`. Fallback là policy bổ
sung của study này, không phải quy tắc gốc của paper và không được báo là đã đạt 95%.
Candidate giữ 0 mẫu không được xem là đạt mục tiêu.

Freeze artifact trước khi áp lên ensemble test. Luôn báo full-set metrics, selected
metrics khi có và coverage. OOF một model/mẫu khác ensemble test ba model/mẫu;
không bảo đảm selected test accuracy đạt 95% chỉ vì OOF đạt. UE vẫn được báo cáo
khi không chọn được threshold. Không coi MI/variance từ một prediction OOF là bằng
chứng các member đồng ý; cần ghi giới hạn/không khả dụng rõ ràng.

## 9. Lộ trình thực hiện

1. Ghi decision record này và cập nhật các prompt triển khai cho phù hợp.
2. Triển khai policy mới, provenance, checkpoint/resume và tests; không chỉ sửa JSON.
3. Smoke member 0 task 0–1, khoảng 2–3 epochs/task, kiểm tra kỹ thuật.
4. Hai pilot preprocessing member 0 task 0–1, max 20 epochs/task/patience 5, chạy tuần tự.
5. Đánh giá validation Macro-F1/Balanced Accuracy, old/new class metrics, loss components,
   best epoch, runtime, VRAM và độ ổn định. Chưa cần threshold hoặc ba member để chọn preprocessing.
6. Thử không gian chọn exemplar sau khi có kết quả preprocessing.
7. Khi thiết kế đủ rõ, pilot ba member để kiểm tra đầy đủ OOF/ensemble/threshold.
8. Chỉ cân nhắc full tám task sau khi các kiểm tra đạt.

Không dùng test để chọn hyper hoặc pipeline thắng. Pilot task 0–1 chưa chứng minh
hiệu quả đến task 7. Nếu sát điểm hoặc trade-off chưa rõ, mở rộng task/member trước
khi kết luận. Các đối chứng rộng hơn, nhiều experiment seed và EWC/GEM/A-GEM xét sau.

Các output phải ở namespace mới, tách A/B và tách result cũ; tên/path chính xác sẽ
được ghi trong runbook triển khai, không tự khởi chạy full job.

## 10. Bảng trạng thái cuối

| Hạng mục | Trạng thái |
| --- | --- |
| Partition/P3/seeds | Đã chốt; giữ assignment đã audit |
| Global budget và overlap giữa buffer | Đã chốt |
| Quota q=10 + sqrt frequency | Đã chốt |
| Current/replay/cap/class scheduling | Đã chốt policy; chi tiết rounding/state cần tests |
| Hyper training | Giữ cũ làm mốc |
| OOF threshold và fallback | Đã chốt |
| Preprocessing cuối cùng | **Đã chốt B `task0_standard_frozen`** |
| Không gian exemplar cuối cùng | **Đã chốt `raw_sample_normalized_class_mean_control_v1`** |
| Xếp hạng chung cho pilot A/B | Đã hoàn tất vai trò control; kết quả validation chọn sample-normalized |
| Prediction provenance/OOF/common-test UE | **Prompt 15 đã triển khai; OOF one-prediction metrics được đánh dấu không khả dụng** |
| Code/config hỗ trợ pilot task 0–1 | Prompt 5–17 đã triển khai; chưa chạy pilot ba member trên dataset thật |
| Full 8-task chính thức | Chưa được phép chạy; chờ review bundle pilot và quyết định go/no-go |

Tài liệu liên quan: [Implementation prompts](TDDI_STRATIFIED_3FOLD_IMPLEMENTATION_PROMPTS.md),
[fold artifact và runbook build/audit](TDDI_STRATIFIED_3FOLD_ARTIFACT.md).
