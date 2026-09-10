# T-DDI Stratified 3-Fold Development Pipeline

## 1. Mục tiêu

Tài liệu này chia việc bổ sung data pipeline mới thành các prompt nhỏ, độc lập và dễ kiểm tra cho study:

```text
replay_distill_fixed_budget_uniform
× tddi_paper_member
× ensemble 3 members
× P3 tail-to-head
× experiment seed 0
```

Data partition mong muốn:

```text
development data = train_extracted.parquet + validation_extracted.parquet

member 0: train B+C, validation A
member 1: train A+C, validation B
member 2: train A+B, validation C

test_extracted.parquet: giữ nguyên cho cả ba member
```

Pipeline mới phải tồn tại song song với pipeline train/validation/test hiện tại. Không xóa hoặc ghi đè dữ liệu, config, checkpoint hay kết quả cũ.

## 1.1. Trạng thái implementation hiện có và nguyên tắc bắt buộc

Repository hiện đã có một implementation chưa được hợp nhất hoàn toàn tại:

```text
src/data/stratified_folds.py
```

Module này đã được `src/training/train_cil.py` import, được
`src/training/tddi_ensemble3_study.py` sử dụng và có test liên quan. Vì vậy,
không được coi đây là file thừa hoặc tạo thêm một hệ thống chia fold song song
mà chưa audit implementation hiện tại.

Trước khi thực hiện bất kỳ prompt triển khai nào trong tài liệu này, bắt buộc
thực hiện Prompt 0 bên dưới. Ưu tiên tái sử dụng hoặc refactor
`src/data/stratified_folds.py`. Chỉ tạo module mới nếu audit chứng minh module
hiện tại không thể mở rộng an toàn và phải ghi rõ kế hoạch migration, import
path cuối cùng và cách loại bỏ chức năng trùng lặp.

Không được để đồng thời hai implementation cùng quyết định fold assignment.
Tại mọi thời điểm phải xác định rõ một source of truth duy nhất cho:

- sample identity;
- fold assignment;
- member-to-validation-fold mapping;
- fold seed và thuật toán stratification;
- provenance/hash dùng khi training và resume.

Lưu ý: tài liệu Markdown chỉ là kế hoạch triển khai, không tự thay đổi behavior
runtime. Tuy nhiên, làm theo các prompt cũ mà bỏ qua code hiện có có thể tạo ra
hai pipeline không tương thích.

## 2. Hai script dữ liệu phải tách riêng

### 2.1. Script tạo fold

File dự kiến:

```text
scripts/build_development_folds.py
```

Script này chỉ tạo fold assignment, không audit toàn bộ pipeline và không train model.

Đầu vào dự kiến:

```bash
python scripts/build_development_folds.py \
  --train train_extracted.parquet \
  --validation validation_extracted.parquet \
  --test test_extracted.parquet \
  --n-splits 3 \
  --fold-seed 0 \
  --outdir study_assets/data_partitions/development_3fold_seed0
```

Artifact dự kiến:

```text
study_assets/data_partitions/development_3fold_seed0/
├── fold_assignments.parquet
└── fold_manifest.json
```

`fold_assignments.parquet` nên chứa:

- `source_split`: `train` hoặc `validation`;
- `source_row_index`: vị trí dòng trong Parquet nguồn;
- `sample_id`: định danh ổn định từ cặp thuốc;
- `raw_class_id`: nhãn class gốc;
- `fold_id`: 0, 1 hoặc 2.

`fold_manifest.json` nên chứa:

- schema và version;
- `fold_seed` và `n_splits`;
- thuật toán chia fold và `shuffle` policy;
- hash và số dòng của train, validation và test nguồn;
- số mẫu tổng cộng và số mẫu mỗi fold;
- phân bố class trong từng fold;
- ánh xạ member sang validation fold;
- SHA256 của assignment artifact;
- timestamp và command tạo artifact.

Script phải ghép train và validation theo thứ tự xác định, stratify theo raw class ID, fail nếu class không đủ mẫu, không đưa test vào development, không ghi đè output đã tồn tại và ghi artifact theo cách atomic.

Không nên tạo sáu file feature Parquet như `member_0_train.parquet`, `member_0_validation.parquet`, v.v. Một sidecar assignment nhỏ giúp tránh nhân bản hàng chục GB dữ liệu.

### 2.2. Script audit fold

File dự kiến:

```text
scripts/audit_development_folds.py
```

Script này chỉ đọc và kiểm tra artifact do script tạo fold sinh ra. Nó không được tự sửa hoặc tạo lại assignment.

Đầu vào dự kiến:

```bash
python scripts/audit_development_folds.py \
  --fold-assignments study_assets/data_partitions/development_3fold_seed0/fold_assignments.parquet \
  --fold-manifest study_assets/data_partitions/development_3fold_seed0/fold_manifest.json \
  --train train_extracted.parquet \
  --validation validation_extracted.parquet \
  --test test_extracted.parquet \
  --outdir study_assets/data_partitions/development_3fold_seed0/audit
```

Artifact audit dự kiến:

```text
fold_audit.json
fold_class_counts.csv
fold_member_views.csv
fold_audit.md
```

Các invariant bắt buộc:

- mỗi development sample xuất hiện đúng một lần;
- không thiếu hoặc thừa dòng;
- chỉ có fold 0, 1 và 2;
- ba fold không chồng nhau;
- hợp ba fold bằng đúng train + validation;
- không có test sample trong development;
- source hash và row count khớp manifest;
- mỗi class có mặt trong cả ba fold;
- phân bố mỗi class giữa các fold cân bằng theo stratification;
- member 0/1/2 có đúng training và validation folds;
- sample order deterministic;
- không có duplicate sample ID gây lỗi prediction export.

Audit phải exit với mã khác 0 nếu invariant bắt buộc bị vi phạm.

## 3. Những script hiện tại có thể tham khảo

- `scripts/inspect_splits.py`: cách đọc schema, class counts và tạo báo cáo audit.
- `scripts/check_leakage.py`: cách xây sample identity và kiểm tra ordered/unordered drug-pair overlap.
- `scripts/preprocess_features.py`: cách scan Parquet theo batch và fit preprocessing chỉ từ training data.
- `scripts/build_cil_tasks.py`: chỉ xây class-to-task protocol; không dùng để chia sample thành folds.

Các script cũ phải được giữ nguyên behavior để tái lập study cũ.

## 4. Lưu ý quan trọng về validation ensemble

Ba member có ba validation set khác nhau:

```text
member 0 validation = A
member 1 validation = B
member 2 validation = C
```

Vì sample IDs khác nhau, không được mean probability của ba validation artifact này. Pipeline vẫn có thể:

- early stopping riêng trên A/B/C;
- ensemble ba member trên test chung;
- tính UE trên test chung;
- ghép A/B/C thành OOF predictions để đánh giá member-level generalization.

OOF predictions không phải prediction của ensemble ba member.

Nếu cần chọn ensemble confidence threshold hoặc fit temperature calibration chỉ bằng validation, study phải dành thêm một common calibration split mà không member nào được train trên đó. Đây là một thiết kế thí nghiệm riêng, không được âm thầm suy ra từ ba held-out folds.

## 5. Lưu ý về preprocessing

Scaler hiện tại được fit trên old train split. Sau khi ghép old train + old validation rồi chia lại A/B/C, scaler cũ có thể đã nhìn thấy mẫu thuộc held-out fold của một member.

Để tránh validation leakage mà vẫn giữ recipe preprocessing hiện tại, nên fit ba scaler riêng:

```text
member 0 scaler: fit trên B+C
member 1 scaler: fit trên A+C
member 2 scaler: fit trên A+B
```

Không scaler nào được fit trên held-out fold hoặc test. Phần này phải được triển khai sau khi build/audit fold đã ổn định.

## 6. Thứ tự triển khai

Thứ tự khuyến nghị:

```text
Prompt 0: audit implementation hiện có
         → Prompt 1 → Prompt 2 → Prompt 3
         → chạy build/audit fold trên máy dữ liệu
         → Prompt 4: decision gate, chưa sửa code training/config
         → người dùng chốt buffer/replay/preprocessing/calibration policy
         → Prompt 5 → ... → Prompt 11
```

Ba prompt đầu chỉ xử lý data partition. Prompt 4 là điểm dừng bắt buộc để đọc số liệu thật và chốt thiết kế. Chưa thay đổi training hoặc tạo config chính thức trước khi fold artifact vượt qua audit và người dùng phê duyệt decision record.

## 7. Các prompt triển khai

### Prompt 0 — Audit implementation stratified 3-fold hiện có

```text
Trước khi triển khai, audit implementation hiện có tại
src/data/stratified_folds.py, src/training/train_cil.py,
src/training/tddi_ensemble3_study.py, config stratified 3-fold hiện tại và
các test liên quan.

Không tạo một hệ thống chia fold song song hoặc module development_folds.py
trùng chức năng. Ưu tiên tái sử dụng/refactor stratified_folds.py. Xác định
chính xác phần nào của kế hoạch đã hoàn thành, phần nào còn thiếu và phần nào
đang khác thiết kế, sau đó chỉ đề xuất triển khai phần thiếu.

Audit tối thiểu phải làm rõ:

- nơi fold assignment được tạo và source of truth hiện tại;
- train + validation được ghép và chia như thế nào;
- member 0/1/2 nhận training/validation folds nào;
- sample identity, duplicate detection và deterministic order;
- fold seed, experiment/member seed và vai trò riêng của từng seed;
- artifact/manifest/hash nào đã có hoặc còn thiếu;
- scaler hiện tại có nguy cơ nhìn thấy held-out fold hay không;
- checkpoint/resume đã validate fold provenance hay chưa;
- validation output là OOF hay ensemble và metric UE nào không có ý nghĩa
  khi mỗi sample chỉ có prediction từ một member;
- config nào hiện đang bật stratified_3fold;
- test nào đang bảo vệ behavior này.

Tạo một audit/checklist ngắn trong docs. Chưa thay đổi config, behavior
training, output hoặc artifact. Dừng lại sau audit để người dùng xác nhận
hướng refactor trước khi chạy Prompt 1.
```

Acceptance criteria:

- không tạo file/module chia fold mới;
- chỉ ra chính xác integration và phần còn thiếu;
- phân biệt OOF validation với common-test ensemble;
- ghi rõ rủi ro preprocessing leakage và provenance;
- đề xuất một source of truth duy nhất, không để hai pipeline cùng tồn tại.

### Prompt 1 — Hoàn thiện fold artifact trên implementation hiện có

```text
Đọc kết quả Prompt 0 và mở rộng/refactor src/data/stratified_folds.py để định
nghĩa schema/version, load/save/validation cho artifact stratified development
3-fold. Không tạo development_folds.py hoặc một implementation song song,
trừ khi audit đã được người dùng phê duyệt migration rõ ràng.

Artifact assignment phải chứa source_split, source_row_index, sample_id,
raw_class_id và fold_id. Manifest phải chứa source hashes, source row counts,
fold seed, n_splits, split strategy, member-to-validation-fold mapping và
assignment SHA256.

Chưa thêm CLI, chưa sửa loader/training. Thêm unit tests cho schema,
round-trip, duplicate identity, invalid fold ID và hash mismatch.
Không thay đổi pipeline cũ.
```

Acceptance criteria:

- artifact có schema/version rõ ràng;
- loader fail rõ khi thiếu field hoặc metadata không khớp;
- round-trip giữ nguyên row order và dtype;
- chưa thay đổi command train hiện tại.

### Prompt 2 — Script tạo folds

```text
Tạo scripts/build_development_folds.py.

Ghép logic train + validation thành development pool nhưng không sao chép
3780 feature columns. Dùng StratifiedKFold theo raw class ID, shuffle=True
và fold seed explicit. Xuất fold_assignments.parquet và fold_manifest.json
theo schema trong src/data/development_folds.py.

Test giữ nguyên và chỉ dùng để kiểm tra overlap. Không đưa test vào fold.
Không ghi đè output tồn tại; dùng atomic writes.

Thêm tests chứng minh cùng seed cho cùng assignment, mỗi sample thuộc đúng
một fold, mỗi class được phân bổ cân bằng và seed khác làm thay đổi assignment.
Không chạy dataset thật.
```

Acceptance criteria:

- cùng inputs + fold seed tạo cùng assignment;
- mỗi development row được gán đúng một fold;
- không có test row trong assignment;
- không tạo bản sao feature Parquet cho từng member.

### Prompt 3 — Script audit độc lập

```text
Tạo scripts/audit_development_folds.py, chỉ audit, không tạo hoặc sửa folds.

Kiểm tra source hash/count, assignment coverage, fold disjointness, class
stratification, member fold mapping, duplicate sample IDs và test leakage.
Xuất fold_audit.json, fold_class_counts.csv, fold_member_views.csv và
fold_audit.md.

Audit phải exit non-zero nếu invariant bắt buộc fail. Thêm synthetic tests.
Không sửa training.
```

Acceptance criteria:

- audit tốt trả exit code 0;
- assignment sai, trùng hoặc thiếu row trả exit code khác 0;
- báo cáo thể hiện rõ train/validation size của từng member;
- chưa thay đổi loader hoặc model.

### Prompt 4 — Decision gate sau audit, chưa triển khai training

```text
Đọc các artifact thật do scripts/build_development_folds.py và
scripts/audit_development_folds.py tạo ra, tối thiểu gồm fold_manifest.json,
fold_audit.json, fold_class_counts.csv và fold_member_views.csv.

Chưa sửa loader, train_cil.py, checkpoint, orchestrator hoặc config training.
Không tự chọn hyperparameter thay người dùng.

Tạo docs/TDDI_STRATIFIED_3FOLD_DECISION_RECORD.md để tổng hợp:

- tổng development samples và số mẫu từng fold;
- training/validation size thực tế của từng member;
- phân bố class và các cảnh báo imbalance/duplicate/leakage;
- exact development fraction được dùng bởi từng member;
- các lựa chọn còn phải chốt: row-level hay group-aware split;
- có hay không common calibration split;
- preprocessing raw/identity hay scaler riêng từng member;
- global buffer fraction, global buffer count và per-member count;
- cách làm tròn budget giữa ba member;
- buffer allocation policy theo class;
- replay fraction, replay draws mỗi epoch và repeat cap;
- current-sample scheduling policy;
- task file P3 được giữ nguyên hay tạo protocol mới;
- output namespace mới.

Với mỗi mục chưa chốt, trình bày 2-3 lựa chọn, trade-off, khuyến nghị và
đánh dấu trạng thái TBD. Không điền số liệu không có trong audit. Kết thúc
bằng một bảng quyết định để người dùng xác nhận trước khi tiếp tục.

Chỉ tạo decision record; dừng lại sau khi báo cáo và chờ người dùng chốt.
```

Acceptance criteria:

- số liệu trong decision record truy ngược được về artifact audit;
- buffer count chưa được chốt trước khi biết `N_development` thật;
- không tạo config training tạm thời chứa giả định chưa được duyệt;
- không sửa behavior của code;
- Prompt 5–11 được xem là blocked cho tới khi decision record được người dùng xác nhận.

### Prompt 5 — Loader fold-aware

```text
Mở rộng src/data/ddi_dataset.py bằng API mới để load development rows từ
hai Parquet nguồn theo fold assignment, role=train/validation,
validation_fold và class_ids.

Không đổi behavior load_split_arrays hiện tại. Bảo toàn deterministic row
order và metadata alignment. Validate source rows/hashes trước khi dùng.
Thêm tests cho member mapping, class filtering, row order và legacy loader.
Chưa sửa train_cil.py.
```

Acceptance criteria:

- `role=train` chỉ trả hai folds training;
- `role=validation` chỉ trả held-out fold;
- class filtering vẫn hoạt động theo P3 task;
- API cũ cho kết quả như trước.

### Prompt 6 — Preprocessing theo member

```text
Thêm pipeline preprocessing fold-aware mà không sửa behavior
scripts/preprocess_features.py cũ.

Fit một scaler riêng cho mỗi member chỉ trên hai training folds của member;
không fit trên held-out fold hoặc test. Lưu fold hash, validation_fold,
rows_fitted và source hashes trong scaler metadata.

Thêm tests chống validation leakage và kiểm tra deterministic artifacts.
Không train model.
```

Acceptance criteria:

- mỗi scaler chỉ dùng rows của hai training folds;
- metadata cho biết chính xác scaler thuộc member/fold nào;
- dùng nhầm scaler phải fail rõ;
- scaler cũ vẫn dùng được cho study cũ.

### Prompt 7 — Tích hợp `train_cil.py`

```text
Thêm optional CLI --development-fold-assignments và --validation-fold vào
src/training/train_cil.py.

Khi bật fold mode, current training samples và replay-buffer exemplars chỉ
đến từ hai training folds; early stopping dùng held-out fold. Test giữ
nguyên. Khi không bật fold mode, command cũ phải giữ nguyên behavior.

Ghi partition mode, fold hash, validation fold và source hashes vào
run_config/training audit. Thêm synthetic two-task tests. Không chạy data thật.
```

Acceptance criteria:

- current samples và exemplars không chứa held-out fold;
- test chưa bao giờ được dùng để train hoặc early stopping;
- legacy CLI không đổi;
- run config đủ provenance để tái lập partition.

### Prompt 8 — Resume provenance

```text
Mở rộng riêng replay task-boundary checkpoint để lưu và validate fold
assignment SHA256, validation_fold, partition schema và source hashes.

Resume bằng fold/config khác phải fail rõ. EWC checkpoint và replay legacy
không được thay đổi behavior. Thêm round-trip synthetic tests.
```

Acceptance criteria:

- resume tiếp tục đúng member, fold và next task;
- checkpoint của member/fold khác bị từ chối;
- continuous run và save/resume tương đương trong tolerance;
- không thay đổi EWC resume.

### Prompt 9 — Orchestrator và config

```text
Mở rộng tddi_ensemble3_study.py cho stratified 3-fold development mode:

member 0 train B+C, validate A;
member 1 train A+C, validate B;
member 2 train A+B, validate C.

Tạo config P3 mới và output namespace mới. Manifest lưu fold provenance.
Dry-run phải hiển thị chính xác fold của từng member. Giữ config và study
P3 cũ hoạt động. Không chạy training.
```

Acceptance criteria:

- dry-run tạo đúng ba command và đúng validation fold;
- member chạy tuần tự;
- complete member được skip, incomplete member được resume an toàn;
- task file P3 và class order không thay đổi;
- output mới không đè study P3 cũ.

### Prompt 10 — Ensemble/export policy

```text
Điều chỉnh study orchestration để validation artifacts của ba held-out
fold không bị đưa nhầm vào offline ensemble.

Cho phép offline ensemble ba member trên common test artifacts. Xuất OOF
validation report riêng và ghi rõ OOF không phải ensemble. Fail rõ nếu cố
aggregate A/B/C như cùng validation set.

Giữ offline ensemble và artifact cũ tương thích. Không train model.
```

Acceptance criteria:

- test artifacts của ba member có cùng sample IDs và ensemble được;
- A/B/C validation artifacts không bị mean sai;
- OOF report ghi rõ semantics;
- artifact pilot cũ vẫn load được.

### Prompt 11 — Runbook và integration tests

```text
Tạo runbook cho P3 stratified-3fold study: build folds, audit, fit
member-specific preprocessing, dry-run, train member 0/1/2 tuần tự,
resume, test ensemble và UE audit.

Thêm synthetic integration test từ fold creation đến orchestrator command.
Không chạy full dataset và không xóa artifact cũ.
```

Acceptance criteria:

- runbook có command chính xác cho máy GPU;
- có preflight, PID/log/VRAM monitoring và resume command;
- chỉ ensemble sau khi đủ ba member;
- synthetic pipeline chạy hoàn chỉnh mà không cần dataset thật.

## 8. Những thứ không được xóa

Không xóa:

- `train_extracted.parquet`;
- `validation_extracted.parquet`;
- `test_extracted.parquet`;
- scripts audit/preprocessing cũ;
- config P3 hiện tại;
- output, checkpoint và prediction artifacts của study trước;
- P3 task file hiện tại.

Các file này là dữ liệu nguồn và baseline cần thiết để so sánh study cũ với study 3-fold mới.

## 9. Điểm dừng bắt buộc sau Prompt 3

Sau khi hoàn thành Prompt 1–3, chưa nên tích hợp training ngay. Trước hết cần chạy hai script trên máy chứa dữ liệu và kiểm tra:

- fold audit đạt toàn bộ invariant;
- kích thước ba fold hợp lý;
- mỗi class xuất hiện trong cả ba fold;
- không có test leakage;
- assignment có thể tái tạo bằng cùng seed;
- artifact không chiếm dung lượng bất hợp lý.

Sau khi các điều kiện trên đạt yêu cầu, chỉ chạy Prompt 4 để tạo decision record. Chưa chạy Prompt 5 trở đi cho đến khi người dùng chốt rõ buffer, replay, preprocessing và validation/calibration policy.
