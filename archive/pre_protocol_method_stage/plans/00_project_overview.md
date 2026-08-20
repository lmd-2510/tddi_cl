# DDI2025-CIL Project Overview

## Project Name

**DDI2025-CIL: Class-Incremental Learning for Drug-Drug Interaction Event Prediction Using Chemical Descriptor Matrices**

## Project Goal

Xây dựng một benchmark **class-incremental learning (CIL)** cho bài toán phân loại sự kiện tương tác thuốc từ ma trận descriptor hóa học đã trích xuất sẵn, theo hướng:

```text
drug A + drug B -> descriptor vector -> DDI class
```

Giai đoạn hiện tại chỉ lập kế hoạch, protocol, checklist, artifact spec, và execution roadmap. Không chạy training thật và không sửa dữ liệu gốc.

## Dataset Summary

Các input chính:

| Split | File Parquet | Rows | Cols |
|---|---|---:|---:|
| train | `/mnt/data/uyen/data_splits/train_extracted.parquet` | 520841 | 3787 |
| validation | `/mnt/data/uyen/data_splits/validation_extracted.parquet` | 173614 | 3787 |
| test | `/mnt/data/uyen/data_splits/test_extracted.parquet` | 173614 | 3787 |

Đã xác nhận trực tiếp từ Parquet metadata:

- `class` ở vị trí cột 3248 nếu đếm từ 1
- `drugname-drug_a` ở vị trí 798
- `drugname-drug_b` ở vị trí 799
- `drugsmiles-drug_a` ở vị trí 2835
- `drugsmiles-drug_b` ở vị trí 2836
- các cột meta text nằm xen giữa bảng, không nằm gọn ở đầu/cuối

## Why This Is Not a Genomics/Fragmentomics Raw Dataset

Repo này đang chứa dữ liệu **DDI descriptor tabular classification**, không phải dữ liệu sequencing, fragmentomics, hay genomics raw input. Do đó:

- input chính là vector descriptor hóa học đã có sẵn
- không có bước alignment, variant calling, genome coordinate normalization, hay fragment-end feature extraction trong phạm vi baseline đầu tiên
- SMILES chỉ giữ làm metadata và kiểm tra leakage/overlap, không dùng làm encoder đầu tiên

## Why This Fits Class-Incremental Learning

Dataset phù hợp với CIL vì:

- là bài toán multi-class với **178 classes**
- phân bố lớp long-tail mạnh, phù hợp để nghiên cứu forgetting và rare-class degradation
- có thể dựng protocol tăng dần số lớp mà vẫn giữ nguyên split train/validation/test gốc
- descriptor matrix tabular cho phép cô lập ảnh hưởng của protocol continual learning trước khi thêm encoder phức tạp hơn

## Planned Main Contributions

1. Xây dựng benchmark **DDI2025-CIL** từ dữ liệu DDI descriptor fixed-split.
2. Đo **catastrophic forgetting** trong bài toán multi-class DDI event prediction.
3. So sánh các baseline:
   - sequential fine-tuning
   - replay
   - replay + distillation
   - EWC hoặc SI
   - joint training upper bound
4. Phân tích:
   - long-tail class distribution
   - rare-class forgetting
   - calibration drift theo task

## Main Problem Definition

- **Input X**: toàn bộ descriptor numeric sau khi loại explicit các cột meta và nhãn
- **Target y**: cột `class`
- **Main benchmark**: 8-task class-incremental learning trên 178 classes
- **Main model family**: descriptor-only MLP

## Core Constraints

- không dùng `class` trong feature input
- không dùng `drugid-*`, `drugname-*`, `drugsmiles-*` làm feature
- không giả định “lấy mọi cột numeric là đủ an toàn”
- không dùng `awk -F,` để kết luận CSV hỏng hoặc suy luận schema
- ưu tiên Parquet cho mọi bước audit/preprocess/train
- không dùng test set cho hyperparameter tuning

## Non-Goals for Phase 1

- không recompute descriptor từ SMILES
- không dùng SMILES encoder trong baseline chính
- không đưa DDI2018 -> DDI2025 thành protocol chính ở vòng đầu
- không làm temporal continual learning claim nếu chưa có timestamp hoặc version ordering
- không chạy full experiment trong bước tạo plan

## Expected Repo Direction

Kế hoạch này giả định workspace sẽ được mở rộng dần thành cấu trúc như:

```text
ddi2025-cil/
  configs/
  plans/
  scripts/
  src/
  outputs/
  reports/
```

Hiện tại workspace mới có dữ liệu split; các thư mục còn lại sẽ được tạo sau theo roadmap.

## Primary Risks to Address Early

- label leakage vì `class` nằm giữa bảng
- meta text leakage vì các cột text xen giữa descriptor matrix
- class imbalance và long-tail mạnh
- pair overlap hoặc reverse-pair overlap giữa các split
- chi phí compute/RAM do file lớn

## Acceptance Criteria

- [ ] Trả lời rõ dữ liệu là gì và mỗi row đại diện cho gì
- [ ] Trả lời rõ `X` là gì và `y` là gì
- [ ] Nêu rõ cột nào phải loại khỏi `X`
- [ ] Nêu rõ vì sao `class` có nguy cơ leakage
- [ ] Nêu rõ vì sao Parquet là input chuẩn cho pipeline
- [ ] Định nghĩa rõ main benchmark và non-goals
- [ ] Không nhầm dataset này với genomics/fragmentomics raw workflow

## Definition of Done

- [ ] Có đủ 13 file plan trong `plans/`
- [ ] Mỗi file có mục tiêu, input/output, checklist, và acceptance criteria
- [ ] Protocol CIL đủ chi tiết để người khác có thể triển khai code mà không phải hỏi lại logic chính
