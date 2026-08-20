# Cấu hình và mô tả thử nghiệm

Nguồn sự thật dạng máy là `configs/protocol_study_tddi.json`. Tài liệu này diễn giải cấu hình đó; nếu hai nơi khác nhau thì dừng chạy và sửa cho đồng bộ trước.

## Dữ liệu

| Thành phần | Giá trị |
|---|---:|
| Tổng số cặp DDI | 868,069 |
| Train | 520,841 (60%) |
| Validation | 173,614 (20%) |
| Test | 173,614 (20%) |
| Số lớp | 178 |
| Số feature số | 3,780 |
| Task | 8 |
| Số lớp/task | 38 ở task 0; 20 ở mỗi task 1–7 |

Ba split đã được audit: không có trùng cặp có thứ tự, không thứ tự hoặc cặp đảo chiều giữa split; không có duplicate pair. Drug riêng lẻ có thể xuất hiện ở nhiều split, nhưng một cặp drug không bị lặp. Feature dùng để train không chứa nhãn hoặc metadata ID.

## Preprocessing

- Dùng đúng danh sách 3,780 feature tại `outputs/audit/feature_columns.json`.
- Giá trị thiếu/vô hạn được thay bằng `0` theo pipeline đã audit.
- `StandardScaler` chỉ fit trên train, lưu tại `outputs/preprocess/scaler.pkl`, sau đó áp dụng nguyên trạng cho validation/test.
- Không PCA, không feature selection, không row cap.
- Task schedule được xây theo raw class ID thật, không giả định class ID là `range(178)`.

## Model T-DDI cố định

| Hạng mục | Giá trị |
|---|---|
| Variant | `tddi` |
| Input | 3,780 |
| Hidden layers | 1,024; 512 |
| Head | Linear, mở rộng theo toàn bộ lớp đã thấy |
| Normalization | LayerNorm sau mỗi hidden linear |
| Activation | GELU |
| Dropout | 0.2 |

Trong code, `tddi` và legacy alias `base` có cùng hidden dimensions, nhưng study chỉ ghi và chạy tên `tddi` để tránh nhập nhằng.

## Huấn luyện cố định

| Hạng mục | Giá trị |
|---|---|
| Method | `replay_distill_fixed_budget_uniform` |
| Batch size | 1,024 |
| Epoch tối đa | 20 mỗi task |
| Early stopping | patience 5 |
| Selection metric | validation macro-F1 trên mọi lớp đã thấy |
| Optimizer | AdamW |
| Learning rate | 0.001 |
| Weight decay | 0.0001 |
| Classification loss | Focal loss, gamma 1.0 |
| Distillation | alpha 1.0, temperature 2.0 |
| Feature distillation | MSE weight 0.5 |
| Training seeds | 0, 1, 2, 3, 4 |
| Device | `auto` (CUDA nếu có, nếu không CPU) |
| Determinism | global seed được khóa theo training seed |

Các argument legacy vẫn còn trong engine để tái lập lịch sử nhưng không hoạt động trong study này: `memory_per_class=50` bị bỏ qua bởi fixed-total-budget, `ewc_lambda=1000` bị bỏ qua vì không chạy EWC, và `export_s02=false`.

### Replay cố định

- Tổng memory budget luôn là 6,800 exemplar, không tăng theo số lớp.
- Buffer chỉ nhận mẫu từ train của current task.
- Exemplar được xếp theo khoảng cách tới class mean với stable-index tie-break.
- Dung lượng phân bổ theo max-min có ràng buộc capacity, tie-break raw class ID.
- Từ task 1 trở đi, mỗi epoch đi qua current-task samples đúng một lần và lấy đúng 6,800 lượt replay cũ.
- Replay chọn class cũ đều nhau; trong class lấy không hoàn lại theo chu kỳ.
- Teacher là snapshot model của task trước; distillation căn đúng cột logits của old classes khi head mở rộng.

## Chi tiết protocol

Các rarity bin đều tính từ số mẫu train: `ultra_tail ≤ 20`, `tail = 21..100`, `medium = 101..1000`, `head > 1000`.

### P0 — Random-CIL

Shuffle toàn bộ class ID bằng NumPy RNG theo schedule seed rồi cắt theo `[38, 20×7]`. Có năm task file seed 0–4; schedule seed và training seed tương ứng nhau.

### P1 — Frequency-balanced

Sắp lớp giảm dần theo train count, chia thành ba stratum gần bằng nhau, rồi round-robin vào các task còn capacity. Đây là một lịch xác định; cùng task file được dùng cho cả 5 training seed.

### P2 — Head→tail

Sắp lớp theo train count giảm dần, class ID tăng dần để phá hòa, rồi cắt tuần tự. Task đầu giàu lớp head, task sau ngày càng hiếm. Cùng một task file cho 5 training seed.

### P3 — Tail→head

Đối xứng P2: train count tăng dần rồi class ID tăng dần. Lớp hiếm xuất hiện trước và lớp head xuất hiện sau. Cùng một task file cho 5 training seed.

### P4 — Constrained mass-balanced

Phân bổ quota của bốn rarity bin theo tỷ lệ vào từng task. Trong mỗi seed, greedy assignment giảm lệch tổng train samples, sau đó hoán đổi class trong cùng bin tối đa 1,000 vòng để cải thiện cân bằng mass mà không phá quota. Có 5 lịch seed 0–4.

### P5 — Multi-factor balanced

Giữ quota rarity như P4, đồng thời tối ưu cân bằng train count, `sqrt(class support)`, tổng số drug duy nhất và descriptor diversity. Tín hiệu lấy từ train. Hoán đổi class chỉ trong cùng rarity bin, tối đa 3,000 vòng. Có 5 lịch.

### P6 — Difficulty balanced

Giữ quota rarity và cân bằng train mass, đồng thời rải đều `difficulty = 1 - validation F1` và validation mean NLL lấy từ static T-DDI seed 0. Checkpoint tĩnh được train trên train, chọn epoch bằng validation; test không tham gia tạo lịch. Hoán đổi tối đa 3,000 vòng. Có 5 lịch.

### P7 — Confusion spread

Giữ quota rarity nhưng phạt việc đặt các lớp có symmetric confusion rate cao vào cùng task. Confusion graph lấy từ dự đoán validation của static T-DDI seed 0; test không được dùng. Initialization có trọng số confusion và tối ưu 1,500 vòng. Có 5 lịch.

### P8 — Controlled rarity drift

Khóa quota thủ công để stream khó dần: số lớp `head` qua 8 task là `[8,6,5,5,4,4,4,3]`, còn `ultra_tail` là `[8,4,4,5,5,6,6,7]`; các quota tail/medium giữ tổng đúng 178 lớp và task capacity. Sau đó cân bằng train mass bằng hoán đổi trong bin tối đa 3,000 vòng. Có 5 lịch.

## Static reference cho P6/P7

- Backbone T-DDI, input 3,780, hidden `[1024, 512]`.
- Batch 1,024; 20 epoch; AdamW `lr=0.001`, `weight_decay=0.0001`.
- Seed 0; CrossEntropy; chọn checkpoint bằng validation; best epoch 19 trong artifact hiện tại.
- Static reference chỉ tạo difficulty/confusion signal. Nó không thay thế CIL run và không dùng test để xây task.

## Đánh giá và artifact

Mỗi task đánh giá trên mọi lớp đã thấy. Báo cáo ít nhất final macro-F1, balanced accuracy, accuracy, weighted-F1 và average forgetting. Mỗi run phải có `run_summary.md`, `metrics.csv`, `forgetting.csv` mới được runner coi là hoàn tất. Kết quả nằm tại:

```text
outputs/runs_backbones/<protocol>_seed<seed>_replay_distill_fixed_budget_uniform_mlptddi/
```
