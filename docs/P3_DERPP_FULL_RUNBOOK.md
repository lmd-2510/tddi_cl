# P3 DER++ full run

## Phạm vi

- 3 member (0, 1, 2), **đều dùng P3 tail-to-head**, cùng 8 task và frozen 3-fold assignment.
- Mỗi member có reservoir riêng **27.778 mẫu (4% development set/member)**; không chia 4% cho cả 3 member.
- T-DDI numerical model và task-0 frozen preprocessing giữ như pipeline P3, nhưng classifier có **178 output cố định từ task 0**. Khi đánh giá task trung gian, chỉ cho phép dự đoán các class đã xuất hiện.
- DER++ online gốc về *cơ chế*: đọc mỗi mẫu train hiện tại đúng **một lần mỗi task**, minibatch hiện tại dùng CE; lấy **hai minibatch replay độc lập** từ reservoir, nhánh thứ nhất MSE với logits đã lưu lúc mẫu đi vào buffer, nhánh thứ hai CE với nhãn thật. Sau optimizer step mới xét thêm batch hiện tại vào reservoir bằng reservoir sampling đồng đều.
- Không dùng focal, hybrid loss, feature distillation, post-WA hay class-balanced buffer trong thí nghiệm này.

Hàm loss tại một bước có buffer: `CE(current) + α·MSE(replay logits) + β·CE(replay labels)`. Mẫu hiện tại luôn có CE; ở bước đầu khi buffer rỗng hai nhánh replay bằng 0. `α=0.3`, `β=0.5`, replay batch 64 là **giá trị khởi đầu tham khảo từ cấu hình chính thức của Mammoth cho một benchmark DER++**, không phải bộ siêu tham số tối ưu đã được chứng minh cho DDI. AdamW, learning rate 0.001, weight decay 0.0001 và mạng T-DDI là lựa chọn giữ theo dự án, **không phải tái lập y nguyên optimizer/kiến trúc của paper**. Chạy DER++ online một pass không phải so sánh cô lập loss với hybrid 30 epoch.

Nguồn phương pháp: [DER++ source của Mammoth](https://github.com/aimagelab/mammoth/blob/master/models/derpp.py), [cấu hình DER++](https://github.com/aimagelab/mammoth/blob/master/models/config/derpp.yaml), [online benchmark config](https://github.com/aimagelab/mammoth/blob/master/datasets/configs/seq-cifar10/online.yaml).

## Chạy trên server

Đồng bộ code mới lên server trước. Từ root repo Linux:

```bash
cd "$HOME/DrugDrug/cil-tddi/cil-tddi"
conda activate ai_env
bash scripts/run_p3_derpp_full.sh check
bash scripts/run_p3_derpp_full.sh start
```

`check` chỉ xác thực input, không train. `start` kiểm tra trước rồi chạy `nohup` ở server; đợi thông báo `[STARTED]` và PID rồi có thể đóng SSH/tắt máy cá nhân. Có thể kiểm tra sau bằng:

```bash
bash scripts/run_p3_derpp_full.sh status
bash scripts/run_p3_derpp_full.sh follow
```

`follow` chỉ hiển thị log (dừng theo dõi bằng Ctrl-C không dừng job). Nếu có nhiều fold root, đặt `DERPP_FOLD_ROOT` trỏ chính xác tới thư mục chứa `fold_assignments.parquet` và `fold_manifest.json`; nếu preprocessing ở nơi khác, đặt `DERPP_PREP_ROOT`. Không đổi task file P3 hay config khi đang chạy/resume. Nếu process thất bại giữa task, có thể chạy worker với cùng run tag sau khi xử lý lỗi; checkpoint chỉ được tạo sau mỗi task hoàn chỉnh. Artefact của task dở dang được giữ trong `interrupted_attempts/`, rồi task đó chạy lại.

## Kết quả

`outputs/p3_derpp_full8_mem4_seed0_<UTC-tag>/` chứa:

- `member_0/1/2`: checkpoint từng task, prediction artifact, macro-F1 theo task, audit loss/buffer/exposure, `run_config.json`.
- `offline_evaluation/task_0..7`: OOF/test ensemble và frozen threshold reports.
- `final_results`: bảng metrics, report chuẩn P3, phân tích task 6→7, `DERPP_RESULTS.md`.
- `diagnostics`: hình buffer, exposure, class-wise F1, forgetting và confusion matrix cho member/ensemble ở task 6/7.
- `review/*.zip` và `.zip.sha256`: bản gửi để phân tích, gồm báo cáo/log/CSV/JSON/PNG, **không chứa NPZ hoặc checkpoint**. ZIP được tạo ngay cả khi training lỗi và `experiment_status.txt` ghi rõ trạng thái, vì vậy không coi việc ZIP tồn tại là bằng chứng full run thành công.

Để xác nhận hoàn tất, `bash scripts/run_p3_derpp_full.sh status` phải cho `train_and_offline_exit=0` và `package_exit=0`; kiểm tra cả `full_manifest.json` và `final_results/ensemble3_p3_task_summary.csv`. Tại task 7, báo cáo mặc định dùng offline ensemble trên common test. Test set chỉ dùng báo cáo, không chọn threshold hay checkpoint.
