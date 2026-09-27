# DER++ P3: pilot buffer chia theo lớp (member 0)

Pilot này chạy **member 0 qua đủ task 0–7**. Nó không chạy member 1/2 và không có offline ensemble. Kết quả chính là test/validation Macro-F1 của member 0; ZIP tổng kết có log, audit, CSV và hình, không chứa NPZ/checkpoint.

## Thay đổi duy nhất so với DER++ vừa chạy

- Giữ P3 tail-to-head, frozen fold/preprocessing, seed, T-DDI fixed 178-class head, optimizer, learning rate, batch, **một pass/task**, `CE(current) + 0.3·MSE(stored logits) + 0.5·CE(replay label)` và **hai lượt rút replay đều trên các slot được giữ**.
- Giữ budget **27.778 slot, tức 4% development data cho member 0**.
- Thay global uniform reservoir bằng quota cân bằng theo lớp đã thấy. Quota dạng water-fill: chia đều tối đa có thể, lớp ít mẫu giữ tối đa mẫu train của chính lớp đó, chỗ dư phân bổ cho lớp nhiều mẫu. Trong từng lớp, chọn bằng reservoir ngẫu nhiên, mỗi hàng train hiện tại đi qua đúng một lần; giữ logits tại lần đầu gặp mẫu, không cập nhật lại logits sau này.
- Quota tại đầu mỗi task dựa vào số train đã thấy trước đó và **số train của task hiện tại**. Không dùng sample của task sau, validation hay test để chọn quota/exemplar. Nếu quota lớp cũ giảm, các exemplar cũ được rút bớt ngẫu nhiên; không xóa toàn bộ lớp cũ để nhường cho task mới.

Đây là **DER++ với buffer thích nghi**, không gọi là DER++ gốc. Không thay đồng thời cách rút replay: khi buffer đã cân bằng, ta mới đo xem riêng cách lưu mẫu có cải thiện Macro-F1 hay không.

## Chạy trên server

Sau khi đồng bộ code mới lên server:

```bash
cd "$HOME/DrugDrug/cil-tddi/cil-tddi"
conda activate ai_env
bash scripts/run_p3_derpp_equal_buffer_pilot.sh check
bash scripts/run_p3_derpp_equal_buffer_pilot.sh start
```

`check` không train; `start` chạy bằng `nohup`. Chỉ tắt SSH/máy cá nhân sau dòng `[STARTED]` có PID. Xem sau bằng:

```bash
bash scripts/run_p3_derpp_equal_buffer_pilot.sh status
bash scripts/run_p3_derpp_equal_buffer_pilot.sh follow
```

Ctrl-C trong `follow` chỉ dừng xem log, không dừng job. Nếu fold root mặc định không còn đúng, đặt `DERPP_PILOT_FOLD_ROOT` trỏ tới thư mục `folds` chứa `fold_assignments.parquet` và `fold_manifest.json`. Preprocessing mặc định ở `study_assets/preprocessing_p3_seed0_fold42`; có thể ghi đè bằng `DERPP_PILOT_PREP_ROOT`.

## Xem kết quả

Output: `outputs/p3_derpp_equal_buffer_member0_<UTC-tag>/`.

- `member_0/metrics.csv`: Macro-F1 validation/test tất cả task; `task_7/metrics.csv` có seen-all, old và current.
- `member_0/task_*/buffer_audit.json`, `replay_exposure.csv`, `training_audit.csv`: mỗi lớp được giữ bao nhiêu, đã rút replay bao nhiêu, loss và optimizer steps. `target_quotas` ghi quota được cấp theo lớp.
- `pilot_results/PILOT_RESULTS.md`, `buffer_by_source_task.csv`, `task7_old_current_metrics.csv`: tổng kết task 6→7 và số lớp mất exemplar theo task nguồn.
- `visualizations_member0/task_6_test`, `task_7_test` và hai thư mục validation tương ứng: mỗi thư mục có 5 PNG (buffer, exposure, per-class F1, forgetting, confusion matrix) cùng CSV. Heatmap `04` luôn chứa đủ trajectory 8 task, kể cả khi nằm trong thư mục hình task 6.
- `review/p3_derpp_equal_buffer_member0_<UTC-tag>.zip` và `.zip.sha256`: bản tải về gửi để phân tích. ZIP vẫn có thể được tạo khi training/hình lỗi; **chỉ coi run hoàn tất khi `status` ghi `train_and_visualizations_exit=0` và `package_exit=0`**.

So sánh công bằng với run DER++ gốc ở **member 0**, đặc biệt task-7 Macro-F1 (mốc trước: khoảng `0.309711`), task 6→7 drop, old/current F1, số lớp không có exemplar và replay draws. Không so trực tiếp một member với con số offline ensemble ba member `0.342927` để kết luận tác động của buffer.
