# S03 Calibration Results

## Phạm vi và protocol

S03 dùng đủ 20 clean S02 runs (`4 methods × 5 seeds × 8 tasks`). Một scalar
temperature được fit riêng cho mỗi run/task bằng validation NLL. Cùng temperature
được áp dụng lên test để báo cáo; test không tham gia fit hoặc lựa chọn tham số.

O05 có 640 unique rows và schema v1. Tất cả 160 fits hội tụ, không fit nào chạm cận
temperature, accuracy không đổi sau scaling và validation NLL không tăng.

## Final-task test calibration

Các giá trị dưới đây là mean trên 5 seeds. Temperature ở hàng raw bằng 1.

| Method | Stage | Temperature | ECE | Brier | NLL | Mean confidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `joint_seen` | raw | 1.0000 | 0.0058 | 0.1070 | 0.2213 | 0.9280 |
| `joint_seen` | scaled | 1.0241 | 0.0038 | 0.1069 | 0.2204 | 0.9262 |
| `replay` | raw | 1.0000 | 0.3755 | 1.1285 | 8.6930 | 0.5637 |
| `replay` | scaled | 4.8608 | 0.1093 | 0.9590 | 4.3948 | 0.0797 |
| `replay_distill` | raw | 1.0000 | 0.2562 | 1.0180 | 6.0755 | 0.4804 |
| `replay_distill` | scaled | 3.2380 | 0.1253 | 0.9516 | 4.2115 | 0.0990 |
| `sequential` | raw | 1.0000 | 0.6279 | 1.4133 | 14.2673 | 0.7587 |
| `sequential` | scaled | 11.1884 | 0.0992 | 0.9842 | 4.9614 | 0.0327 |

## Nhận xét

1. `joint_seen` vốn đã calibrated tốt; temperature gần 1 và thay đổi metric rất nhỏ.
2. Replay và sequential models bị overconfidence nặng. Temperature lớn làm giảm mạnh
   NLL và ECE nhưng cũng làm mean confidence giảm sâu.
3. Ở final task, NLL giảm trên 5/5 seeds cho cả bốn methods. ECE giảm trên 5/5 replay,
   5/5 sequential, 3/5 replay-distill và 2/5 joint-seen seeds. Điều này phù hợp vì
   temperature được tối ưu theo NLL, không trực tiếp tối ưu ECE.
4. Trên toàn bộ 160 test run/tasks, ECE giảm ở 139, Brier giảm ở 138 và NLL giảm ở
   157 trường hợp.
5. Sau scaling, nhiều replay/sequential task không còn mẫu confidence ≥ 0.9. Khi đó
   high-confidence error rate không xác định và O05 ghi ô rỗng cùng support bằng 0;
   không diễn giải ô rỗng là error rate bằng 0.

## Kết luận S03

S03 hoàn thành về kỹ thuật và thực nghiệm. Raw confidence của replay-family và
sequential không đủ tin cậy để dùng trực tiếp cho uncertainty-aware decisions.
Các bước E03 hoặc replay allocation dùng uncertainty phải lấy calibrated probability
và giữ provenance của temperature theo đúng run/task.
