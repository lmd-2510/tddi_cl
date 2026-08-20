# Định hướng khóa: T-DDI Protocol Study

## Câu hỏi duy nhất ở giai đoạn hiện tại

Với cùng dữ liệu, cùng T-DDI backbone, cùng cơ chế replay–distillation và cùng ngân sách huấn luyện, **cách phân các lớp DDI mất cân bằng vào chuỗi task ảnh hưởng thế nào tới hiệu năng cuối, cân bằng lớp và forgetting?**

Đây là study về protocol, không phải study về method hoặc backbone. Chỉ thay `task-file`; mọi thành phần còn lại phải giữ nguyên theo `configs/protocol_study_tddi.json`.

## Vai trò của từng protocol

| ID | Tên | Vai trò |
|---|---|---|
| P0 | Random-CIL | Reference ngẫu nhiên chuẩn, có 5 lịch tương ứng seed 0–4. |
| P1 | Frequency-balanced | Baseline xác định, rải head/medium/tail bằng round-robin theo ba tầng tần suất. |
| P2 | Head→tail | Stress test: lớp phổ biến đến trước, lớp hiếm đến sau. |
| P3 | Tail→head | Stress test đối chứng với P2: lớp hiếm đến trước, lớp phổ biến đến sau. |
| P4 | Constrained mass-balanced | Protocol chính: khóa quota bốn nhóm hiếm và cân bằng tổng số mẫu giữa task. |
| P5 | Multi-factor balanced | Cân bằng thêm effective support, số drug duy nhất và độ đa dạng descriptor. |
| P6 | Difficulty balanced | Cân bằng khối lượng và độ khó dự đoán từ static T-DDI trên validation. |
| P7 | Confusion spread | Phân tán các cặp lớp hay nhầm sang các task khác nhau. |
| P8 | Controlled rarity drift | Tạo dịch chuyển độ hiếm có kiểm soát, giảm dần head và tăng dần ultra-tail. |

## Quyết định protocol chính

**P4 là protocol chính hiện tại.** Lý do là P4:

- trực tiếp xử lý mất cân bằng nhưng không dùng tín hiệu học từ model;
- giữ quota rarity trong từng task và tối ưu sample mass;
- có 5 lịch seed để đo độ nhạy của cách chia;
- dễ giải thích, dễ tái lập và không phụ thuộc checkpoint tham chiếu;
- phù hợp để sau này so method/backbone mà không tạo lợi thế cho riêng T-DDI.

P6 có macro-F1 trung bình cao nhất trong nhóm P4–P8, nhưng nó dùng độ khó do static T-DDI sinh trên validation. Vì vậy P6 là protocol nâng cao/model-informed, không phải protocol chính trung lập. P2 và P3 không dùng để chọn “winner” vì chúng cố tình tạo hai chế độ stream khác nhau và dẫn tới trade-off metric rất lớn.

## Tiêu chí đánh giá

Không chọn protocol chỉ bằng accuracy. Thứ tự xem xét:

1. final macro-F1 trên toàn bộ 178 lớp đã thấy;
2. final balanced accuracy;
3. average forgetting, càng thấp càng tốt;
4. độ ổn định giữa 5 seed;
5. tính trung lập, khả năng diễn giải và không dùng test để thiết kế lịch.

Test chỉ được dùng để báo cáo sau khi checkpoint/epoch được chọn bằng validation. P5–P8 chỉ dùng thống kê train và tín hiệu validation từ static T-DDI; tuyệt đối không dùng test để xây lịch.

## Quy tắc thay đổi

Một thay đổi chỉ còn thuộc study này nếu nó sửa lỗi mà không làm đổi semantics đã khóa. Nếu cần thay backbone, method, loss, replay budget, optimizer, split hay số task, hãy tạo study ID/config/runner/output root mới. Không sửa kết quả P0–P8 cũ rồi so chung dưới cùng nhãn.
