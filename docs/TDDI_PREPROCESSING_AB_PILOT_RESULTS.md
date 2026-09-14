# Kết quả pilot preprocessing A/B — member 0, P3 task 0–1

Ngày đánh giá: 2026-09-14. Nguồn: gói `review.tar.gz` do người dùng xuất từ
server GPU. Đây là đánh giá **validation-only**; không dùng test, threshold hoặc
ensemble để chọn preprocessing.

## 1. Câu hỏi của pilot

- **A — `raw_identity`:** descriptor gốc đi thẳng vào input LayerNorm của model.
- **B — `task0_standard_frozen`:** fit StandardScaler chỉ bằng training rows của
  task 0/member 0, rồi đóng băng và dùng lại cho các task sau.

Hai case dùng cùng member 0, P3 tail-to-head, task 0–1, seed, model, hyperparameter,
buffer, exemplar ranking và replay policy. Mục tiêu là chỉ so tác dụng của
preprocessing.

## 2. Tính hợp lệ của phép so sánh

Comparator trong repo đã đọc lại bundle và kết thúc thành công. Kết quả alignment
là `PASS`; trường `winner` vẫn để trống đúng thiết kế.

| Kiểm tra | Kết quả |
| --- | --- |
| Assignment SHA256 | Cùng `c84dc034...de79a43` |
| Task-file SHA256 | Cùng `0d64c465...a26e79` |
| Experiment/member | Cùng experiment seed 0, member 0, member seed 409845317 |
| Current sample IDs | Giống nhau ở task 0 và task 1 |
| Validation sample IDs | Giống nhau ở task 0 và task 1 |
| Replay IDs trước task | Giống nhau |
| Retained exemplar IDs sau task | Giống nhau: 331 rồi 760 slots |
| Sampler order | Giống nhau ở mọi epoch chung |
| Replay task 1 | Fraction thực tế 12,449%; class coverage 100%; max repeat 1 |

Vì bundle là bản report-only, lần đọc lại này kiểm tra JSON audit và source hashes,
không đọc lại dataset hàng GB và không kiểm tra nội dung tensor trong checkpoint.
Các kích thước checkpoint dưới đây đến từ telemetry đã ghi lúc chạy trên server.

## 3. Kết quả validation

`seen_all` ở task 1 nghĩa là model sau khi học task 1 được đánh giá một lần trên
toàn bộ 58 class đã thấy từ task 0–1. Đây không phải trung bình metric qua hai task.

| Case | Task | Nhóm | N | Accuracy | Macro-F1 | Balanced Accuracy |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| A | 0 | seen_all/current | 162 | 0,919753 | **0,914008** | **0,921460** |
| B | 0 | seen_all/current | 162 | **0,925926** | 0,908564 | 0,908521 |
| A | 1 | seen_all | 378 | 0,870370 | 0,838958 | 0,830732 |
| B | 1 | seen_all | 378 | **0,880952** | **0,859409** | **0,853691** |
| A | 1 | old — class task 0 | 162 | 0,796296 | 0,829479 | 0,785276 |
| B | 1 | old — class task 0 | 162 | **0,839506** | **0,863828** | **0,827444** |
| A | 1 | current — class task 1 | 216 | **0,925926** | **0,919381** | **0,917100** |
| B | 1 | current — class task 1 | 216 | 0,912037 | 0,918035 | 0,903560 |

Delta B − A tại task 1 `seen_all`:

- Accuracy: **+0,010582**.
- Macro-F1: **+0,020450**.
- Balanced Accuracy: **+0,022958**.

Ý nghĩa của trade-off:

- B giữ các class cũ tốt hơn: old Macro-F1 tăng khoảng 3,43 điểm phần trăm và old
  Balanced Accuracy tăng khoảng 4,22 điểm phần trăm.
- A nhỉnh hơn trên class mới: current Macro-F1 chỉ hơn khoảng 0,13 điểm phần trăm,
  nhưng current Balanced Accuracy hơn khoảng 1,35 điểm phần trăm.
- Nhờ cải thiện class cũ lớn hơn phần giảm ở class mới, B có kết quả `seen_all`
  tốt hơn. Với CIL, đây là tín hiệu có lợi vì giữ kiến thức cũ là mục tiêu chính.

## 4. Hội tụ, loss và tài nguyên

| Case | Task | Best epoch | Epoch đã chạy | Runtime | Peak allocated | Peak reserved |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A | 0 | 15 | 20 | 192,95 s | 1,62 GiB | 1,74 GiB |
| B | 0 | 10 | 15 | 193,60 s | 1,62 GiB | 1,74 GiB |
| A | 1 | **20** | 20 | 198,62 s | 1,95 GiB | 2,08 GiB |
| B | 1 | 13 | 18 | 193,44 s | 1,95 GiB | 2,08 GiB |

- B đạt vùng điểm tốt sớm hơn và dừng sớm; tài nguyên GPU gần như giống A.
- A đạt best task 1 đúng epoch tối đa 20. Vì vậy chưa thể khẳng định A đã hội tụ;
  giới hạn epoch có thể đang bất lợi cho A.
- Validation loss task 1 tại best checkpoint của A là 0,429222, thấp hơn B là
  0,637436, dù B có Macro-F1/Balanced Accuracy cao hơn. Điều này không mâu thuẫn:
  cross-entropy đo độ tin cậy xác suất, còn hai metric macro đo độ đúng cân bằng giữa
  class. Checkpoint của pilot được chọn theo validation Macro-F1.
- Ở B, feature-distillation raw loss task 1 khoảng 3,7–4,8, cao hơn A khoảng
  0,5–0,7; logit-distillation cũng tăng mạnh ở vài epoch đầu rồi giảm. Các giá trị
  đều hữu hạn và metric hội tụ, nhưng scaler đã làm thay đổi thang latent nên trọng
  số feature distillation 0,5 có hiệu lực tương đối khác giữa hai pipeline. Cần ghi
  nhận điểm này khi giải thích kết quả.
- P3 bắt đầu từ tail: task 0 chỉ có 331 training rows, task 1 có 429 current rows và
  khoảng 61 replay draws mỗi epoch. Với effective batch 1.024, mỗi epoch chỉ có một
  optimizer step. Do đó đường học và metric trên 162/378 validation rows còn nhạy
  với nhiễu; pilot này là bằng chứng định hướng, chưa phải kết luận full8.

Checkpoint model khoảng 328–329 MiB; boundary checkpoint khoảng 339–354 MiB.
Không có dấu hiệu OOM hoặc non-finite loss trong artifact đã gửi.

## 5. Đánh giá và đề xuất

### Kết luận hiện tại

**B (`task0_standard_frozen`) đang dẫn trước tạm thời**, vì hai metric chính ở task 1
`seen_all` đều cao hơn và khả năng giữ class cũ tốt hơn rõ rệt. Tuy nhiên đây chưa phải
“winner cuối” vì:

1. chênh lệch `seen_all` chỉ khoảng 2 điểm phần trăm;
2. A tốt hơn nhẹ trên class mới;
3. A còn tăng ở epoch 20;
4. mới có một member và hai task tail rất nhỏ;
5. chưa chứng minh behavior đến task 7.

### Hướng tiếp theo được khuyến nghị

- Nếu ưu tiên tiến độ: người dùng có thể **chấp thuận B** làm preprocessing cố định,
  sau đó mới thực hiện Prompt 14 để so không gian chọn exemplar.
- Nếu ưu tiên bằng chứng chắc hơn: chạy thêm A/B với cùng thiết kế đến ít nhất task 2
  (hoặc lặp task 0–1 trên member 1), rồi mới chốt. Không đổi loss weight, epoch hoặc
  sampler riêng cho một case.

Người dùng đã xác nhận ngày 2026-09-14: **chọn B `task0_standard_frozen`** làm
preprocessing cho bước tiếp theo. Quyết định này cho phép triển khai Prompt 14 để
so không gian exemplar; chưa cho phép tự chạy full8.
