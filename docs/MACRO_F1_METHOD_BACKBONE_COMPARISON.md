# So sánh Macro-F1 theo method và backbone

Metric là **Final Macro-F1 của offline ensemble ba member tại task 7**, trên toàn bộ test của 178 lớp, coverage 100%. Các run dùng protocol Head-to-Tail, experiment seed `0`, fold seed `42`.

| Method | TDDI | DDINet | MFP-Siamese |
| --- | ---: | ---: | ---: |
| EWC | 0.377209 | 0.223290 | — |
| A-GEM | 0.388085 | 0.256433 | — |
| ER | **0.500563** | **0.521134** | — |
| IDER | 0.376521 | 0.301342 | — |
| UPCL | 0.219061 | 0.284112 | — |
| Task Arithmetic | 0.208722 | 0.202022 | — |
| TIES | 0.231388 | 0.203080 | — |
| OPCM | 0.082172 | 0.003996 | — |
| MagMax | 0.132647 | 0.217054 | — |

- In đậm giá trị cao nhất trong mỗi cột backbone.
- Dấu `—` dành cho kết quả đang chờ bổ sung.
- UPCL–DDINet dùng hidden cuối 256 chiều. Các run DDINet còn lại dùng hidden cuối 128 chiều.
- Nguồn số liệu và cấu hình từng run nằm trong [bảng tổng hợp experiments](EXPERIMENT_METRICS_SUMMARY.md).
