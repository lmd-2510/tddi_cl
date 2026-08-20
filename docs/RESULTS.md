# Kết quả T-DDI Protocol Study

Tất cả giá trị dưới đây là **mean ± sample standard deviation của 5 training seed (0–4)** tại task cuối. Backbone, method và training config giống hệt nhau; chỉ task schedule thay đổi.

| Protocol | Macro-F1 ↑ | Balanced acc. ↑ | Accuracy ↑ | Weighted-F1 ↑ | Forgetting ↓ |
|---|---:|---:|---:|---:|---:|
| P0 Random | 0.376864 ± 0.019378 | 0.502997 ± 0.039036 | 0.219642 ± 0.062131 | 0.151110 ± 0.043221 | 0.321320 ± 0.041817 |
| P1 Frequency-balanced | 0.358491 ± 0.008062 | 0.518880 ± 0.015255 | 0.121489 ± 0.003700 | 0.075366 ± 0.003605 | 0.323575 ± 0.012571 |
| P2 Head→tail | 0.271933 ± 0.004970 | **0.764197 ± 0.017001** | 0.464476 ± 0.011882 | 0.545225 ± 0.011982 | **0.044137 ± 0.001055** |
| P3 Tail→head | **0.493526 ± 0.009007** | 0.452994 ± 0.013229 | **0.799556 ± 0.001323** | **0.762206 ± 0.002120** | 0.500292 ± 0.010709 |
| P4 Constrained mass-balanced | 0.382512 ± 0.024559 | 0.479544 ± 0.024990 | 0.162072 ± 0.014033 | 0.092925 ± 0.011320 | 0.356245 ± 0.028327 |
| P5 Multi-factor balanced | 0.379128 ± 0.016695 | 0.478522 ± 0.016990 | 0.160079 ± 0.019208 | 0.088828 ± 0.012071 | 0.356960 ± 0.027028 |
| P6 Difficulty balanced | 0.383124 ± 0.018418 | 0.494718 ± 0.019119 | 0.166201 ± 0.015583 | 0.096253 ± 0.012953 | 0.343788 ± 0.032406 |
| P7 Confusion spread | 0.380497 ± 0.020576 | 0.479161 ± 0.014044 | 0.165093 ± 0.014932 | 0.093393 ± 0.011142 | 0.366822 ± 0.014343 |
| P8 Controlled rarity drift | 0.375868 ± 0.023778 | 0.478833 ± 0.013550 | 0.181378 ± 0.026168 | 0.109418 ± 0.016392 | 0.352867 ± 0.012699 |

## Diễn giải

- P2 và P3 chứng minh thứ tự class arrival làm thay đổi mạnh metric. P2 giữ balanced accuracy và forgetting tốt nhưng macro-F1/accuracy thấp; P3 có accuracy và macro-F1 cao nhất nhưng forgetting tệ nhất. Đây là stress test, không phải hai ứng viên protocol trung lập.
- Trong nhóm protocol cân bằng P4–P8, P6 nhỉnh nhất về macro-F1/balanced accuracy/forgetting, nhưng dùng tín hiệu validation từ chính T-DDI.
- P4 không đứng đầu tuyệt đối về một metric, nhưng là lựa chọn chính hợp lý nhất để so method/backbone sau này vì không model-informed, có rarity constraint rõ và có 5 schedule seed.
- P0 tiếp tục được giữ làm random reference. P1 không còn được gọi là protocol chính vì cách chia ba stratum round-robin quá thô và kết quả macro-F1 thấp hơn P0/P4–P7.

## Kết luận khóa

Protocol chính: **P4**. Reference: **P0**. Stress tests: **P2/P3**. Advanced secondary protocols: **P5–P8**, trong đó **P6** là ứng viên model-informed tốt nhất. P1 là baseline frequency-balanced đơn giản.

Không so trực tiếp kết quả ở bảng này với run khác backbone/method/config. Kết quả thô và run provenance nằm trong `outputs/runs_backbones/`; thư mục này được ignore vì dung lượng lớn.
