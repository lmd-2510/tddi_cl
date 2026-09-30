# P4 Hybrid + equal-class buffer full run

Run này chuyển **cấu hình được chốt để phát triển tiếp** trong
`docs/best_historical_config.md` từ P3 sang P4 `constrained_mass_balanced`.
Ngoài task schedule và preprocessing task-0 tương ứng P4, các thiết lập khoa
học được giữ nguyên:

- Hybrid loss: Focal cho current-task samples, CE cho replay; không distillation;
- equal/macro buffer: `fold_equal_class_capacity_v1`;
- replay fraction `12.5%`, repeat cap `3`;
- `27,778` slots cho mỗi member (`83,334` physical slots cho ba member);
- 30 epoch tối đa/task, patience 5, ba member chạy tuần tự;
- offline probability ensemble cho cả validation OOF và test;
- final metrics, visualization và chẩn đoán task 6 → task 7;
- review ZIP loại trừ NPZ, checkpoint và dữ liệu nguồn nặng.

Đây là một thí nghiệm mới theo P4, không ghi đè hay được gọi là tái lập kết quả
P3 historical anchor.

## Chạy một lệnh

```bash
cd "$HOME/DrugDrug/cil-tddi/cil-tddi"
conda activate ai_env
bash scripts/run_p4_hybrid_equal_buffer_full.sh start
```

`start` chạy bằng `nohup`. Sau khi terminal in `[STARTED]`, có thể đóng VS Code
hoặc máy cá nhân; tiến trình tiếp tục trên server. Nếu preprocessing P4 chưa có,
runner sẽ tự tạo trong `study_assets/preprocessing_p4_seed0_fold42` trước khi
bắt đầu member 0.

## Theo dõi và tiếp tục

```bash
bash scripts/run_p4_hybrid_equal_buffer_full.sh status
bash scripts/run_p4_hybrid_equal_buffer_full.sh follow
```

Thoát chế độ `follow` bằng `Ctrl+C` không dừng job nền. Nếu server/job bị ngắt,
tiếp tục đúng run gần nhất bằng:

```bash
bash scripts/run_p4_hybrid_equal_buffer_full.sh resume
```

## Kết quả

Khi hoàn tất, `status` hiển thị output root. Trong đó có `full_manifest.json`,
`offline_evaluation/`, `final_results/`, `diagnostics/` và:

```text
review/p4_hybrid_equal_buffer_full8_e30_mem4_review_<timestamp>.zip
review/p4_hybrid_equal_buffer_full8_e30_mem4_review_<timestamp>.zip.sha256
```

ZIP chứa config, summary/metric, log, CSV/JSON và PNG cần cho đánh giá; không
chứa prediction NPZ hoặc model checkpoint.
