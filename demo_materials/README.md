# Tài liệu phục vụ demo

Thư mục này gồm các file cần cho demo và phân tích mẫu đúng/sai trong slide.

## Nội dung chính

- `metrics/comparison_table.md`: bảng metric gọn cho slide kết quả.
- `reports/leaderboard_test.md`: leaderboard đầy đủ trên tập test.
- `examples/selected_cases.md`: các mẫu đúng/sai đã chọn sẵn, có SRC/REF và dự đoán của ViT5, BARTpho, MBR, FELIX++.
- `examples/*_errors_test.md` và `examples/*_correct_test.md`: danh sách mẫu đúng/sai theo từng hệ thống.
- `predictions/*_predictions_test.jsonl`: dự đoán đầy đủ nếu cần tra cứu thêm.
- `configs/`: cấu hình đã dùng cho các lần chạy.

## Mô hình và checkpoint

FELIX++ 15 epoch đã được tải về máy:

```text
outputs/felix_plus_xlmr_large_15ep/model
outputs/felix_plus_xlmr_large_15ep/model/encoder/model.safetensors
outputs/felix_plus_xlmr_large_15ep/model/heads.pt
```

Model/checkpoint của ViT5-base và BARTpho vẫn còn trên Modal volume `vsl-artifacts`. Tải về khi cần demo suy luận trên máy:

```powershell
modal volume get vsl-artifacts /outputs/vit5_base_15ep ./outputs/vit5_base_15ep_backup
modal volume get vsl-artifacts /outputs/bartpho_syllable ./outputs/bartpho_syllable_backup
```
