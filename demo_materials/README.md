# Demo materials

Thu muc nay gom cac file can cho demo va phan tich mau dung/sai trong slide.

## Noi dung chinh

- `metrics/comparison_table.md`: bang metric gon cho slide ket qua.
- `reports/leaderboard_test.md`: leaderboard day du tren test set.
- `examples/selected_cases.md`: cac mau dung/sai da chon san, co SRC/REF/du doan cua ViT5, BARTpho, MBR, FELIX++.
- `examples/*_errors_test.md` va `examples/*_correct_test.md`: danh sach mau dung/sai theo tung he thong.
- `predictions/*_predictions_test.jsonl`: du doan day du neu can tra cuu them.
- `configs/`: config da dung cho cac run.

## Model/checkpoint

FELIX++ 15 epoch da duoc tai ve local:

```text
outputs/felix_plus_xlmr_large_15ep/model
outputs/felix_plus_xlmr_large_15ep/model/encoder/model.safetensors
outputs/felix_plus_xlmr_large_15ep/model/heads.pt
```

ViT5-base va BARTpho model/checkpoint van con tren Modal volume `vsl-artifacts`. Tai ve khi can demo inference local:

```powershell
modal volume get vsl-artifacts /outputs/vit5_base_15ep ./outputs/vit5_base_15ep_backup
modal volume get vsl-artifacts /outputs/bartpho_syllable ./outputs/bartpho_syllable_backup
```
