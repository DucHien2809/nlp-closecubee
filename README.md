# Chuyển câu tiếng Việt sang gloss VSL (text-to-text)

Dự án nhận một câu tiếng Việt và sinh ra chuỗi gloss Ngôn ngữ Ký hiệu Việt Nam
(VSL) ở dạng văn bản, theo quy ước của bộ ngữ liệu song song Parallel-Corpus-Vie-VSL.
Đây là bài toán NLP text-to-text thuần; không có video, pose hay xử lý ảnh.

```
"Tôi 19 tuổi ."      ->  "Tôi tuổi 19 ."
"Ai biết bơi ?"      ->  "Biết bơi ai ?"
"Mít thì ngọt ."     ->  "Mít ngọt ."
"Con gà ăn thóc ."   ->  "Con gà thóc ăn ."
```

Đầu ra được hiểu là gloss VSL theo quy ước của corpus, không phải "ngữ pháp NNKH
chuẩn tuyệt đối". Mô hình học ánh xạ thống kê từ dữ liệu nên kết quả phản ánh quy
ước của chính corpus đó.

## 1. Bài toán

Cho câu nguồn tiếng Việt `x = (x₁,…,xₙ)` (token theo âm tiết), sinh chuỗi gloss
đích `y = (y₁,…,yₘ)`. Về bản chất đây là một bài toán chuyển dịch chuỗi-sang-chuỗi
trong cùng một ngôn ngữ. Khi phân tích corpus, chúng tôi thấy phép biến đổi chủ
yếu là xoá hư từ và đảo trật tự thành phần; chỉ khoảng 2% số cặp có token mới
không nằm trong câu nguồn. Vì thế bài toán gần với chỉnh sửa văn bản hơn là dịch
máy mở.

## 2. Dữ liệu

Nguồn: Parallel-Corpus-Vie-VSL, gồm 10.000 cặp câu căn lề 1-1 theo dòng giữa bốn
file: `Vie10k.txt` / `VSL10k.txt` (câu) và `Vie10k_phantich.txt` /
`VSL10k_phantich.txt` (cây cú pháp thành phần kèm nhãn từ loại theo VietTreebank).
Cây phía tiếng Việt cho sẵn word-segmentation và POS, chúng tôi dùng lại cho
baseline luật và cho phân tích lỗi.

Một vài con số sau khi chuẩn hoá (xem `reports/corpus_stats.md`):

| Chỉ số | Giá trị |
|---|---|
| Số cặp | 10.000 (9.477 cặp khác nhau, 9.457 câu nguồn khác nhau) |
| Độ dài trung bình nguồn / đích | 8.7 / 7.4 token |
| Cây gold khớp bề mặt câu nguồn | 9.996 / 10.000 |

Mỗi cặp được gán một nhãn kiểu biến đổi (`vsl_gloss/data/taxonomy.py`, so khớp
token đã hạ-hoa để bỏ nhiễu viết hoa):

| Kiểu | Ý nghĩa | Tỉ lệ |
|---|---|---:|
| `identical` | nguồn trùng đích | 23.9% |
| `reorder_only` | cùng tập từ, khác thứ tự | 1.3% |
| `deletion_only` | đích là tập con của nguồn, giữ thứ tự | 62.2% |
| `deletion_reorder` | tập con nhưng đảo thứ tự | 11.1% |
| `lexical` | đích có token không thuộc nguồn | 1.6% |

Điểm đáng chú ý nhất: phần lớn biến đổi là xoá từ, không phải đảo trật tự. Quan
sát này định hướng cả cách chọn mô hình lẫn cách đọc số liệu ở phần sau.

## 3. Chia dữ liệu

`train / val / test = 7.586 / 942 / 949`, seed cố định (`vsl_gloss/data/split.py`),
với hai ràng buộc:

1. Không rò rỉ nguồn. 377 câu nguồn xuất hiện nhiều lần (21 câu có nhiều gloss
   khác nhau) được gom lại để không câu nguồn nào lọt vào hai tập cùng lúc; có
   hàm `verify_no_leakage` kiểm tra tự động. Bản gốc dùng `train_test_split`
   không seed nên dễ rò rỉ, đó là lý do con số WER 1.79% trong báo cáo cũ không
   tái lập được.
2. Phân tầng theo kiểu biến đổi để val/test giữ đúng tỉ lệ copy/đảo/xoá.

## 4. Các hệ thống

### 4.1 Baseline (`vsl_gloss/baselines/`)

- `copy`: xuất nguyên câu nguồn. Vì gần 24% cặp trùng nhau nên copy đã khá mạnh
  và là mốc bắt buộc phải vượt.
- `rule_based`: bộ luật chạy trên cây gold gồm xoá hư từ (`thì, rất, và, các,
  những`), đảo SVO sang SOV, đưa số từ ra sau danh từ (`19 tuổi` -> `tuổi 19`) và
  chuyển từ hỏi về cuối câu (`Ai biết bơi?` -> `Biết bơi ai?`). Mỗi luật bật/tắt
  độc lập để phục vụ ablation, và bị giới hạn bởi độ chính xác của cây phân tích.

### 4.2 Seq2seq (`vsl_gloss/train.py`)

Fine-tune ViT5 dạng encoder-decoder. Cấu hình mặc định dùng `VietAI/vit5-base`
(`configs/default.yaml`); có sẵn `configs/vit5_large.yaml` cho bản large và
`configs/bartpho.yaml` nếu muốn đổi backbone sang BARTpho-syllable. So với script
gốc, bản này: đánh giá bằng generate sau mỗi epoch để có BLEU/WER/EM thật và chọn
checkpoint theo BLEU; dùng cùng một task prefix lúc train và lúc suy luận; thêm
label smoothing, lịch học cosine và early stopping; xuất file dự đoán theo định
dạng chung để chấm trên cùng leaderboard.

### 4.3 Constrained decoding (`vsl_gloss/models/constrained_decoding.py`)

Tận dụng việc đích gần như luôn là tập con của nguồn, một `LogitsProcessor` chỉ
cho phép sinh các sub-word có trong câu nguồn cộng với tập token luôn được phép
(eos, pad, dấu câu). Đây là một dạng constrained decoding nhẹ, không cần huấn
luyện thêm. Kết quả luôn được báo cáo kèm bản decode tự do để thấy rõ đánh đổi.

### 4.4 FELIX – mô hình chỉnh sửa (`vsl_gloss/felix/`)

Thay vì sinh tự do, hệ này mô hình hoá trực tiếp hai phép biến đổi mà corpus thực
hiện theo kiến trúc FELIX (Mallinson et al., Findings of EMNLP 2020). Một encoder
chung (`xlm-roberta-base`) với hai đầu ra:

- Tagging head gán mỗi token nguồn nhãn KEEP hoặc DELETE.
- Pointer network sắp xếp lại các token được giữ bằng cách dự đoán token kế tiếp
  cho từng vị trí, rồi decode tham lam theo con trỏ.

Nhãn huấn luyện được trích tất định từ cặp `(vie, vsl)` nhờ căn lề 1-1 của corpus
(`felix/labels.py`). Trước khi train, file `reports/felix_coverage.json` cho biết
trần lý thuyết của hướng này: 85.8% số ca chỉ cần tag, 12.5% cần pointer, 1.7%
cần chèn từ mới (ngoài tầm của bản hiện tại), tức biểu diễn được 98.3%. Nếu áp
nhãn vàng, oracle ceiling đạt BLEU 99.25 / WER 0.54, cao hơn cả seq2seq. Hạn chế
là bản này không có insertion head nên không sinh được khoảng 2% ca `lexical`.

## 5. Đánh giá

Các metric: BLEU (sacreBLEU, `tokenize=none`), chrF, WER (jiwer, là metric chính,
càng thấp càng tốt), TER và Exact-Match. Mỗi metric được báo cáo ở ba lát cắt:
tổng thể, theo từng kiểu biến đổi, và bản hạ-hoa. Vì 24% test là copy nên nếu chỉ
nhìn số tổng thể sẽ bị đánh lừa; bảng theo category mới cho thấy hệ thống có thực
sự học được phép biến đổi hay không. Toàn bộ prediction đi qua đúng một bộ chuẩn
hoá như tham chiếu. Mã ở `vsl_gloss/metrics.py` và `vsl_gloss/evaluate.py`.

## 6. Kết quả (test, n = 949)

Bảng đầy đủ ở `reports/leaderboard_test.md`, số liệu từng hệ ở
`outputs/<system>/metrics_test.json`.

| Hệ thống | BLEU | chrF | WER | TER | EM |
|---|---:|---:|---:|---:|---:|
| ViT5-base + BARTpho + MBR | 97.00 | 98.53 | 1.89 | 1.66 | 93.36 |
| ViT5-base, 15 epoch | 96.98 | 98.32 | 1.92 | 1.73 | 93.15 |
| ViT5-base, 1 epoch | 96.62 | 98.19 | 2.21 | 2.02 | 92.52 |
| BARTpho-syllable, 15 epoch | 96.38 | 97.82 | 2.30 | 2.07 | 90.83 |
| ViT5-large, 15 epoch | 96.06 | 97.71 | 2.43 | 2.16 | 90.09 |
| ViT5-base 1ep + constrained | 94.23 | 96.41 | 3.90 | 3.60 | 81.88 |
| ViT5-base 15ep + constrained | 93.27 | 95.05 | 4.97 | 4.70 | 82.61 |
| ViT5-large + constrained | 89.12 | 92.49 | 6.74 | 6.47 | 68.49 |
| FELIX (tag + pointer) | 88.82 | 95.77 | 8.61 | 1.62 | 46.26 |
| FELIX++ (XLM-R-large), 15 epoch | 84.92 | 93.17 | 12.35 | 11.79 | 28.45 |
| baseline copy | 61.71 | 84.50 | 20.97 | 20.55 | 23.71 |
| baseline rule | 34.61 | 68.59 | 48.01 | 30.75 | 6.74 |

Một vài điều rút ra:

- Ensemble ViT5-base + BARTpho + MBR đạt kết quả tốt nhất. MBR không huấn luyện
  thêm mô hình mà chọn câu có utility chrF kỳ vọng cao nhất từ tập ứng viên do
  hai backbone sinh ra; mức cải thiện so với ViT5-base nhỏ nhưng nhất quán.
- ViT5-base 15 epoch là mô hình đơn tốt nhất. Bản 1 epoch đã gần chạm trần, cho
  thấy bài toán không quá khó với seq2seq.
- ViT5-large không tốt hơn base; với corpus chỉ 7.6k câu huấn luyện, mô hình lớn
  hơn có dấu hiệu overfit.
- Constrained decoding luôn làm giảm điểm. Mô hình sau fine-tune đã tự học "chỉ
  dùng từ trong câu nguồn", nên ràng buộc cứng chỉ làm mất sự linh hoạt.
- FELIX vượt xa baseline nhưng thua seq2seq. Trần lý thuyết của nó cao hơn (oracle
  99.25) nên khoảng cách nằm ở khâu pointer-reorder (EM chỉ 46%), không phải ở khả
  năng biểu diễn. Nói cách khác, trên corpus thiên về xoá từ, học copy-có-chỉnh-sửa
  dễ hơn học hoán vị.

## 7. Ablation

- Luật: `scripts/run_ablation.py` tạo 6 biến thể (full, delete-only, bỏ delete, bỏ
  SOV, bỏ numeral, bỏ wh) để tách đóng góp của từng luật. Đáng chú ý là luật SOV
  toàn cục làm giảm điểm so với copy, vì corpus chủ yếu giữ nguyên trật tự.
- Mô hình: decode tự do so với constrained; base so với large.
- Hướng mô hình hoá: seq2seq sinh tự do so với edit-based (FELIX), hai cách giải
  khác hẳn cho cùng một bài toán, chấm trên cùng leaderboard.

## 8. Phân tích lỗi

`outputs/<system>/errors_test.md` liệt kê các ca sai theo category (SRC/REF/HYP).
Nhìn chung, ca `lexical` là khó nhất với mọi hệ (mô hình phải sinh từ không có
trong nguồn), khớp với việc loại này chỉ chiếm 1.6% corpus nên thiếu dữ liệu để
học. Phần còn lại chủ yếu là xoá thừa/thiếu hư từ và đảo sai cục bộ.

## 9. Bố cục mã nguồn

```
vsl-text2gloss/
├── configs/                 default.yaml, vit5_large.yaml, bartpho.yaml, felix.yaml
├── vsl_gloss/
│   ├── config.py            cấu hình typed nạp từ YAML
│   ├── utils.py             logging, seed, JSONL IO, override CLI
│   ├── data/
│   │   ├── normalize.py     chuẩn hoá văn bản
│   │   ├── parse_tree.py    đọc cây cú pháp (.._phantich.txt)
│   │   ├── taxonomy.py      phân loại kiểu biến đổi
│   │   ├── prepare.py       làm sạch dữ liệu + thống kê
│   │   └── split.py         chia train/val/test chống rò rỉ
│   ├── baselines/           copy, rule_based
│   ├── models/              constrained_decoding
│   ├── felix/               hệ edit-based (xem mục 4.4)
│   ├── metrics.py           BLEU/chrF/WER/TER/EM
│   ├── train.py             fine-tune ViT5/BARTpho
│   ├── predict.py           suy luận theo batch
│   └── evaluate.py          chấm điểm + leaderboard
├── scripts/                 run_pipeline.py, run_ablation.py
└── modal_app.py             chạy toàn bộ trên GPU Modal
```

## 10. Cách chạy

```bash
pip install -r requirements.txt

# Data + baseline + leaderboard (chạy CPU, nhanh)
python scripts/run_pipeline.py
python scripts/run_ablation.py
python -m vsl_gloss.evaluate --compare

# Fine-tune (cần GPU)
python scripts/run_pipeline.py --train --config configs/default.yaml

# Trên Modal GPU (sau khi modal token set ...)
# Phải chỉ rõ entrypoint: ::main (seq2seq full), ::train (1 backbone),
# ::felix (edit model), ::ensemble (MBR ensemble các backbone đã train).
modal run --detach modal_app.py::main --config configs/default.yaml --gpu A10G
modal run --detach modal_app.py::main --config configs/vit5_large.yaml
modal run --detach modal_app.py::felix --gpu A100

# Cải tiến BARTpho + MBR-ensemble (chạy TUẦN TỰ, đừng song song chung Volume):
#   1) train backbone thứ 2 (BARTpho) -> outputs/bartpho_syllable
modal run --detach modal_app.py::train --config configs/bartpho.yaml --gpu A10G
#   2) MBR-ensemble ViT5 (đã có) + BARTpho -> outputs/ensemble_mbr
modal run --detach modal_app.py::ensemble --config configs/ensemble_mbr.yaml --gpu A10G

modal volume get vsl-artifacts /reports ./reports_remote
```

Dịch nhanh một câu:

```bash
python -m vsl_gloss.predict --model outputs/vit5_base_15ep/model --text "Tôi 19 tuổi ."
```

## 11. Tham khảo

- Phan et al., ViT5, NAACL-SRW 2022; Tran et al., BARTpho, INTERSPEECH 2022 (backbone).
- Mallinson et al., FELIX, Findings of EMNLP 2020; Malmi et al., LaserTagger,
  EMNLP 2019 (chỉnh sửa văn bản dạng tag).
- Hokamp & Liu 2017; Post & Vilar 2018 (constrained decoding).
- Eikema & Aziz, *Is MAP Decoding All You Need?*, EMNLP 2020 (MBR decoding —
  cơ sở cho hệ `ensemble_mbr` gộp candidate của ViT5 + BARTpho).
- Post, A Call for Clarity in Reporting BLEU, WMT 2018 (sacreBLEU).
- Yin & Read, STMC-Transformer, COLING 2020; Müller et al., ACL 2023 (về giới hạn
  của gloss, lý do không tuyên bố "chuẩn tuyệt đối").

## 12. Hạn chế

- Baseline luật phụ thuộc cây phân tích gold; với câu mới sẽ cần một POS tagger
  bên ngoài.
- Corpus có quy ước riêng và không hoàn toàn nhất quán (ví dụ `học môn 4` giữ SVO
  trong khi `mèo thích` lại chuyển SOV), nên trần điểm tuyệt đối có giới hạn.
- Đây là hệ hỗ trợ cho mục đích nghiên cứu và học tập, không thay thế người phiên
  dịch NNKH.
