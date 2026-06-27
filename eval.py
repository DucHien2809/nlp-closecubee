import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import evaluate
from tqdm.auto import tqdm
import unicodedata

# Thư viện Tokenizer gốc cho mô hình Custom
from tokenizers import Tokenizer as RawTokenizer

# ====== 1. Cấu hình chung ======
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
file_path = "dataset/1k_TV4_T2T.xlsx"
SEED = 42 

# ====== 2. Định nghĩa kiến trúc Mô hình Custom (Để nạp trọng số) ======

class SwiGLUFFN(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_model, d_ff)
        self.w3 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        gate = F.silu(self.w1(x))
        gated_linear = gate * self.w2(x)
        return self.dropout(self.w3(gated_linear))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class AdvancedSeq2SeqTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_encoder_layers=6, 
                 num_decoder_layers=6, d_ff=1024, dropout=0.1, pad_id=0):
        super().__init__()
        self.d_model = d_model
        self.pad_id = pad_id
        
        self.shared_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.positional_encoding = PositionalEncoding(d_model, dropout=dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True
        )
        for i in range(num_encoder_layers):
            encoder_layer.linear1 = nn.Identity()
            encoder_layer.linear2 = nn.Identity()
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.enc_ffns = nn.ModuleList([SwiGLUFFN(d_model, d_ff, dropout) for _ in range(num_encoder_layers)])

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        self.dec_ffns = nn.ModuleList([SwiGLUFFN(d_model, d_ff, dropout) for _ in range(num_decoder_layers)])
        
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.fc_out.weight = self.shared_embedding.weight

    def _generate_padding_mask(self, seq, pad_id):
        return (seq == pad_id)

    def _generate_subsequent_mask(self, sz, device):
        mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
        return mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))

    def forward(self, src, trg):
        src_padding_mask = self._generate_padding_mask(src, self.pad_id)
        trg_padding_mask = self._generate_padding_mask(trg, self.pad_id)
        trg_mask = self._generate_subsequent_mask(trg.size(1), src.device)

        src_emb = self.positional_encoding(self.shared_embedding(src) * math.sqrt(self.d_model))
        trg_emb = self.positional_encoding(self.shared_embedding(trg) * math.sqrt(self.d_model))

        enc_output = src_emb
        for layer, ffn in zip(self.encoder.layers, self.enc_ffns):
            attn_output = layer.self_attn(
                layer.norm1(enc_output), layer.norm1(enc_output), layer.norm1(enc_output),
                key_padding_mask=src_padding_mask
            )[0]
            enc_output = enc_output + layer.dropout1(attn_output)
            enc_output = enc_output + ffn(layer.norm2(enc_output))

        dec_output = trg_emb
        for layer, ffn in zip(self.decoder.layers, self.dec_ffns):
            attn_output1 = layer.self_attn(
                layer.norm1(dec_output), layer.norm1(dec_output), layer.norm1(dec_output),
                attn_mask=trg_mask, key_padding_mask=trg_padding_mask
            )[0]
            dec_output = dec_output + layer.dropout1(attn_output1)

            attn_output2 = layer.multihead_attn(
                layer.norm2(dec_output), enc_output, enc_output,
                key_padding_mask=src_padding_mask
            )[0]
            dec_output = dec_output + layer.dropout2(attn_output2)
            dec_output = dec_output + ffn(layer.norm3(dec_output))

        return self.fc_out(dec_output)


# ====== 3. Hàm chuẩn hóa văn bản (QUAN TRỌNG) ======
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize('NFC', text)
    text = text.lower()
    text = " ".join(text.split())
    text = text.replace(".", "").replace(",", "").replace("?", "").replace("!", "")
    return text.strip()


def safe_text_for_metrics(text):
    """
    Một số thư viện tính WER/ROUGE (jiwer, rouge_score) sẽ lỗi hoặc cho kết quả
    không hợp lệ nếu gặp chuỗi rỗng. Hàm này đảm bảo luôn có ít nhất 1 ký tự.
    """
    text = text.strip()
    return text if text else " "


# ====== 4. Load và Tách tập Test từ Excel ======
df = pd.read_excel(file_path)
df.columns = [col.lower().strip() for col in df.columns]
df = df.dropna(subset=['input', 'output'])
dataset = Dataset.from_pandas(df).train_test_split(test_size=0.1, seed=SEED)
test_dataset = dataset["test"]

print(f"Số lượng câu trong tập Test: {len(test_dataset)}")


# ====== 5. Thuật toán dịch mã Beam Search cho mô hình Custom ======
def padding_list(list_ids, max_length, pad_id):
    if len(list_ids) < max_length:
        return list_ids + [pad_id] * (max_length - len(list_ids))
    return list_ids[:max_length]


def custom_model_generate_beam(model, tokenizer, sentence, device, num_beams=5, max_length=64):
    """Bộ giải mã Beam Search tương thích cho mô hình Custom."""
    bos_id = tokenizer.token_to_id("[BOS]")
    eos_id = tokenizer.token_to_id("[EOS]")
    pad_id = tokenizer.token_to_id("[PAD]")
    
    src_ids = padding_list(tokenizer.encode(str(sentence)).ids, 64, pad_id)
    src_tensor = torch.tensor([src_ids], dtype=torch.long).to(device)
    
    beams = [([bos_id], 0.0)]
    completed_beams = []
    
    for t in range(max_length - 1):
        candidates = []
        for seq, score in beams:
            if seq[-1] == eos_id:
                completed_beams.append((seq, score))
                continue
                
            trg_tensor = torch.tensor([seq], dtype=torch.long).to(device)
            with torch.no_grad():
                logits = model(src_tensor, trg_tensor)
                next_token_logits = logits[0, -1, :]
                
            log_probs = F.log_softmax(next_token_logits, dim=-1)
            topk_log_probs, topk_ids = torch.topk(log_probs, num_beams)
            
            for log_p, token_id in zip(topk_log_probs, topk_ids):
                candidates.append((seq + [token_id.item()], score + log_p.item()))
                
        if not candidates:
            break
            
        candidates.sort(key=lambda x: x[1], reverse=True)
        beams = candidates[:num_beams]
        
        if all(seq[-1] == eos_id for seq, _ in beams):
            break
            
    all_completed = completed_beams + beams
    all_completed.sort(key=lambda x: x[1], reverse=True)
    best_seq = all_completed[0][0]
    
    # Loại bỏ các token đặc biệt khi giải mã chuỗi đầu ra
    generated_ids = [idx for idx in best_seq if idx not in [bos_id, eos_id, pad_id]]
    return tokenizer.decode(generated_ids)


# ====== 6. Hàm nạp an toàn Tokenizer & Trọng số Custom ======
def load_custom_model_safely(model_dir, excel_path, device):
    # Nạp thử nghiệm Tokenizer, tự động build lại nếu lệch phiên bản
    try:
        tokenizer = RawTokenizer.from_file(os.path.join(model_dir, "custom_tokenizer.json"))
    except Exception as e:
        print("[Thông báo] Khởi động chế độ tái cấu trúc Tokenizer cục bộ...")
        from tokenizers.models import BPE
        from tokenizers.trainers import BpeTrainer
        from tokenizers.pre_tokenizers import Whitespace
        from tokenizers.processors import TemplateProcessing
        
        df_local = pd.read_excel(excel_path)
        df_local.columns = [col.lower().strip() for col in df_local.columns]
        
        def get_training_corpus():
            for item in zip(df_local["input"], df_local["output"]):
                yield str(item[0])
                yield str(item[1])
                
        tokenizer = RawTokenizer(BPE(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = BpeTrainer(vocab_size=8000, special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"])
        tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)
        
        tokenizer.post_processor = TemplateProcessing(
            single="[BOS] $A [EOS]",
            special_tokens=[
                ("[BOS]", tokenizer.token_to_id("[BOS]")),
                ("[EOS]", tokenizer.token_to_id("[EOS]")),
            ],
        )
        tokenizer.save(os.path.join(model_dir, "custom_tokenizer.json"))

    vocab_size = tokenizer.get_vocab_size()
    pad_id = tokenizer.token_to_id("[PAD]")
    
    # Khởi tạo mạng nơ-ron
    model = AdvancedSeq2SeqTransformer(
        vocab_size=vocab_size,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        d_ff=1024,
        dropout=0.1,
        pad_id=pad_id
    )
    model.load_state_dict(torch.load(os.path.join(model_dir, "best_model_weights.pt"), map_location=device))
    model.to(device)
    model.eval()
    
    return tokenizer, model


# ====== 7. Hàm Inference tổng quát ======
def run_evaluation(model_dir, model_type="custom_advanced"):
    print(f"\n--- Đang đánh giá model: {model_type.upper()} ---")
    
    if model_type == "vit5":
        # Thử nạp từ thư mục cục bộ, nếu lỗi phiên bản sẽ tự động nạp từ Hugging Face Hub online
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_dir)
        except Exception as e:
            print("[Thông báo] Phát hiện lệch phiên bản Tokenizer cục bộ của ViT5. Tiến hành nạp online từ Hugging Face Hub...")
            tokenizer = AutoTokenizer.from_pretrained("VietAI/vit5-base")
            
        model = AutoModelForSeq2SeqLM.from_pretrained(model_dir).to(device)
        model.eval()
    else:
        # Nạp mô hình Custom nâng cao bằng hàm nạp an toàn
        tokenizer, model = load_custom_model_safely(model_dir, file_path, device)

    predictions = []
    references = []
    exact_matches = 0

    # Danh sách câu đã chuẩn hóa (dùng riêng cho WER & ROUGE - so sánh nguyên câu)
    cleaned_predictions = []
    cleaned_references = []

    for item in tqdm(test_dataset):
        input_text = item["input"]
        target_text = item["output"]

        # Thực hiện dịch bằng cơ chế tương ứng
        if model_type == "vit5":
            input_prompt = "sign-grammar: " + str(input_text)
            inputs = tokenizer(input_prompt, return_tensors="pt", max_length=128, truncation=True).to(device)
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_length=128,
                    num_beams=5
                )
            pred_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        else:
            # Gọi bộ giải mã Beam Search tùy chỉnh cho mô hình Custom
            pred_text = custom_model_generate_beam(model, tokenizer, input_text, device, num_beams=5)
        
        # --- CHUẨN HÓA TRƯỚC KHI SO SÁNH ---
        p_cleaned = clean_text(pred_text)
        t_cleaned = clean_text(target_text)
        
        predictions.append(pred_text)
        references.append([str(target_text).strip()])

        # Gom các câu đã chuẩn hóa để tính WER & ROUGE (so sánh nguyên câu)
        cleaned_predictions.append(safe_text_for_metrics(p_cleaned))
        cleaned_references.append(safe_text_for_metrics(t_cleaned))

        if p_cleaned == t_cleaned:
            exact_matches += 1

    # Tính điểm BLEU
    sacrebleu = evaluate.load("sacrebleu")
    results = sacrebleu.compute(predictions=predictions, references=references)

    # Tính điểm WER (Word Error Rate) - so sánh nguyên câu đã chuẩn hóa
    wer_metric = evaluate.load("wer")
    wer_score = wer_metric.compute(
        predictions=cleaned_predictions,
        references=cleaned_references
    ) * 100  # Quy về thang %

    # Tính điểm ROUGE-1, ROUGE-2, ROUGE-L - so sánh nguyên câu đã chuẩn hóa
    rouge_metric = evaluate.load("rouge")
    rouge_results = rouge_metric.compute(
        predictions=cleaned_predictions,
        references=cleaned_references
    )

    exact_match_score = (exact_matches / len(test_dataset)) * 100
    
    return {
        "bleu": results["score"],
        "exact_match": exact_match_score,
        "wer": wer_score,
        "rouge1": rouge_results["rouge1"] * 100,
        "rouge2": rouge_results["rouge2"] * 100,
        "rougeL": rouge_results["rougeL"] * 100,
        "sample_preds": predictions[:5],
        "sample_labels": [r[0] for r in references[:5]]
    }


# ====== 8. Chạy so sánh ======
vit5_results = run_evaluation("./bartpho_vsl_best_model", model_type="vit5")
custom_results = run_evaluation("./custom_sign_grammar", model_type="custom_advanced")


# ====== 9. Hiển thị bảng kết quả ======
print("\n" + "="*60)
print(f"{'Tiêu chí (Đã chuẩn hóa)':<25} | {'ViT5':<15} | {'Custom Advanced':<15}")
print("-" * 60)
print(f"{'BLEU Score':<25} | {vit5_results['bleu']:<15.2f} | {custom_results['bleu']:<15.2f}")
print(f"{'Exact Match (%)':<25} | {vit5_results['exact_match']:<15.2f} | {custom_results['exact_match']:<15.2f}")
print(f"{'WER (%)':<25} | {vit5_results['wer']:<15.2f} | {custom_results['wer']:<15.2f}")
print(f"{'ROUGE-1 (%)':<25} | {vit5_results['rouge1']:<15.2f} | {custom_results['rouge1']:<15.2f}")
print(f"{'ROUGE-2 (%)':<25} | {vit5_results['rouge2']:<15.2f} | {custom_results['rouge2']:<15.2f}")
print(f"{'ROUGE-L (%)':<25} | {vit5_results['rougeL']:<15.2f} | {custom_results['rougeL']:<15.2f}")
print("="*60)

# So sánh chi tiết
print("\nKiểm tra kỹ thuật câu lỗi của Custom Advanced:")
for i in range(len(custom_results['sample_preds'])):
    p = custom_results['sample_preds'][i]
    l = custom_results['sample_labels'][i]
    if clean_text(p) != clean_text(l):
        print(f"Câu {i} không khớp!")
        print(f"  - Pred:  [{p}]")
        print(f"  - Label: [{l}]")
    else:
        print(f"Câu {i}: Khớp hoàn hảo sau chuẩn hóa.")