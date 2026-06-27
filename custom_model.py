import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
from tqdm.auto import tqdm

# Thư viện dùng để tự tạo Tokenizer từ đầu
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing

# =====================================================================
# ====== PHẦN 1: TỰ HUẤN LUYỆN TOKENIZER CHUYÊN BIỆT TỪ ĐẦU ==========
# =====================================================================

# 1. Đọc Dataset từ CSV để lấy dữ liệu huấn luyện Tokenizer
data_file_path = "/data/Parallel-Corpus-Vie-VSL/Corpus-Vie-VSL-10K.csv"
dataset = load_dataset("csv", data_files={"train": data_file_path})

# Trích xuất toàn bộ văn bản để học từ vựng
def get_training_corpus():
    for item in dataset["train"]:
        yield item["input"]
        yield item["output"]

# 2. Khởi tạo và cấu hình bộ tách từ BPE hoàn toàn offline
raw_tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
raw_tokenizer.pre_tokenizer = Whitespace()

# Huấn luyện bộ từ vựng với kích thước tối ưu là 8000 từ (phù hợp với tập 10k câu)
trainer = BpeTrainer(
    vocab_size=8000, 
    special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"]
)
raw_tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)

# Định nghĩa cách đóng gói token: tự động thêm [BOS] ở đầu và [EOS] ở cuối câu
raw_tokenizer.post_processor = TemplateProcessing(
    single="[BOS] $A [EOS]",
    special_tokens=[
        ("[BOS]", raw_tokenizer.token_to_id("[BOS]")),
        ("[EOS]", raw_tokenizer.token_to_id("[EOS]")),
    ],
)

pad_token_id = raw_tokenizer.token_to_id("[PAD]")
vocab_size = raw_tokenizer.get_vocab_size()

print(f"Hoàn tất huấn luyện Tokenizer riêng. Kích thước từ vựng: {vocab_size}")


# =====================================================================
# ====== PHẦN 2: ĐỊNH NGHĨA KIẾN TRÚC MÔ HÌNH CUSTOM ADVANCED SEQ2SEQ ==
# =====================================================================

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
        
        # Encoder với Pre-LN
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True
        )
        for i in range(num_encoder_layers):
            encoder_layer.linear1 = nn.Identity()
            encoder_layer.linear2 = nn.Identity()
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.enc_ffns = nn.ModuleList([SwiGLUFFN(d_model, d_ff, dropout) for _ in range(num_encoder_layers)])

        # Decoder với Pre-LN
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        self.dec_ffns = nn.ModuleList([SwiGLUFFN(d_model, d_ff, dropout) for _ in range(num_decoder_layers)])
        
        # Lớp dự đoán đầu ra chia sẻ trọng số
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


# =====================================================================
# ====== PHẦN 3: XỬ LÝ DỮ LIỆU & HUẤN LUYỆN (TRAINING PIPELINE) ======
# =====================================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hàm mã hóa câu văn bản thủ công bằng Tokenizer vừa tự huấn luyện
def padding_list(list_ids, max_length, pad_id):
    if len(list_ids) < max_length:
        return list_ids + [pad_id] * (max_length - len(list_ids))
    return list_ids[:max_length]

def preprocess_function_custom(examples):
    src_batch = []
    trg_batch = []
    for inp, out in zip(examples["input"], examples["output"]):
        # Mã hóa câu sang ID số và tự động đệm (padding)
        src_ids = padding_list(raw_tokenizer.encode(inp).ids, 64, pad_token_id)
        trg_ids = padding_list(raw_tokenizer.encode(out).ids, 64, pad_token_id)
        src_batch.append(src_ids)
        trg_batch.append(trg_ids)
    return {
        "src_ids": src_batch,
        "trg_ids": trg_batch
    }

tokenized_datasets = dataset.map(preprocess_function_custom, batched=True)

# Chia train / eval
split_datasets = tokenized_datasets["train"].train_test_split(test_size=0.2, seed=42)
train_dataset = split_datasets["train"]
eval_dataset = split_datasets["test"]

def collate_fn(batch):
    src_ids = torch.tensor([item["src_ids"] for item in batch])
    trg_ids = torch.tensor([item["trg_ids"] for item in batch])
    return src_ids, trg_ids

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
eval_loader = DataLoader(eval_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

# Khởi tạo mô hình Transformer Seq2Seq nâng cao
model = AdvancedSeq2SeqTransformer(
    vocab_size=vocab_size,
    d_model=512,
    nhead=8,
    num_encoder_layers=6,
    num_decoder_layers=6,
    d_ff=1024,
    dropout=0.1,
    pad_id=pad_token_id
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id, label_smoothing=0.1)

gradient_accumulation_steps = 4
num_epochs = 10
best_eval_loss = float("inf")

print(f"Bắt đầu huấn luyện trên thiết bị: {device}")

for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0
    optimizer.zero_grad()
    
    train_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]")
    for step, (src_ids, trg_ids) in enumerate(train_bar):
        src_ids = src_ids.to(device)
        trg_ids = trg_ids.to(device)
        
        trg_input = trg_ids[:, :-1]
        trg_target = trg_ids[:, 1:]
        
        with torch.autocast(device_type="cuda" if "cuda" in str(device) else "cpu", dtype=torch.bfloat16):
            logits = model(src_ids, trg_input)
            loss = criterion(logits.reshape(-1, logits.size(-1)), trg_target.reshape(-1))
            loss = loss / gradient_accumulation_steps
            
        loss.backward()
        total_train_loss += loss.item() * gradient_accumulation_steps
        
        if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == len(train_loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            
        train_bar.set_postfix(loss=loss.item() * gradient_accumulation_steps)

    avg_train_loss = total_train_loss / len(train_loader)

    # Đánh giá mô hình
    model.eval()
    total_eval_loss = 0
    with torch.no_grad():
        for src_ids, trg_ids in tqdm(eval_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Eval]"):
            src_ids = src_ids.to(device)
            trg_ids = trg_ids.to(device)
            
            trg_input = trg_ids[:, :-1]
            trg_target = trg_ids[:, 1:]
            
            logits = model(src_ids, trg_input)
            loss = criterion(logits.reshape(-1, logits.size(-1)), trg_target.reshape(-1))
            total_eval_loss += loss.item()
            
    avg_eval_loss = total_eval_loss / len(eval_loader)
    print(f"\n>>> Epoch {epoch + 1}: Train Loss = {avg_train_loss:.4f} | Eval Loss = {avg_eval_loss:.4f}")

    # Lưu trọng số tốt nhất
    output_dir = "./custom_advanced_sign_grammar"
    os.makedirs(output_dir, exist_ok=True)
    if avg_eval_loss < best_eval_loss:
        best_eval_loss = avg_eval_loss
        torch.save(model.state_dict(), os.path.join(output_dir, "best_model_weights.pt"))
        print(f"==> Đã lưu trọng số mô hình tốt nhất với Eval Loss: {best_eval_loss:.4f}")

# Lưu Tokenizer tự huấn luyện dạng file JSON để nạp lại sau này
raw_tokenizer.save(os.path.join(output_dir, "custom_tokenizer.json"))
print("Hoàn tất huấn luyện.")