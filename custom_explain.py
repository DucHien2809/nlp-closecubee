"""
explain.py
==========
Giải thích white-box quá trình mô hình Custom Seq2Seq sinh ra câu ngữ pháp VSL từng bước một:
tại mỗi bước decode, hiển thị top-k từ ứng viên kèm xác suất, và từ nào
thực sự được chọn dựa trên thuật toán Beam Search tự định nghĩa.

Hỗ trợ:
  - Tải Tokenizer BPE riêng biệt và Trọng số mô hình Custom Transformer từ thư mục huấn luyện.
  - Tự động chạy thuật toán Beam Search lưu vết (Logged Beam Search).
  - Xuất dữ liệu phân tích ra JSON.
  - Vẽ đồ thị cây quyết định (Beam Decision Tree) lưu thành PNG.
  - Vẽ Heatmap xác suất các ứng viên qua từng bước sinh lưu thành PNG.
"""

import json
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# Thư viện hỗ trợ trực quan hóa
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from tokenizers import Tokenizer  # Thư viện Tokenizer gốc

# ====== Cấu hình mô hình tự định nghĩa ======
MODEL_DIR = "./custom_sign_grammar"
TOP_K = 5            # số ứng viên hiển thị tại mỗi bước
MAX_LENGTH = 64      # Độ dài max khớp với cấu hình lúc train
NUM_BEAMS = 5


# =====================================================================
# ====== 1. KHAI BÁO LẠI KIẾN TRÚC MÔ HÌNH ĐỂ PYTORCH KHÔI PHỤC =======
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


# =====================================================================
# ====== 2. CÁC HÀM TIỆN ÍCH CHO MÔ HÌNH VÀ TOKENIZER TỰ ĐỊNH NGHĨA ===
# =====================================================================

# Đường dẫn tới tệp dữ liệu gốc (để dùng làm fallback xây dựng lại Tokenizer nếu bị lệch phiên bản)
DATA_EXCEL_PATH = "./dataset/Corpus-Vie-VSL-10K.xlsx" # Bạn hãy điều chỉnh đường dẫn này cho đúng trên NAS của bạn

def load_model(model_dir=MODEL_DIR, excel_path=DATA_EXCEL_PATH):
    # 1. Thử nạp Tokenizer từ file JSON
    try:
        tokenizer = Tokenizer.from_file(os.path.join(model_dir, "custom_tokenizer.json"))
        print("[Thông tin] Nạp Tokenizer lưu trữ thành công.")
    except Exception as e:
        print(f"[Cảnh báo] Không thể nạp Tokenizer do lệch phiên bản thư viện: {str(e)}")
        print("==> Hệ thống tiến hành tự động xây dựng lại Tokenizer tương thích cục bộ...")
        
        # Tiến hành build lại Tokenizer ngay tại chỗ từ file Excel
        import pandas as pd
        from tokenizers import Tokenizer as RawTokenizer
        from tokenizers.models import BPE
        from tokenizers.trainers import BpeTrainer
        from tokenizers.pre_tokenizers import Whitespace
        from tokenizers.processors import TemplateProcessing
        
        if not os.path.exists(excel_path):
            raise FileNotFoundError(
                f"Không tìm thấy file dữ liệu tại '{excel_path}' để rebuild Tokenizer. "
                "Vui lòng kiểm tra lại cấu hình biến `DATA_EXCEL_PATH`."
            )
            
        df = pd.read_excel(excel_path)
        df.columns = [col.lower().strip() for col in df.columns]
        
        def get_training_corpus():
            for item in zip(df["input"], df["output"]):
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
        # Ghi đè lại file cục bộ để lần sau không phải build lại
        tokenizer.save(os.path.join(model_dir, "custom_tokenizer.json"))
        print("[Thành công] Đã xây dựng và đồng bộ hóa thành công Tokenizer mới tương thích với hệ thống hiện tại.")

    vocab_size = tokenizer.get_vocab_size()
    pad_id = tokenizer.token_to_id("[PAD]")
    
    # 2. Khởi tạo mô hình mạng nơ-ron
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
    
    # 3. Nạp trọng số mô hình tốt nhất (.pt)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(os.path.join(model_dir, "best_model_weights.pt"), map_location=device))
    model.to(device)
    model.eval()
    
    return tokenizer, model, device


def padding_list(list_ids, max_length, pad_id):
    if len(list_ids) < max_length:
        return list_ids + [pad_id] * (max_length - len(list_ids))
    return list_ids[:max_length]


def _format_token(tokenizer, token_id):
    """Giải mã 1 token_id thành chuỗi hiển thị."""
    token_str = tokenizer.id_to_token(token_id)
    return token_str if token_str is not None else f"<id:{token_id}>"


def _topk_from_logits(tokenizer, logits_row, chosen_id, k=TOP_K):
    probs = F.softmax(logits_row, dim=-1)
    topk_probs, topk_ids = torch.topk(probs, k)

    candidates = [
        (_format_token(tokenizer, tid.item()), prob.item())
        for tid, prob in zip(topk_ids, topk_probs)
    ]
    chosen_prob = probs[chosen_id].item()
    return candidates, chosen_prob


# =====================================================================
# ====== 3. THUẬT TOÁN BEAM SEARCH TỰ ĐỊNH NGHĨA LƯU VẾT LOGS =========
# =====================================================================

def explain_beam(tokenizer, model, device, sentence, top_k=TOP_K, num_beams=NUM_BEAMS, max_length=MAX_LENGTH):
    """
    Tự chạy thuật toán Beam Search thủ công và ghi lại vết xác suất (white-box trace) 
    của các phương án cạnh tranh trên đúng luồng giành chiến thắng cuối cùng.
    """
    bos_id = tokenizer.token_to_id("[BOS]")
    eos_id = tokenizer.token_to_id("[EOS]")
    pad_id = tokenizer.token_to_id("[PAD]")
    
    # Chuẩn bị đầu vào giống hệt lúc huấn luyện (độ dài 64)
    src_ids = padding_list(tokenizer.encode(str(sentence)).ids, 64, pad_id)
    src_tensor = torch.tensor([src_ids], dtype=torch.long).to(device)
    
    # Cấu trúc của 1 beam: (danh_sách_id_đầu_ra, điểm_log_cộng_dồn, lịch_sử_từng_bước)
    # Lịch sử từng bước lưu lại: {"parent_beam_idx": chỉ_mục_beam_cha, "logits": vector_logits_cpu}
    beams = [([bos_id], 0.0, [])]
    completed_beams = []
    
    for t in range(max_length - 1):
        candidates = []
        
        for beam_idx, (seq, score, history) in enumerate(beams):
            # Nếu beam đã chạm đích [EOS], đưa vào danh sách hoàn thành
            if seq[-1] == eos_id:
                completed_beams.append((seq, score, history))
                continue
                
            trg_tensor = torch.tensor([seq], dtype=torch.long).to(device)
            
            with torch.no_grad():
                logits = model(src_tensor, trg_tensor)
                next_token_logits = logits[0, -1, :]  # Lấy phân phối logits của vị trí tiếp theo
                
            log_probs = F.log_softmax(next_token_logits, dim=-1)
            
            # Lấy top-N ứng viên có điểm tích lũy tốt nhất
            topk_log_probs, topk_ids = torch.topk(log_probs, num_beams)
            
            for log_p, token_id in zip(topk_log_probs, topk_ids):
                token_id = token_id.item()
                new_seq = seq + [token_id]
                new_score = score + log_p.item()
                
                step_log = {
                    "parent_beam_idx": beam_idx,
                    "logits": next_token_logits.cpu()
                }
                new_history = history + [step_log]
                
                candidates.append((new_seq, new_score, new_history))
                
        if not candidates:
            break
            
        # Sắp xếp và chỉ giữ lại số lượng beam tối ưu (num_beams)
        candidates.sort(key=lambda x: x[1], reverse=True)
        beams = candidates[:num_beams]
        
        # Dừng sớm nếu tất cả các luồng hiện tại đều đã sinh ra token [EOS]
        if all(seq[-1] == eos_id for seq, _, _ in beams):
            break
            
    # Gộp các luồng đã hoàn thành và luồng hiện tại để tìm ra người chiến thắng
    all_completed = completed_beams + beams
    all_completed.sort(key=lambda x: x[1], reverse=True)
    
    best_seq, best_score, best_history = all_completed[0]
    
    # Giải mã chuỗi kết quả
    generated_ids = best_seq[1:]  # Loại bỏ BOS ở đầu
    
    steps = []
    prefix_ids = []
    for t, step_info in enumerate(best_history):
        chosen_id = generated_ids[t]
        logits_row = step_info["logits"]
        beam_index = step_info["parent_beam_idx"]
        
        candidates_list, chosen_prob = _topk_from_logits(tokenizer, logits_row, chosen_id, k=top_k)
        chosen_str = _format_token(tokenizer, chosen_id)
        
        steps.append({
            "step": t + 1,
            "beam_index": beam_index,
            "prefix_text": tokenizer.decode(prefix_ids),
            "candidates": candidates_list,
            "chosen_token": chosen_str,
            "chosen_prob": chosen_prob,
        })
        
        prefix_ids.append(chosen_id)
        if chosen_id == eos_id:
            break
            
    final_text = tokenizer.decode(generated_ids)
    
    return {
        "strategy": f"custom_beam_search_num_beams_{num_beams}",
        "input_sentence": sentence,
        "output_sentence": final_text,
        "steps": steps,
    }


def print_trace(trace):
    print("=" * 60)
    print(f"Câu gốc   : {trace['input_sentence']}")
    print(f"Chiến lược: {trace['strategy']}")
    print(f"Kết quả   : {trace['output_sentence']}")
    print("-" * 60)
    for step in trace["steps"]:
        prefix = step["prefix_text"] if step["prefix_text"] else "<bắt đầu>"
        cand_str = ", ".join(f"{tok} [{prob:.2f}]" for tok, prob in step["candidates"])
        beam_tag = f" (beam {step['beam_index']})" if "beam_index" in step else ""
        marker = "  -> đã chọn: " + f"{step['chosen_token']} [{step['chosen_prob']:.2f}]"
        print(f"Bước {step['step']}{beam_tag}: {prefix} --> {cand_str}")
        print(marker)
    print("=" * 60)


# ====== CÁC HÀM XUẤT JSON, VẼ CÂY & HEATMAP ======

def export_to_json(data, filepath):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"[Đã xuất JSON] -> {filepath}")


def plot_prob_heatmap(trace, filepath):
    steps = trace["steps"]
    if not steps:
        print("Không có bước sinh nào để vẽ heatmap.")
        return

    num_steps = len(steps)
    top_k = len(steps[0]["candidates"])

    prob_matrix = np.zeros((top_k, num_steps))
    cell_labels = []

    for step_idx, step in enumerate(steps):
        col_labels = []
        for cand_idx, (tok, prob) in enumerate(step["candidates"]):
            prob_matrix[cand_idx, step_idx] = prob
            is_chosen = (tok == step["chosen_token"])
            marker = " ★" if is_chosen else ""
            col_labels.append(f"{tok}\n({prob:.2f}){marker}")
        cell_labels.append(col_labels)

    cell_labels = list(map(list, zip(*cell_labels)))

    fig, ax = plt.subplots(figsize=(max(8, num_steps * 1.8), top_k * 1.2))
    im = ax.imshow(prob_matrix, cmap="YlGnBu", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(np.arange(num_steps))
    ax.set_yticks(np.arange(top_k))

    x_labels = [f"B. {s['step']}\n({s['chosen_token']})" for s in steps]
    y_labels = [f"Hạng {i+1}" for i in range(top_k)]

    ax.set_xticklabels(x_labels, fontsize=9)
    ax.set_yticklabels(y_labels, fontsize=9)

    for i in range(top_k):
        for j in range(num_steps):
            val = prob_matrix[i, j]
            color = "white" if val > 0.5 else "black"
            ax.text(j, i, cell_labels[i][j], ha="center", va="center", color=color, fontsize=8)

    ax.set_title(f"Heatmap xác suất của các ứng viên ({trace['strategy']})\nĐầu ra: {trace['output_sentence']}", fontsize=11, pad=15)
    fig.colorbar(im, ax=ax, shrink=0.7)
    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close()
    print(f"[Đã lưu Heatmap] -> {filepath}")


def plot_decision_tree(trace, filepath):
    steps = trace["steps"]
    if not steps:
        return

    G = nx.DiGraph()
    root_id = "bắt đầu"
    G.add_node(root_id, label="<bắt đầu>", level=0, is_chosen=True)

    parent_id = root_id
    for t, step in enumerate(steps):
        next_parent_id = None
        
        for cand_idx, (tok, prob) in enumerate(step["candidates"]):
            node_id = f"s{t}_c{cand_idx}_{tok}"
            is_chosen = (tok == step["chosen_token"])
            
            label = f"{tok}\n({prob:.2f})"
            G.add_node(node_id, label=label, level=t + 1, is_chosen=is_chosen)
            G.add_edge(parent_id, node_id)
            
            if is_chosen:
                next_parent_id = node_id

        if next_parent_id is None:
            node_id = f"s{t}_chosen_{step['chosen_token']}"
            label = f"{step['chosen_token']}\n({step['chosen_prob']:.2f})"
            G.add_node(node_id, label=label, level=t + 1, is_chosen=True)
            G.add_edge(parent_id, node_id)
            next_parent_id = node_id

        parent_id = next_parent_id

    pos = {}
    levels = {}
    for node, data in G.nodes(data=True):
        lvl = data["level"]
        levels.setdefault(lvl, []).append(node)

    for lvl, nodes in levels.items():
        num_nodes = len(nodes)
        for idx, node in enumerate(nodes):
            y = -(idx - (num_nodes - 1) / 2.0)
            pos[node] = (lvl * 2.5, y)

    node_colors = []
    for node, data in G.nodes(data=True):
        if data.get("is_chosen", False):
            node_colors.append("#a1e8af")
        else:
            node_colors.append("#ffb3b3")

    labels = {node: data["label"] for node, data in G.nodes(data=True)}

    plt.figure(figsize=(max(10, len(steps) * 2.2), 7))
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=1600, edgecolors="gray", linewidths=1)
    nx.draw_networkx_edges(G, pos, edge_color="gray", arrows=True, arrowsize=12, width=1.2)
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)

    plt.title(f"Cây quyết định giải mã ({trace['strategy']})\nMàu Xanh: Chuỗi chiến thắng | Màu Đỏ: Phương án rẽ thay thế", fontsize=11, pad=15)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close()
    print(f"[Đã lưu Đồ thị Cây] -> {filepath}")


def explain_translation(sentence, tokenizer, model, device, top_k=TOP_K, num_beams=NUM_BEAMS, max_length=MAX_LENGTH, verbose=True):
    beam_trace = explain_beam(tokenizer, model, device, sentence, top_k=top_k, num_beams=num_beams, max_length=max_length)

    if verbose:
        print_trace(beam_trace)

    return beam_trace


# =====================================================================
# ====== 4. HÀM CHẠY KIỂM THỬ CHƯƠNG TRÌNH CHÍNH ======================
# =====================================================================

if __name__ == "__main__":
    # Nạp mô hình Custom và Tokenizer riêng từ thư mục lưu trữ
    tokenizer, model, device = load_model(MODEL_DIR)

    test_sentences = [
        "Tôi 19 tuổi"
    ]

    for idx, sentence in enumerate(test_sentences):
        # Chạy phân tích giải mã bằng Beam Search tự định nghĩa
        beam_result = explain_translation(sentence, tokenizer, model, device, verbose=True)
        
        json_file = f"trace_beam_sentence_{idx}.json"
        
        # 1. Xuất dữ liệu ra JSON
        export_to_json(beam_result, json_file)

        # 2. Vẽ heatmap biểu diễn phân phối xác suất
        plot_prob_heatmap(beam_result, f"heatmap_beam_sentence_{idx}.png")

        # 3. Vẽ cây quyết định giải mã tương quan
        plot_decision_tree(beam_result, f"tree_beam_sentence_{idx}.png")