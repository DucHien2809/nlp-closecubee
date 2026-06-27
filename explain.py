"""
explain.py
==========
Giải thích white-box quá trình model sinh ra câu ngữ pháp VSL từng bước một:
tại mỗi bước decode, hiển thị top-k từ ứng viên kèm xác suất, và từ nào
thực sự được chọn dựa trên thuật toán Beam Search.

Hỗ trợ:
  - beam   : beam search (num_beams=5 mặc định), truy lại được xác suất của 
             các ứng viên cạnh tranh tại mỗi bước trên đúng "beam" cuối cùng đã thắng.
  - Xuất dữ liệu phân tích ra JSON.
  - Vẽ đồ thị cây quyết định (Beam Decision Tree) lưu thành PNG.
  - Vẽ Heatmap xác suất các ứng viên qua từng bước sinh lưu thành PNG.
"""

import json
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Thư viện hỗ trợ trực quan hóa
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# ====== Cấu hình ======
MODEL_DIR = "./ViT5/vit5_sign_grammar_finetuned_10_epochs"
TOP_K = 5            # số ứng viên hiển thị tại mỗi bước
MAX_LENGTH = 128
NUM_BEAMS = 5


def load_model(model_dir=MODEL_DIR):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device


def _format_token(tokenizer, token_id):
    """Giải mã 1 token_id thành chuỗi hiển thị, loại bỏ khoảng trắng dư."""
    text = tokenizer.decode([token_id], skip_special_tokens=False)
    return text if text.strip() != "" else f"<id:{token_id}>"


def _topk_from_logits(tokenizer, logits_row, chosen_id, k=TOP_K):
    """
    logits_row: tensor 1D shape (vocab_size,) — logits CHƯA softmax tại 1 bước.
    chosen_id : id của token thực sự được chọn ở bước này.
    """
    probs = F.softmax(logits_row, dim=-1)
    topk_probs, topk_ids = torch.topk(probs, k)

    candidates = [
        (_format_token(tokenizer, tid.item()), prob.item())
        for tid, prob in zip(topk_ids, topk_probs)
    ]
    chosen_prob = probs[chosen_id].item()
    return candidates, chosen_prob


def explain_beam(tokenizer, model, device, sentence, top_k=TOP_K, num_beams=NUM_BEAMS, max_length=MAX_LENGTH):
    """
    Decode bằng beam search và trả về structured trace của từng bước decode,
    theo đúng "đường đi" của beam thắng cuối cùng (best sequence).
    """
    inputs = tokenizer(sentence.strip(), return_tensors="pt", truncation=True, max_length=max_length)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True,
            no_repeat_ngram_size=2,
            output_scores=True,
            return_dict_in_generate=True,
            num_return_sequences=1,
        )

    sequence = outputs.sequences[0]
    scores = outputs.scores
    beam_indices = outputs.beam_indices[0]
    generated_ids = sequence[1:]

    steps = []
    prefix_ids = []
    for t, logits in enumerate(scores):
        beam_idx = beam_indices[t].item()
        if beam_idx < 0:
            break

        chosen_id = generated_ids[t].item()
        logits_row = logits[beam_idx]

        candidates, chosen_prob = _topk_from_logits(tokenizer, logits_row, chosen_id, k=top_k)
        chosen_str = _format_token(tokenizer, chosen_id)

        steps.append({
            "step": t + 1,
            "beam_index": beam_idx,
            "prefix_text": tokenizer.decode(prefix_ids, skip_special_tokens=True),
            "candidates": candidates,
            "chosen_token": chosen_str,
            "chosen_prob": chosen_prob,
        })

        prefix_ids.append(chosen_id)
        if chosen_id == tokenizer.eos_token_id:
            break

    final_text = tokenizer.decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

    return {
        "strategy": f"beam_search_num_beams_{num_beams}",
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
    """Xuất toàn bộ cấu trúc dữ liệu phân tích ra file JSON."""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"[Đã xuất JSON] -> {filepath}")


def plot_prob_heatmap(trace, filepath):
    """
    Vẽ heatmap biểu diễn xác suất của các từ ứng viên qua từng bước sinh.
    """
    steps = trace["steps"]
    if not steps:
        print("Không có bước sinh nào để vẽ heatmap.")
        return

    num_steps = len(steps)
    top_k = len(steps[0]["candidates"])

    # Chuẩn bị ma trận dữ liệu xác suất (top_k x num_steps)
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

    # Chuyển đổi nhãn ô để khớp với định dạng hàng/cột của pyplot
    cell_labels = list(map(list, zip(*cell_labels)))

    fig, ax = plt.subplots(figsize=(max(8, num_steps * 1.8), top_k * 1.2))
    im = ax.imshow(prob_matrix, cmap="YlGnBu", aspect="auto", vmin=0, vmax=1)

    # Thiết lập tọa độ và nhãn
    ax.set_xticks(np.arange(num_steps))
    ax.set_yticks(np.arange(top_k))

    x_labels = [f"B. {s['step']}\n({s['chosen_token']})" for s in steps]
    y_labels = [f"Hạng {i+1}" for i in range(top_k)]

    ax.set_xticklabels(x_labels, fontsize=9)
    ax.set_yticklabels(y_labels, fontsize=9)

    # Điền chữ và xác suất vào các ô
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
    """
    Vẽ đồ thị cây mô tả tiến trình lựa chọn của mô hình (Winning Path và các nhánh rẽ).
    """
    steps = trace["steps"]
    if not steps:
        return

    G = nx.DiGraph()
    
    # Gốc của cây quyết định
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

        # Dự phòng trường hợp token được chọn không nằm trong top-K ứng viên hiển thị
        if next_parent_id is None:
            node_id = f"s{t}_chosen_{step['chosen_token']}"
            label = f"{step['chosen_token']}\n({step['chosen_prob']:.2f})"
            G.add_node(node_id, label=label, level=t + 1, is_chosen=True)
            G.add_edge(parent_id, node_id)
            next_parent_id = node_id

        parent_id = next_parent_id

    # Định vị các node theo chiều ngang (trục X là bước, trục Y là các phương án rẽ)
    pos = {}
    levels = {}
    for node, data in G.nodes(data=True):
        lvl = data["level"]
        levels.setdefault(lvl, []).append(node)

    for lvl, nodes in levels.items():
        num_nodes = len(nodes)
        for idx, node in enumerate(nodes):
            # Căn giữa các lựa chọn theo chiều dọc
            y = -(idx - (num_nodes - 1) / 2.0)
            pos[node] = (lvl * 2.5, y)

    # Chia màu sắc cho các node đã chọn (Xanh nhạt) và node bị loại (Đỏ nhạt)
    node_colors = []
    for node, data in G.nodes(data=True):
        if data.get("is_chosen", False):
            node_colors.append("#a1e8af")  # Xanh pastel
        else:
            node_colors.append("#ffb3b3")  # Đỏ pastel

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
    """
    Hàm tiện ích: chạy chiến lược beam search cho 1 câu, in ra console
    nếu verbose=True, và trả về dữ liệu trace.
    """
    beam_trace = explain_beam(tokenizer, model, device, sentence, top_k=top_k, num_beams=num_beams, max_length=max_length)

    if verbose:
        print_trace(beam_trace)

    return beam_trace


if __name__ == "__main__":
    # Nạp mô hình
    tokenizer, model, device = load_model(MODEL_DIR)

    test_sentences = [
        "Tôi 19 tuổi"
    ]

    for idx, sentence in enumerate(test_sentences):
        # Chạy phân tích bước dịch bằng Beam Search
        beam_result = explain_translation(sentence, tokenizer, model, device, verbose=True)
        
        # Định danh file lưu theo chỉ mục của câu thử nghiệm
        json_file = f"trace_beam_sentence_{idx}.json"
        
        # 1. Xuất dữ liệu ra JSON
        export_to_json(beam_result, json_file)

        # 2. Vẽ heatmap cho Beam Search
        plot_prob_heatmap(beam_result, f"heatmap_beam_sentence_{idx}.png")

        # 3. Vẽ cây quyết định cho Beam Search
        plot_decision_tree(beam_result, f"tree_beam_sentence_{idx}.png")