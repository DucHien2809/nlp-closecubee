"""
app_v2.py  —  VSL Decoder · Interactive Explainer
===================================================
3 tab chuyên sâu:
  Tab 1 · Beam Search Visualiser  (cây quyết định động)
  Tab 2 · Kiến trúc Transformer   (sơ đồ + attention heatmap realtime)
  Tab 3 · So sánh chiến lược      (Greedy vs Beam vs Sampling song song)

Chạy:  streamlit run app_v2.py
"""

from __future__ import annotations
import json, math
import streamlit as st
import streamlit.components.v1 as components
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ══════════════════════════════════════════════════════════════════════════════
# Cấu hình trang
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="VSL · Transformer Explainer",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# CSS toàn cục
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
:root{
  --bg:       #090e1a;
  --card:     #0f1829;
  --elevated: #162035;
  --border:   #1c3354;
  --cyan:     #0ea5e9;
  --teal:     #22d3ee;
  --green:    #34d399;
  --amber:    #fbbf24;
  --rose:     #f87171;
  --violet:   #a78bfa;
  --txt:      #eef4ff;
  --muted:    #6b90b0;
  --dim:      #364f6b;
}
.stApp{ background:var(--bg); font-family:'Inter',sans-serif; color:var(--txt); }
#MainMenu,footer,header{ visibility:hidden; }

/* ── Tabs ── */
[data-testid="stTabs"] [data-baseweb="tab-list"]{
  gap:4px; background:var(--card);
  padding:6px 8px; border-radius:12px;
  border:1px solid var(--border);
}
[data-testid="stTabs"] [data-baseweb="tab"]{
  border-radius:8px !important;
  padding:8px 20px !important;
  font-weight:600 !important;
  font-size:0.84rem !important;
  color:var(--muted) !important;
  background:transparent !important;
  transition:all .2s !important;
}
[data-testid="stTabs"] [aria-selected="true"]{
  background:linear-gradient(135deg,#0ea5e9,#22d3ee) !important;
  color:#050d1a !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"]{ background:var(--card) !important; border-right:1px solid var(--border) !important; }
[data-testid="stSidebar"] *{ color:var(--txt) !important; }
[data-testid="stSidebar"] label{ color:var(--muted) !important; font-size:.76rem !important; text-transform:uppercase; letter-spacing:.08em; }

/* ── Sliders ── */
[data-testid="stSlider"] [role="slider"]{ background:var(--cyan) !important; border-color:var(--cyan) !important; }

/* ── Inputs ── */
[data-testid="stTextInput"] input{
  background:var(--elevated) !important; border:1px solid var(--border) !important;
  border-radius:10px !important; color:var(--txt) !important;
  font-size:1rem !important; padding:.6rem 1rem !important;
}
[data-testid="stTextInput"] input:focus{ border-color:var(--cyan) !important; box-shadow:0 0 0 3px rgba(14,165,233,.18) !important; }

/* ── Buttons ── */
[data-testid="stButton"]>button{
  background:linear-gradient(135deg,#0ea5e9,#22d3ee) !important;
  color:#050d1a !important; font-weight:700 !important;
  border:none !important; border-radius:10px !important;
  padding:.6rem 1.6rem !important; transition:opacity .2s,transform .15s !important;
}
[data-testid="stButton"]>button:hover{ opacity:.85 !important; transform:translateY(-1px) !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar{ width:5px; height:5px; }
::-webkit-scrollbar-track{ background:var(--card); }
::-webkit-scrollbar-thumb{ background:var(--border); border-radius:3px; }
::-webkit-scrollbar-thumb:hover{ background:var(--cyan); }

hr{ border-color:var(--border) !important; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# Hero header
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
.hero{ padding:1.6rem 0 .8rem; display:flex; align-items:center; gap:1rem; }
.hero-icon{ font-size:2.6rem; filter:drop-shadow(0 0 16px #0ea5e9cc); }
.hero-title{
  margin:0; font-size:1.75rem; font-weight:700;
  background:linear-gradient(120deg,#eef4ff 20%,#22d3ee 60%,#0ea5e9 100%);
  -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;
}
.hero-sub{ margin:.2rem 0 0; font-size:.85rem; color:#6b90b0; }
.hero-badges{ margin:.5rem 0 0; display:flex; gap:.4rem; flex-wrap:wrap; }
.hb{ display:inline-block; padding:.18rem .65rem; border-radius:20px; font-size:.68rem;
     font-weight:600; letter-spacing:.06em; text-transform:uppercase; }
.hb-c{ background:rgba(14,165,233,.15); color:#38bdf8; border:1px solid rgba(14,165,233,.3); }
.hb-t{ background:rgba(34,211,238,.12); color:#67e8f9; border:1px solid rgba(34,211,238,.25); }
.hb-g{ background:rgba(52,211,153,.12); color:#6ee7b7; border:1px solid rgba(52,211,153,.25); }
.hb-v{ background:rgba(167,139,250,.12); color:#c4b5fd; border:1px solid rgba(167,139,250,.25); }
</style>
<div class="hero">
  <div class="hero-icon">🤟</div>
  <div>
    <h1 class="hero-title">VSL Transformer Explainer</h1>
    <p class="hero-sub">Trực quan hoá mô hình ViT5 dịch Tiếng Việt → Ngôn ngữ Ký hiệu Việt Nam (VSL)</p>
    <div class="hero-badges">
      <span class="hb hb-c">Beam Search</span>
      <span class="hb hb-t">ViT5 Fine-tuned</span>
      <span class="hb hb-g">VSL Grammar</span>
      <span class="hb hb-v">Attention Map</span>
    </div>
  </div>
</div>
<hr style="margin:.2rem 0 1rem">
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# Model loading
# ══════════════════════════════════════════════════════════════════════════════
MODEL_DIR = "./ViT5/vit5_sign_grammar_finetuned_10_epochs"

@st.cache_resource
def load_model_cached(model_dir=MODEL_DIR):
    try:
        tok = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
        mdl = AutoModelForSeq2SeqLM.from_pretrained(model_dir, output_attentions=True)
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mdl.to(dev).eval()
        return tok, mdl, dev
    except Exception as e:
        st.error(f"❌ Không nạp được mô hình từ `{model_dir}`: {e}")
        return None, None, None

# ══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════════════
SAMPLE_SENTENCES = [
    "Mẹ đang nấu cơm ở trong bếp",
    "Anh ấy học tiếng Anh mỗi ngày",
    "Tôi muốn uống một ly nước lạnh",
    "Con mèo đang ngủ trên ghế sofa",
    "Hôm nay trời mưa rất to",
]

with st.sidebar:
    st.markdown("""
    <div style="padding:.3rem 0 1rem">
      <p style="font-size:1rem;font-weight:700;color:#eef4ff;margin:0 0 .2rem">⚙️ Cấu hình</p>
      <p style="font-size:.75rem;color:#364f6b;margin:0">Tham số giải mã & hiển thị</p>
    </div>""", unsafe_allow_html=True)

    num_beams        = st.slider("Độ rộng Beam",   1, 10, 5, help="num_beams")
    top_k_candidates = st.slider("Top-K ứng viên", 2,  8, 5, help="Hiển thị top-k tại mỗi bước")
    anim_speed_ms    = st.slider("Tốc độ hoạt ảnh (ms/bước)", 400, 2000, 1000, step=100)

    st.markdown('<hr style="margin:.8rem 0">', unsafe_allow_html=True)
    st.markdown('<p style="font-size:.75rem;color:#6b90b0;font-weight:600;text-transform:uppercase;letter-spacing:.08em;margin:0 0 .5rem">📌 Câu mẫu nhanh</p>', unsafe_allow_html=True)
    preset = st.selectbox("Chọn câu mẫu", ["— tự nhập —"] + SAMPLE_SENTENCES, label_visibility="collapsed")

    st.markdown('<hr style="margin:.8rem 0">', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:.72rem;color:#364f6b;line-height:1.8">
      <p style="color:#6b90b0;font-weight:600;margin:0 0 .4rem;font-size:.74rem">📖 CHÚ THÍCH CÂY</p>
      <p style="margin:.2rem 0"><span style="display:inline-block;width:9px;height:9px;background:#34d399;border-radius:2px;margin-right:5px;vertical-align:middle"></span><b style="color:#eef4ff">Xanh lá</b> — token được chọn</p>
      <p style="margin:.2rem 0"><span style="display:inline-block;width:9px;height:9px;background:#f87171;border-radius:2px;margin-right:5px;vertical-align:middle"></span><b style="color:#eef4ff">Đỏ</b> — ứng viên bị loại</p>
      <p style="margin:.2rem 0"><span style="display:inline-block;width:9px;height:9px;background:#fbbf24;border-radius:2px;margin-right:5px;vertical-align:middle"></span><b style="color:#eef4ff">Vàng</b> — Greedy pick</p>
      <p style="margin:.2rem 0"><span style="display:inline-block;width:9px;height:9px;background:#a78bfa;border-radius:2px;margin-right:5px;vertical-align:middle"></span><b style="color:#eef4ff">Tím</b> — Sampling pick</p>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# Core decode helpers
# ══════════════════════════════════════════════════════════════════════════════
def fmt_token(tok, tid):
    t = tok.decode([tid], skip_special_tokens=False)
    return t if t.strip() else f"<id:{tid}>"

def topk_from_logits(tok, logits_row, chosen_id, k=5):
    probs = F.softmax(logits_row, dim=-1)
    kp, ki = torch.topk(probs, k)
    cands = [(fmt_token(tok, i.item()), p.item()) for i,p in zip(ki, kp)]
    return cands, probs[chosen_id].item()

def run_beam_trace(tok, mdl, dev, sentence, top_k=5, num_beams=5, max_len=128):
    inp = tok(sentence.strip(), return_tensors="pt", truncation=True, max_length=max_len)
    iid = inp["input_ids"].to(dev)
    atm = inp["attention_mask"].to(dev)
    with torch.no_grad():
        out = mdl.generate(iid, attention_mask=atm, max_length=max_len,
                           num_beams=num_beams, early_stopping=True,
                           no_repeat_ngram_size=2, output_scores=True,
                           return_dict_in_generate=True, num_return_sequences=1)
    seq = out.sequences[0]; scores = out.scores; bidx = out.beam_indices[0]
    gen = seq[1:]; steps=[]; prefix=[]
    for t, logits in enumerate(scores):
        bi = bidx[t].item()
        if bi < 0: break
        cid = gen[t].item()
        cands, cp = topk_from_logits(tok, logits[bi], cid, k=top_k)
        steps.append({"step":t+1,"beam_index":bi,
                      "prefix":tok.decode(prefix,skip_special_tokens=True),
                      "candidates":cands,"chosen_token":fmt_token(tok,cid),"chosen_prob":cp})
        prefix.append(cid)
        if cid == tok.eos_token_id: break
    return {"strategy":f"Beam Search (beams={num_beams})","input":sentence,
            "output":tok.decode(gen,skip_special_tokens=True,clean_up_tokenization_spaces=True),
            "steps":steps}

def run_greedy_trace(tok, mdl, dev, sentence, top_k=5, max_len=128):
    inp = tok(sentence.strip(), return_tensors="pt", truncation=True, max_length=max_len)
    iid = inp["input_ids"].to(dev); atm = inp["attention_mask"].to(dev)
    with torch.no_grad():
        out = mdl.generate(iid, attention_mask=atm, max_length=max_len,
                           num_beams=1, do_sample=False,
                           output_scores=True, return_dict_in_generate=True)
    seq = out.sequences[0]; scores = out.scores; gen = seq[1:]
    steps=[]; prefix=[]
    for t, logits in enumerate(scores):
        cid = gen[t].item()
        cands, cp = topk_from_logits(tok, logits[0], cid, k=top_k)
        steps.append({"step":t+1,"prefix":tok.decode(prefix,skip_special_tokens=True),
                      "candidates":cands,"chosen_token":fmt_token(tok,cid),"chosen_prob":cp})
        prefix.append(cid)
        if cid == tok.eos_token_id: break
    return {"strategy":"Greedy","input":sentence,
            "output":tok.decode(gen,skip_special_tokens=True,clean_up_tokenization_spaces=True),
            "steps":steps}

def run_sampling_trace(tok, mdl, dev, sentence, top_k_sample=50, top_p=0.9,
                       top_k_show=5, max_len=128):
    inp = tok(sentence.strip(), return_tensors="pt", truncation=True, max_length=max_len)
    iid = inp["input_ids"].to(dev); atm = inp["attention_mask"].to(dev)
    with torch.no_grad():
        out = mdl.generate(iid, attention_mask=atm, max_length=max_len,
                           num_beams=1, do_sample=True,
                           top_k=top_k_sample, top_p=top_p,
                           output_scores=True, return_dict_in_generate=True)
    seq = out.sequences[0]; scores = out.scores; gen = seq[1:]
    steps=[]; prefix=[]
    for t, logits in enumerate(scores):
        cid = gen[t].item()
        cands, cp = topk_from_logits(tok, logits[0], cid, k=top_k_show)
        steps.append({"step":t+1,"prefix":tok.decode(prefix,skip_special_tokens=True),
                      "candidates":cands,"chosen_token":fmt_token(tok,cid),"chosen_prob":cp})
        prefix.append(cid)
        if cid == tok.eos_token_id: break
    return {"strategy":"Top-p Sampling (p=0.9)","input":sentence,
            "output":tok.decode(gen,skip_special_tokens=True,clean_up_tokenization_spaces=True),
            "steps":steps}

def get_cross_attention(tok, mdl, dev, src_sentence, tgt_sentence, max_len=128):
    """Trả về ma trận cross-attention trung bình head ở lớp cuối decoder."""
    src = tok(src_sentence.strip(), return_tensors="pt", truncation=True, max_length=max_len)
    tgt = tok(tgt_sentence.strip(), return_tensors="pt", truncation=True, max_length=max_len)
    src_ids = src["input_ids"].to(dev)
    tgt_ids = tgt["input_ids"].to(dev)
    src_tokens = [tok.decode([i], skip_special_tokens=True) or f"[{i}]" for i in src_ids[0].tolist()]
    tgt_tokens = [tok.decode([i], skip_special_tokens=True) or f"[{i}]" for i in tgt_ids[0].tolist()]
    decoder_input_ids = mdl._shift_right(tgt_ids)
    with torch.no_grad():
        out = mdl(input_ids=src_ids,
                  attention_mask=src["attention_mask"].to(dev),
                  decoder_input_ids=decoder_input_ids,
                  output_attentions=True)
    # cross_attentions: tuple of (1, num_heads, tgt_len, src_len) per layer
    # lấy lớp cuối, trung bình tất cả heads
    last_ca = out.cross_attentions[-1][0]          # (heads, tgt, src)
    avg_ca  = last_ca.mean(dim=0).cpu().tolist()   # (tgt, src)
    return src_tokens, tgt_tokens, avg_ca

# ══════════════════════════════════════════════════════════════════════════════
# HTML generators
# ══════════════════════════════════════════════════════════════════════════════

def _tree_html(trace, anim_ms, chosen_color="#34d399", chosen_bg="#0d3d2a",
               chosen_border="#34d399", reject_color="#f87171",
               reject_bg="#3d0d0d", reject_border="#f87171"):
    steps = trace["steps"]; N = len(steps)
    input_txt = trace["input"]; output_txt = trace["output"]; strategy = trace["strategy"]
    nodes=[{"id":"root","label":"⟨bắt đầu⟩","level":0,
            "color":{"background":"#0d4f3c","border":"#34d399"},
            "font":{"size":12,"bold":True,"color":"#a7f3d0"}}]
    edges=[]; parent="root"
    for t, step in enumerate(steps):
        nxt=None
        for ci,(tok_str,prob) in enumerate(step["candidates"]):
            nid = f"s{t}_c{ci}_{tok_str.replace(' ','_').replace('<','').replace('>','')[:12]}"
            chosen = tok_str == step["chosen_token"]
            nodes.append({"id":nid,"label":f"{tok_str}\n{prob:.3f}","level":t+1,
                "color":{"background":chosen_bg if chosen else reject_bg,
                         "border":chosen_border if chosen else reject_border},
                "font":{"size":11,"bold":chosen,"color":chosen_color if chosen else reject_color}})
            edges.append({"from":parent,"to":nid,
                "color":{"color":chosen_border if chosen else "#2a3f58"},
                "width":2.5 if chosen else 1,"dashes":not chosen})
            if chosen: nxt=nid
        if nxt is None:
            nid=f"s{t}_fb"
            nodes.append({"id":nid,"label":f"{step['chosen_token']}\n{step['chosen_prob']:.3f}","level":t+1,
                "color":{"background":chosen_bg,"border":chosen_border},
                "font":{"size":11,"bold":True,"color":chosen_color}})
            edges.append({"from":parent,"to":nid,"color":{"color":chosen_border},"width":2.5})
            nxt=nid
        parent=nxt
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&family=JetBrains+Mono:wght@400;500&display=swap');
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:#090e1a;color:#eef4ff;font-family:'Inter',sans-serif}}
.info-bar{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;padding:10px 14px;background:#0f1829;border-bottom:1px solid #1c3354}}
.ic{{background:#162035;border:1px solid #1c3354;border-radius:8px;padding:7px 11px}}
.il{{font-size:.62rem;font-weight:600;text-transform:uppercase;letter-spacing:.09em;color:#364f6b;margin-bottom:3px}}
.iv{{font-size:.8rem;font-weight:600;color:#eef4ff;font-family:'JetBrains Mono',monospace;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}
.iv.g{{color:#34d399}}.iv.c{{color:#22d3ee}}
.ctrl{{display:flex;align-items:center;gap:8px;padding:8px 14px;background:#0a1120;border-bottom:1px solid #1c3354}}
.btn{{display:flex;align-items:center;gap:5px;padding:6px 14px;background:linear-gradient(135deg,#0ea5e9,#22d3ee);color:#050d1a;font-weight:700;font-size:.76rem;letter-spacing:.05em;border:none;border-radius:7px;cursor:pointer;transition:.2s}}
.btn:hover{{opacity:.85;transform:translateY(-1px)}}
.pw{{flex:1;display:flex;align-items:center;gap:8px}}
.pbg{{flex:1;height:4px;background:#162035;border-radius:2px;overflow:hidden}}
.pf{{height:100%;width:0%;background:linear-gradient(90deg,#0ea5e9,#34d399);border-radius:2px;transition:width .4s ease}}
.pl{{font-size:.7rem;color:#364f6b;font-family:'JetBrains Mono',monospace;white-space:nowrap;min-width:68px;text-align:right}}
.dot{{width:7px;height:7px;border-radius:50%;background:#22d3ee;box-shadow:0 0 6px #22d3ee;animation:pd 1.4s infinite;flex-shrink:0}}
.dot.done{{background:#34d399;box-shadow:0 0 6px #34d399;animation:none}}
@keyframes pd{{0%,100%{{opacity:1;transform:scale(1)}}50%{{opacity:.35;transform:scale(.65)}}}}
#net{{width:100%;height:430px;background:#090e1a}}
.leg{{display:flex;gap:14px;align-items:center;padding:7px 14px;background:#0a1120;border-top:1px solid #1c3354;flex-wrap:wrap}}
.li{{display:flex;align-items:center;gap:5px;font-size:.7rem;color:#6b90b0}}
.ld{{width:9px;height:9px;border-radius:2px;flex-shrink:0}}
</style></head><body>
<div class="info-bar">
  <div class="ic"><div class="il">📥 Câu đầu vào</div><div class="iv c" title="{input_txt}">{input_txt}</div></div>
  <div class="ic"><div class="il">📤 Kết quả VSL</div><div class="iv g" title="{output_txt}">{output_txt}</div></div>
  <div class="ic"><div class="il">🔧 Chiến lược</div><div class="iv" title="{strategy}">{strategy}</div></div>
</div>
<div class="ctrl">
  <button class="btn" onclick="restart()">
    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round"><path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/><path d="M3 3v5h5"/></svg>
    Phát lại
  </button>
  <div class="pw">
    <div class="pbg"><div class="pf" id="pf"></div></div>
    <div class="pl" id="pl">0/{N}</div>
    <div class="dot" id="dot"></div>
  </div>
</div>
<div id="net"></div>
<div class="leg">
  <div class="li"><div class="ld" style="background:{chosen_color};border:1px solid {chosen_color}"></div>Token được chọn</div>
  <div class="li"><div class="ld" style="background:{reject_color};border:1px solid {reject_color}"></div>Bị loại</div>
  <div class="li"><div class="ld" style="background:#a7f3d0;border:1px solid #34d399"></div>Gốc</div>
  <div class="li" style="margin-left:auto;color:#364f6b;font-style:italic">Số dưới mỗi node = xác suất</div>
</div>
<script>
const AN={json.dumps(nodes)},AE={json.dumps(edges)},MS={N},SPD={anim_ms};
var ns=new vis.DataSet([]),es=new vis.DataSet([]);
var net=new vis.Network(document.getElementById('net'),{{nodes:ns,edges:es}},{{
  nodes:{{shape:'box',margin:{{top:7,bottom:7,left:9,right:9}},shapeProperties:{{borderRadius:5}},shadow:{{enabled:true,color:'rgba(0,0,0,.6)',size:7}}}},
  edges:{{arrows:{{to:{{enabled:true,scaleFactor:.65}}}},smooth:{{type:'cubicBezier',forceDirection:'horizontal',roundness:.45}}}},
  layout:{{hierarchical:{{direction:'LR',sortMethod:'directed',nodeSpacing:85,levelSeparation:185}}}},
  physics:{{enabled:false}},
  interaction:{{hover:true,tooltipDelay:80}}
}});
let lvl=0,tid=null;
function sp(s){{const p=MS?Math.round(s/MS*100):100;document.getElementById('pf').style.width=p+'%';document.getElementById('pl').textContent=s+'/'+MS;}}
function restart(){{
  if(tid)clearInterval(tid);ns.clear();es.clear();lvl=0;sp(0);
  document.getElementById('dot').classList.remove('done');
  const r=AN.find(n=>n.id==='root');if(r)ns.add(r);
  setTimeout(()=>net.fit({{animation:false}}),50);
  setTimeout(()=>{{
    tid=setInterval(()=>{{
      lvl++;if(lvl>MS){{clearInterval(tid);sp(MS);document.getElementById('dot').classList.add('done');return;}}
      sp(lvl);
      const ln=AN.filter(n=>n.level===lvl);if(ln.length)ns.add(ln);
      const li=ln.map(n=>n.id),le=AE.filter(e=>li.includes(e.to));if(le.length)es.add(le);
      setTimeout(()=>net.fit({{animation:{{duration:500,easingFunction:'easeInOutQuad'}}}}),70);
    }},SPD);
  }},200);
}}
(function(){{
  const el=document.getElementById('net');
  let done=false;
  const ro=new ResizeObserver(entries=>{{
    for(const e of entries){{
      if(e.contentRect.height>10&&!done){{done=true;ro.disconnect();
        requestAnimationFrame(()=>setTimeout(restart,80));
      }}
    }}
  }});
  ro.observe(el);
  if(document.readyState==='complete'&&el.clientHeight>10){{done=true;ro.disconnect();setTimeout(restart,80);}}
}})();
</script></body></html>"""


def attention_heatmap_html(src_tokens, tgt_tokens, matrix):
    """Tạo heatmap cross-attention tương tác bằng Canvas."""
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&family=Inter:wght@400;600&display=swap');
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:#090e1a;color:#eef4ff;font-family:'Inter',sans-serif;padding:12px}}
h3{{font-size:.82rem;font-weight:600;color:#6b90b0;text-transform:uppercase;letter-spacing:.09em;margin-bottom:10px}}
.wrap{{overflow:auto;max-height:420px}}
table{{border-collapse:collapse;font-size:.72rem;font-family:'JetBrains Mono',monospace}}
th{{position:sticky;top:0;background:#0f1829;padding:4px 6px;color:#6b90b0;font-weight:500;white-space:nowrap;border-bottom:1px solid #1c3354;z-index:2}}
th.row-h{{position:sticky;left:0;z-index:3;background:#0f1829}}
td.label{{position:sticky;left:0;background:#0f1829;padding:4px 8px;color:#eef4ff;white-space:nowrap;border-right:1px solid #1c3354;font-weight:500}}
td.cell{{padding:0;width:34px;height:26px;text-align:center;font-size:.65rem;cursor:default;transition:outline .1s}}
td.cell:hover{{outline:2px solid #22d3ee;outline-offset:-1px}}
.tip{{display:none;position:fixed;background:#0f1829;border:1px solid #1c3354;border-radius:6px;padding:6px 10px;font-size:.72rem;color:#eef4ff;pointer-events:none;z-index:99;font-family:'JetBrains Mono',monospace}}
</style></head><body>
<h3>Cross-Attention — Lớp decoder cuối (trung bình tất cả heads)</h3>
<div class="wrap">
<table id="tbl"></table>
</div>
<div class="tip" id="tip"></div>
<script>
const src={json.dumps(src_tokens)};
const tgt={json.dumps(tgt_tokens)};
const mat={json.dumps(matrix)};

function heatColor(v){{
  // 0→deep blue, 1→bright cyan/green
  const r=Math.round(9+(14-9)*v*0);
  const g=Math.round(14+180*v);
  const b=Math.round(26+200*v);
  const a=0.15+v*0.85;
  return `rgba(${{Math.round(9+5*v)}},${{Math.round(80+140*v)}},${{Math.round(100+120*v)}},${{a.toFixed(2)}})`;
}}
function textColor(v){{return v>0.5?'#050d1a':'#eef4ff';}}

const tbl=document.getElementById('tbl');
// header row
let hr='<tr><th class="row-h">tgt \\ src</th>';
src.forEach(s=>hr+=`<th title="${{s}}">${{s.length>6?s.slice(0,6)+'…':s}}</th>`);
hr+='</tr>';
tbl.innerHTML=hr;

mat.forEach((row,ti)=>{{
  const max=Math.max(...row);
  let tr=`<tr><td class="label">${{tgt[ti]||''}}</td>`;
  row.forEach((v,si)=>{{
    const norm=max>0?v/max:0;
    const bg=heatColor(norm);
    const tc=textColor(norm);
    const pct=(v*100).toFixed(1);
    tr+=`<td class="cell" style="background:${{bg}};color:${{tc}}"
         onmousemove="showTip(event,'${{tgt[ti]}}→${{src[si]}}: ${{pct}}%')"
         onmouseleave="hideTip()">${{pct}}</td>`;
  }});
  tr+='</tr>';
  tbl.innerHTML+=tr;
}});

const tip=document.getElementById('tip');
function showTip(e,msg){{tip.style.display='block';tip.style.left=(e.clientX+12)+'px';tip.style.top=(e.clientY-28)+'px';tip.textContent=msg;}}
function hideTip(){{tip.style.display='none';}}
</script></body></html>"""


def transformer_arch_html():
    """Sơ đồ kiến trúc Encoder-Decoder — click vào khối để mở popup giải thích."""
    return """<!DOCTYPE html><html><head><meta charset="utf-8">
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
*{box-sizing:border-box;margin:0;padding:0}
html,body{background:#090e1a;color:#eef4ff;font-family:'Inter',sans-serif}

/* ── layout ── */
.page{display:flex;flex-direction:column;gap:0;min-height:100vh}
.hint{text-align:center;padding:8px 0 10px;font-size:.7rem;color:#364f6b;font-style:italic;letter-spacing:.02em}

/* ── diagram row ── */
.arch{display:flex;gap:20px;align-items:flex-start;justify-content:center;padding:0 14px 10px;flex-wrap:nowrap}
.col{display:flex;flex-direction:column;gap:6px;align-items:center;width:170px;flex-shrink:0}
.col-title{font-size:.68rem;font-weight:700;text-transform:uppercase;letter-spacing:.1em;
           padding:4px 14px;border-radius:20px;margin-bottom:2px;white-space:nowrap}
.enc-title{background:rgba(14,165,233,.18);color:#38bdf8;border:1px solid rgba(14,165,233,.3)}
.dec-title{background:rgba(167,139,250,.18);color:#c4b5fd;border:1px solid rgba(167,139,250,.3)}

/* ── blocks ── */
.block{
  width:160px;padding:9px 10px;border-radius:9px;text-align:center;
  cursor:pointer;transition:transform .15s,box-shadow .15s,outline .1s;
  font-size:.76rem;font-weight:600;user-select:none;
  -webkit-user-select:none;
}
.block:hover{transform:translateY(-2px);box-shadow:0 5px 18px rgba(0,0,0,.55)}
.block.active{outline:2px solid #22d3ee !important;outline-offset:2px;transform:translateY(-2px)}

.b-input {background:#0d2a40;border:1px solid #0ea5e9;color:#7dd3fc}
.b-embed {background:#0a2035;border:1px solid #0369a1;color:#38bdf8}
.b-posenc{background:#0a2035;border:1px solid #0284c7;color:#38bdf8}
.b-mha   {background:#0d1f3d;border:1px solid #3b82f6;color:#93c5fd}
.b-ffn   {background:#0d1a3d;border:1px solid #2563eb;color:#a5b4fc}
.b-norm  {background:#0a1830;border:1px solid #1d4ed8;color:#c7d2fe}
.b-cross {background:#1a0d3d;border:1px solid #7c3aed;color:#c4b5fd}
.b-dmha  {background:#1a0f3d;border:1px solid #6d28d9;color:#c4b5fd}
.b-lm    {background:#0d3d20;border:1px solid #059669;color:#6ee7b7}
.b-out   {background:#0d3d2a;border:1px solid #34d399;color:#a7f3d0}
.b-enc-out{background:#0a2240;border:2px solid #0ea5e9;color:#7dd3fc}

.arrow{color:#2a4a6a;font-size:1.1rem;line-height:1;text-align:center;pointer-events:none}
.arrow-h{color:#2a4a6a;font-size:1.4rem;margin-top:60px;pointer-events:none;flex-shrink:0}

.layer-box{
  border-radius:10px;padding:7px 6px;
  display:flex;flex-direction:column;gap:5px;align-items:center;width:170px
}
.layer-label{font-size:.6rem;color:#364f6b;text-transform:uppercase;letter-spacing:.07em;text-align:center;margin-bottom:1px}

/* ── legend ── */
.legend-row{
  display:flex;gap:12px;flex-wrap:wrap;justify-content:center;
  padding:8px 14px 12px;border-top:1px solid #1c3354;margin-top:2px
}
.leg-item{display:flex;align-items:center;gap:5px;font-size:.67rem;color:#6b90b0}
.leg-dot{width:9px;height:9px;border-radius:2px;flex-shrink:0}

/* ══════════════════════════════ MODAL / POPUP ══════════════════════════════ */
.modal-overlay{
  position:fixed;inset:0;background:rgba(4,8,16,.72);
  backdrop-filter:blur(3px);
  display:none;align-items:center;justify-content:center;
  z-index:1000;padding:20px;
  opacity:0;transition:opacity .18s ease;
}
.modal-overlay.show{display:flex;opacity:1}
.modal-box{
  background:#0f1829;border:1px solid #2a3f58;border-radius:14px;
  max-width:480px;width:100%;max-height:80vh;overflow-y:auto;
  padding:20px 22px 18px;
  box-shadow:0 20px 60px rgba(0,0,0,.65),0 0 0 1px rgba(34,211,238,.08);
  transform:translateY(8px) scale(.97);transition:transform .18s ease;
}
.modal-overlay.show .modal-box{transform:translateY(0) scale(1)}
.modal-head{display:flex;align-items:flex-start;justify-content:space-between;gap:10px;margin-bottom:10px}
.modal-title{font-size:1rem;font-weight:700;color:#22d3ee;line-height:1.4}
.modal-close{
  flex-shrink:0;width:26px;height:26px;border-radius:7px;border:1px solid #1c3354;
  background:#162035;color:#6b90b0;font-size:.85rem;cursor:pointer;
  display:flex;align-items:center;justify-content:center;transition:.15s;
}
.modal-close:hover{background:#1c3354;color:#eef4ff}
.modal-body{font-size:.82rem;color:#8aabb5;line-height:1.7;margin-bottom:10px}
.modal-formula{
  font-size:.74rem;color:#a78bfa;font-family:'JetBrains Mono',monospace;
  background:#0a1020;border-left:3px solid #4a2080;border-radius:4px;
  padding:8px 12px;white-space:pre-wrap;line-height:1.65
}
.modal-hint{margin-top:12px;font-size:.66rem;color:#364f6b;text-align:center;font-style:italic}
</style>
</head>
<body>
<div class="page">
  <div class="hint">💡 Click vào từng khối để xem popup giải thích</div>

  <div class="arch">
    <!-- ENCODER -->
    <div class="col">
      <div class="col-title enc-title">📥 Encoder</div>
      <div class="block b-input"   data-key="input">Câu Tiếng Việt<div style="font-size:.6rem;color:#364f6b;margin-top:2px">Mẹ đang nấu cơm...</div></div>
      <div class="arrow">↓</div>
      <div class="block b-embed"   data-key="embed">Token Embedding</div>
      <div class="arrow">↓</div>
      <div class="block b-posenc"  data-key="posenc">Positional Encoding</div>
      <div class="arrow">↓</div>
      <div class="layer-box" style="border:1px dashed #1c3354">
        <div class="layer-label">× N lớp</div>
        <div class="block b-mha"   data-key="mha">Multi-Head<br>Self-Attention</div>
        <div class="block b-norm"  data-key="norm">Add &amp; LayerNorm</div>
        <div class="block b-ffn"   data-key="ffn">Feed-Forward<br>Network</div>
        <div class="block b-norm"  data-key="norm">Add &amp; LayerNorm</div>
      </div>
      <div class="arrow">↓</div>
      <div class="block b-enc-out" data-key="enc_out">Encoder Output<br><span style="font-size:.6rem;color:#364f6b">(context vectors)</span></div>
    </div>

    <!-- center arrow -->
    <div class="arrow-h">→</div>

    <!-- DECODER -->
    <div class="col">
      <div class="col-title dec-title">🔄 Decoder</div>
      <div class="block b-input" style="border-color:#7c3aed;color:#c4b5fd" data-key="dec_input">Token VSL trước<br><span style="font-size:.6rem">(autoregressive)</span></div>
      <div class="arrow">↓</div>
      <div class="block b-embed"   data-key="embed">Token Embedding</div>
      <div class="arrow">↓</div>
      <div class="block b-posenc"  data-key="posenc">Positional Encoding</div>
      <div class="arrow">↓</div>
      <div class="layer-box" style="border:1px dashed #3b0764">
        <div class="layer-label">× N lớp</div>
        <div class="block b-dmha"  data-key="masked_mha">Masked Multi-Head<br>Self-Attention</div>
        <div class="block b-norm"  data-key="norm">Add &amp; LayerNorm</div>
        <div class="block b-cross" data-key="cross_attn">Cross-Attention<br><span style="font-size:.6rem;color:#7c3aed">← Encoder Output</span></div>
        <div class="block b-norm"  data-key="norm">Add &amp; LayerNorm</div>
        <div class="block b-ffn"   data-key="ffn">Feed-Forward<br>Network</div>
        <div class="block b-norm"  data-key="norm">Add &amp; LayerNorm</div>
      </div>
      <div class="arrow">↓</div>
      <div class="block b-lm"      data-key="lm_head">Linear + Softmax<br><span style="font-size:.6rem;color:#364f6b">(LM Head)</span></div>
      <div class="arrow">↓</div>
      <div class="block b-out"     data-key="output">Token VSL tiếp theo<br><span style="font-size:.6rem">Beam Search chọn</span></div>
    </div>
  </div>

  <!-- Legend -->
  <div class="legend-row">
    <div class="leg-item"><div class="leg-dot" style="background:#0ea5e9"></div>Encoder</div>
    <div class="leg-item"><div class="leg-dot" style="background:#7c3aed"></div>Decoder</div>
    <div class="leg-item"><div class="leg-dot" style="background:#34d399"></div>Output</div>
    <div class="leg-item"><div class="leg-dot" style="background:#a78bfa"></div>Cross-Attention</div>
    <div class="leg-item"><div class="leg-dot" style="background:#3b82f6"></div>Self-Attention</div>
  </div>
</div>

<!-- ── Modal popup ── -->
<div class="modal-overlay" id="overlay">
  <div class="modal-box" id="modalBox">
    <div class="modal-head">
      <div class="modal-title" id="mTitle"></div>
      <button class="modal-close" id="mClose" aria-label="Đóng">✕</button>
    </div>
    <div class="modal-body" id="mBody"></div>
    <div class="modal-formula" id="mFormula"></div>
    <div class="modal-hint">Nhấn Esc hoặc click ra ngoài để đóng</div>
  </div>
</div>

<script>
var INFO = {
  input:{
    title:"📥 Đầu vào — Câu Tiếng Việt",
    body:"Câu tiếng Việt tự nhiên được tokenizer ViT5 tách thành các subword bằng SentencePiece. Mỗi token nhận một ID số nguyên trong vocabulary ~32 000 từ. Quá trình tokenize xử lý cả dấu thanh điệu tiếng Việt.",
    formula:"Ví dụ: 'Mẹ đang nấu cơm' → ['▁Mẹ', '▁đang', '▁nấu', '▁cơm', '</s>']"
  },
  embed:{
    title:"🔢 Token Embedding",
    body:"Mỗi token ID được tra bảng embedding E ∈ ℝ^(V×d_model) để lấy vector 512 chiều. Vector này mang thông tin ngữ nghĩa của từ, học được qua quá trình pre-training ViT5 trên dữ liệu tiếng Việt lớn.",
    formula:"e(t) = E[token_id] × √d_model      (d_model = 512)"
  },
  posenc:{
    title:"📍 Positional Encoding",
    body:"Transformer xử lý song song tất cả token nên không biết thứ tự. ViT5 (T5-based) dùng relative position bias — học được, được thêm vào điểm attention thay vì cộng thẳng vào embedding như BERT. Linh hoạt hơn với độ dài câu khác nhau.",
    formula:"h₀ = Dropout(e(t))   +   learned_rel_pos_bias trong Attention"
  },
  mha:{
    title:"👁 Multi-Head Self-Attention (Encoder)",
    body:"Mỗi token trong câu tiếng Việt 'nhìn' toàn bộ các token khác để học ngữ cảnh. 8 heads chạy song song — mỗi head chuyên học một loại quan hệ: head 1 cú pháp, head 2 coreference, head 3 ngữ nghĩa... Kết quả các head được nối lại.",
    formula:"Attention(Q,K,V) = softmax( QKᵀ/√d_k + bias_pos ) · V\\nMultiHead = Concat(head₁…head₈) · Wᴼ\\n(d_k = d_model/h = 64)"
  },
  masked_mha:{
    title:"🔒 Masked Self-Attention (Decoder)",
    body:"Giống Self-Attention encoder nhưng áp mask tam giác trên: token ở vị trí t CHỈ thấy token 0…t-1, không nhìn sang phải. Đây là điều kiện tiên quyết của autoregressive generation — mô hình không được 'nhìn trước' token tương lai khi training.",
    formula:"score[i,j] = QᵢKⱼᵀ/√d_k + pos_bias[i,j]\\nmask[i,j]  = -∞  nếu j > i  (causal mask)\\n             0    nếu j ≤ i"
  },
  cross_attn:{
    title:"🔗 Cross-Attention — Cầu nối Enc→Dec",
    body:"Đây là khối quan trọng nhất! Query (Q) từ decoder hỏi: 'Tôi đang sinh token VSL này, tôi cần tập trung vào phần nào của câu tiếng Việt?' Key/Value (K,V) đến từ encoder output — mang toàn bộ thông tin câu nguồn. Heatmap Tab 2 trực quan hoá trọng số attention này.",
    formula:"Q = h_dec · Wᵠ   (từ decoder)\\nK = h_enc · Wᴷ   (từ encoder output)\\nV = h_enc · Wᵛ   (từ encoder output)\\nAttn = softmax(QKᵀ/√d_k) · V"
  },
  ffn:{
    title:"⚡ Feed-Forward Network (FFN)",
    body:"Sau khi attention tổng hợp ngữ cảnh, FFN xử lý từng vị trí token độc lập (position-wise). Lớp ẩn d_ff=2048 (×4 d_model) cho phép mô hình học các biến đổi phi tuyến phức tạp — tương đương 'bộ nhớ factual' của Transformer.",
    formula:"FFN(x) = ReLU(x·W₁ + b₁)·W₂ + b₂\\nd_model=512 → d_ff=2048 → d_model=512"
  },
  norm:{
    title:"⚖ Add & LayerNorm",
    body:"Residual connection (Add) cho phép gradient chảy thẳng qua skip connection, tránh vanishing gradient khi stack nhiều lớp. LayerNorm chuẩn hoá theo chiều feature (không phải batch), ổn định training và tăng tốc hội tụ.",
    formula:"y = LayerNorm(x + Sublayer(x))\\nLayerNorm(x) = γ·(x−μ)/σ + β\\n(μ, σ tính trên d_model=512 chiều)"
  },
  lm_head:{
    title:"📊 LM Head — Linear + Softmax",
    body:"Vector decoder cuối (512 chiều) được chiếu lên không gian vocabulary ~32 000 chiều qua một lớp Linear (thường chia sẻ trọng số với embedding). Softmax cho phân phối xác suất. Beam Search lấy top-K token từ phân phối này ở mỗi bước.",
    formula:"logits = h_dec · Eᵀ + b   (E chia sẻ với embedding)\\nP(token | context) = softmax(logits / temperature)\\nBeam chọn top-B theo log P tích luỹ"
  },
  output:{
    title:"✅ Token VSL được chọn — Autoregressive",
    body:"Tại bước t, Beam Search duy trì B=num_beams chuỗi tốt nhất theo log-prob tích luỹ. Token được chọn ghép vào output rồi đưa trở lại decoder làm input bước t+1. Quá trình dừng khi gặp </s> (EOS) hoặc đạt max_length.",
    formula:"score(y₁…yₜ) = Σ_{i=1}^{t} log P(yᵢ | y<i, x)\\nBeam giữ B chuỗi có score cao nhất\\nDừng khi yₜ = </s>"
  },
  enc_out:{
    title:"🔵 Encoder Output — Context Vectors",
    body:"Sau N lớp encoder, mỗi token tiếng Việt được biểu diễn bởi một vector 512 chiều mang đầy đủ ngữ cảnh của toàn câu (bidirectional). Tập hợp các vector này là 'bộ nhớ' mà decoder sẽ truy vấn qua Cross-Attention ở mỗi bước giải mã.",
    formula:"H_enc = [h₁, h₂, ..., h_n] ∈ ℝ^{n×512}\\nn = số token câu tiếng Việt\\nDecoder dùng H_enc làm K, V trong Cross-Attention"
  },
  dec_input:{
    title:"🔄 Input Decoder — Autoregressive",
    body:"Trong inference, decoder nhận token đã sinh ở bước trước làm input hiện tại. Chuỗi bắt đầu bằng token BOS (beginning-of-sequence). Ở mỗi bước, toàn bộ chuỗi đã sinh được đưa vào (nhưng Masked Attention đảm bảo không nhìn về tương lai).",
    formula:"t=0: input = [<BOS>]\\nt=1: input = [<BOS>, y₁]\\nt=2: input = [<BOS>, y₁, y₂]\\n...\\nDừng khi output = </s>"
  }
};

var activeEl = null;
var overlay  = document.getElementById('overlay');
var modalBox = document.getElementById('modalBox');

function openModal(el, key) {
  if (activeEl) activeEl.classList.remove('active');
  activeEl = el;
  el.classList.add('active');

  var d = INFO[key] || {title: key, body: '', formula: ''};
  document.getElementById('mTitle').textContent = d.title;
  document.getElementById('mBody').textContent  = d.body;
  var f = document.getElementById('mFormula');
  if (d.formula) { f.style.display = 'block'; f.textContent = d.formula; }
  else           { f.style.display = 'none'; }

  overlay.classList.add('show');
}

function closeModal() {
  overlay.classList.remove('show');
  if (activeEl) { activeEl.classList.remove('active'); activeEl = null; }
}

document.addEventListener('DOMContentLoaded', function() {
  var blocks = document.querySelectorAll('.block[data-key]');
  blocks.forEach(function(el) {
    var key = el.getAttribute('data-key');
    el.addEventListener('click', function(e) {
      e.stopPropagation();
      openModal(el, key);
    });
  });

  // Click ra ngoài modal-box (trên overlay) để đóng
  overlay.addEventListener('click', function(e) {
    if (e.target === overlay) closeModal();
  });
  document.getElementById('mClose').addEventListener('click', closeModal);

  // Esc để đóng
  document.addEventListener('keydown', function(e) {
    if (e.key === 'Escape' && overlay.classList.contains('show')) closeModal();
  });
});
</script>
</body></html>"""


def strategy_compare_html(beam_trace, greedy_trace, samp_trace, anim_ms):
    """3 cây song song trong 1 iframe."""
    def build_nodes_edges(trace, chosen_color, chosen_bg, chosen_border,
                          reject_color, reject_bg, reject_border):
        steps = trace["steps"]
        nodes=[{"id":"root","label":"⟨start⟩","level":0,
                "color":{"background":"#0d3020","border":chosen_border},
                "font":{"size":10,"bold":True,"color":chosen_color}}]
        edges=[]; parent="root"
        for t,step in enumerate(steps):
            nxt=None
            for ci,(tk,prob) in enumerate(step["candidates"]):
                nid=f"s{t}_c{ci}_{tk.replace(' ','_').replace('<','').replace('>','')[:10]}"
                chosen=tk==step["chosen_token"]
                nodes.append({"id":nid,"label":f"{tk}\n{prob:.2f}","level":t+1,
                    "color":{"background":chosen_bg if chosen else reject_bg,
                             "border":chosen_border if chosen else reject_border},
                    "font":{"size":9,"bold":chosen,"color":chosen_color if chosen else reject_color}})
                edges.append({"from":parent,"to":nid,
                    "color":{"color":chosen_border if chosen else "#1c2f45"},
                    "width":2 if chosen else 0.8,"dashes":not chosen})
                if chosen: nxt=nid
            if nxt is None:
                nid=f"s{t}_fb"
                nodes.append({"id":nid,"label":f"{step['chosen_token']}\n{step['chosen_prob']:.2f}","level":t+1,
                    "color":{"background":chosen_bg,"border":chosen_border},
                    "font":{"size":9,"bold":True,"color":chosen_color}})
                edges.append({"from":parent,"to":nid,"color":{"color":chosen_border},"width":2})
                nxt=nid
            parent=nxt
        return nodes, edges

    bn,be = build_nodes_edges(beam_trace,  "#34d399","#0d3d2a","#34d399","#f87171","#3d0d0d","#f87171")
    gn,ge = build_nodes_edges(greedy_trace,"#fbbf24","#3d2a00","#fbbf24","#f87171","#3d0d0d","#f87171")
    sn,se = build_nodes_edges(samp_trace,  "#a78bfa","#1e0d3d","#a78bfa","#f87171","#3d0d0d","#f87171")

    b_steps = len(beam_trace["steps"])
    g_steps = len(greedy_trace["steps"])
    s_steps = len(samp_trace["steps"])
    max_steps = max(b_steps, g_steps, s_steps)

    return f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=JetBrains+Mono:wght@400&display=swap');
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:#090e1a;color:#eef4ff;font-family:'Inter',sans-serif}}
.topbar{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;padding:10px;background:#0f1829;border-bottom:1px solid #1c3354}}
.tc{{border-radius:8px;padding:8px 10px;text-align:center}}
.tc-b{{background:#0d1f0d;border:1px solid #34d399}}.tc-g{{background:#1f1700;border:1px solid #fbbf24}}.tc-s{{background:#0f0d1f;border:1px solid #a78bfa}}
.tc-name{{font-size:.7rem;font-weight:700;text-transform:uppercase;letter-spacing:.07em;margin-bottom:3px}}
.tc-out{{font-size:.72rem;font-family:'JetBrains Mono',monospace;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}
.tc-b .tc-name{{color:#34d399}}.tc-g .tc-name{{color:#fbbf24}}.tc-s .tc-name{{color:#a78bfa}}
.tc-b .tc-out{{color:#6ee7b7}}.tc-g .tc-out{{color:#fde68a}}.tc-s .tc-out{{color:#c4b5fd}}
.ctrl{{display:flex;align-items:center;gap:8px;padding:7px 10px;background:#0a1120;border-bottom:1px solid #1c3354}}
.btn{{padding:5px 12px;background:linear-gradient(135deg,#0ea5e9,#22d3ee);color:#050d1a;font-weight:700;font-size:.72rem;border:none;border-radius:6px;cursor:pointer}}
.pw{{flex:1;display:flex;align-items:center;gap:7px}}
.pbg{{flex:1;height:3px;background:#162035;border-radius:2px;overflow:hidden}}
.pf{{height:100%;width:0%;background:linear-gradient(90deg,#0ea5e9,#34d399);transition:width .4s}}
.pl{{font-size:.67rem;color:#364f6b;font-family:'JetBrains Mono',monospace;white-space:nowrap}}
.nets{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:0}}
.net-wrap{{position:relative;border-right:1px solid #1c3354;height:430px;overflow:hidden}}
.net-wrap:last-child{{border-right:none}}
.net-label{{position:absolute;top:6px;left:50%;transform:translateX(-50%);font-size:.65rem;font-weight:700;text-transform:uppercase;letter-spacing:.08em;padding:2px 8px;border-radius:10px;z-index:10;pointer-events:none}}
.lb-b{{background:rgba(52,211,153,.15);color:#34d399;border:1px solid rgba(52,211,153,.3)}}
.lb-g{{background:rgba(251,191,36,.15);color:#fbbf24;border:1px solid rgba(251,191,36,.3)}}
.lb-s{{background:rgba(167,139,250,.15);color:#a78bfa;border:1px solid rgba(167,139,250,.3)}}
/* Chiều cao tuyệt đối (px) — không phụ thuộc % của cha để tránh canvas bị khởi tạo với kích thước 0 */
.network-div{{width:100%;height:430px;background:#090e1a}}
.net-empty{{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;color:#364f6b;font-size:.72rem;font-style:italic;pointer-events:none}}
</style></head><body>
<div class="topbar">
  <div class="tc tc-b"><div class="tc-name">🌲 Beam Search</div><div class="tc-out" title="{beam_trace['output']}">{beam_trace['output']}</div></div>
  <div class="tc tc-g"><div class="tc-name">⚡ Greedy</div><div class="tc-out" title="{greedy_trace['output']}">{greedy_trace['output']}</div></div>
  <div class="tc tc-s"><div class="tc-name">🎲 Top-p Sampling</div><div class="tc-out" title="{samp_trace['output']}">{samp_trace['output']}</div></div>
</div>
<div class="ctrl">
  <button class="btn" onclick="restart()">▶ Phát lại</button>
  <div class="pw">
    <div class="pbg"><div class="pf" id="pf"></div></div>
    <div class="pl" id="pl">0/{max_steps}</div>
  </div>
</div>
<div class="nets">
  <div class="net-wrap"><div class="net-label lb-b">Beam Search</div><div class="network-div" id="nb"></div><div class="net-empty" id="nb-empty">Đang dựng cây…</div></div>
  <div class="net-wrap"><div class="net-label lb-g">Greedy</div><div class="network-div" id="ng"></div><div class="net-empty" id="ng-empty">Đang dựng cây…</div></div>
  <div class="net-wrap"><div class="net-label lb-s">Sampling</div><div class="network-div" id="ns"></div><div class="net-empty" id="ns-empty">Đang dựng cây…</div></div>
</div>
<script>
const BN={json.dumps(bn)},BE={json.dumps(be)};
const GN={json.dumps(gn)},GE={json.dumps(ge)};
const SN={json.dumps(sn)},SE={json.dumps(se)};
const MAX={max_steps},SPD={anim_ms};

/* ── Shared options ── */
const NET_OPTS={{
  nodes:{{
    shape:'box',
    margin:{{top:6,bottom:6,left:9,right:9}},
    shapeProperties:{{borderRadius:5}},
    shadow:{{enabled:true,color:'rgba(0,0,0,.55)',size:6}}
  }},
  edges:{{
    arrows:{{to:{{enabled:true,scaleFactor:.55}}}},
    smooth:{{type:'cubicBezier',forceDirection:'horizontal',roundness:.42}}
  }},
  layout:{{
    hierarchical:{{
      direction:'LR',
      sortMethod:'directed',
      nodeSpacing:70,
      levelSeparation:145,
      blockShifting:true,
      edgeMinimization:true
    }}
  }},
  physics:{{enabled:false}},
  interaction:{{hover:true,zoomView:true,dragView:true}}
}};

let nets=[], lvl=0, tid=null, ready=false;

/* Tạo network NGAY (đồng bộ) — không chờ ResizeObserver mới khởi tạo,
   vì .network-div đã có height cố định bằng CSS (430px) nên container
   luôn có kích thước thật tại thời điểm này. */
function mkNet(id,allN,allE){{
  const ns=new vis.DataSet([]),es=new vis.DataSet([]);
  const container=document.getElementById(id);
  const net=new vis.Network(container,{{nodes:ns,edges:es}},NET_OPTS);
  return {{net,ns,es,allN,allE,emptyEl:document.getElementById(id+'-empty')}};
}}

function setProg(s){{
  const p=MAX?Math.round(s/MAX*100):100;
  document.getElementById('pf').style.width=p+'%';
  document.getElementById('pl').textContent=s+'/'+MAX;
}}

function revealLevel(o, lvl){{
  const ln=o.allN.filter(n=>n.level===lvl);
  if(ln.length) o.ns.add(ln);
  const li=ln.map(n=>n.id);
  const le=o.allE.filter(e=>li.includes(e.to));
  if(le.length) o.es.add(le);
  if(o.emptyEl) o.emptyEl.style.display='none';
  const visibleIds=o.ns.getIds();
  o.net.fit({{nodes:visibleIds,animation:{{duration:380,easingFunction:'easeInOutQuad'}}}});
}}

function restart(){{
  if(tid)clearInterval(tid);
  if(!nets.length) return;
  nets.forEach(o=>{{o.ns.clear();o.es.clear(); if(o.emptyEl) o.emptyEl.style.display='flex';}});
  lvl=0; setProg(0);
  /* Hiện root ngay lập tức ở cả 3 cây */
  nets.forEach(o=>revealLevel(o,0));
  tid=setInterval(()=>{{
    lvl++;
    if(lvl>MAX){{clearInterval(tid);setProg(MAX);return;}}
    setProg(lvl);
    nets.forEach(o=>revealLevel(o,lvl));
  }},SPD);
}}

function initAll(){{
  if(ready) return;
  ready=true;
  nets=[mkNet('nb',BN,BE),mkNet('ng',GN,GE),mkNet('ns',SN,SE)];
  /* Đợi 1 frame để canvas vẽ xong khung rồi mới phát animation */
  requestAnimationFrame(()=>setTimeout(restart,120));
}}

if(document.readyState==='complete'){{
  initAll();
}}else{{
  window.addEventListener('load',initAll);
  /* Phòng trường hợp 'load' không bắn (một số iframe sandbox) */
  setTimeout(initAll,800);
}}
</script></body></html>"""


# ══════════════════════════════════════════════════════════════════════════════
# Main UI — 3 Tabs
# ══════════════════════════════════════════════════════════════════════════════
tokenizer, model, device = load_model_cached()

if tokenizer is None:
    st.warning("⚠️ Không tìm thấy mô hình. Kiểm tra lại đường dẫn `MODEL_DIR`.")
    st.stop()

tab1, tab2, tab3 = st.tabs([
    "🌳  Beam Search — Cây quyết định",
    "🧠  Kiến trúc Transformer",
    "⚡  So sánh chiến lược giải mã",
])

# ── Tab 1 : Beam Search ────────────────────────────────────────────────────
with tab1:
    st.markdown("""
    <p style="font-size:.76rem;color:#6b90b0;text-transform:uppercase;letter-spacing:.08em;font-weight:600;margin-bottom:.6rem">
    📝 Câu Tiếng Việt cần chuyển đổi
    </p>""", unsafe_allow_html=True)

    default_val = preset if preset != "— tự nhập —" else "Mẹ đang nấu cơm ở trong bếp"
    c1, c2 = st.columns([5, 1])
    with c1:
        sent1 = st.text_input("s1", value=default_val, label_visibility="collapsed",
                              placeholder="Nhập câu tiếng Việt...")
    with c2:
        run1 = st.button("▶  Dịch & Vẽ", key="run1", use_container_width=True)

    if run1 and sent1.strip():
        with st.spinner("Đang giải mã..."):
            st.session_state["trace1"] = run_beam_trace(
                tokenizer, model, device, sent1, top_k=top_k_candidates, num_beams=num_beams)

    if "trace1" in st.session_state:
        tr = st.session_state["trace1"]
        st.markdown(f"""
        <div style="margin:.8rem 0;padding:.8rem 1.1rem;background:#0d1f0d;border:1px solid #1a4d1a;
                    border-left:4px solid #34d399;border-radius:10px;display:flex;align-items:center;gap:.8rem">
          <span style="font-size:1.2rem">✅</span>
          <div>
            <p style="margin:0;font-size:.68rem;color:#364f6b;text-transform:uppercase;letter-spacing:.08em;font-weight:600">Kết quả VSL</p>
            <p style="margin:.2rem 0 0;font-size:1rem;font-weight:700;color:#34d399;font-family:'JetBrains Mono',monospace">{tr['output']}</p>
          </div>
        </div>""", unsafe_allow_html=True)

        st.markdown('<p style="margin:.8rem 0 .4rem;font-size:.72rem;color:#6b90b0;text-transform:uppercase;letter-spacing:.08em;font-weight:600">🌳 Cây quyết định — sinh trưởng từ trái sang phải</p>', unsafe_allow_html=True)
        components.html(_tree_html(tr, anim_ms=anim_speed_ms), height=580, scrolling=False)

# ── Tab 2 : Kiến trúc Transformer ─────────────────────────────────────────
with tab2:
    col_arch, col_heat = st.columns([1, 1], gap="medium")

    with col_arch:
        st.markdown('<p style="font-size:.72rem;color:#6b90b0;text-transform:uppercase;letter-spacing:.08em;font-weight:600;margin-bottom:.5rem">🏗 Sơ đồ kiến trúc — Click để xem giải thích</p>', unsafe_allow_html=True)
        components.html(transformer_arch_html(), height=680, scrolling=True)

    with col_heat:
        st.markdown('<p style="font-size:.72rem;color:#6b90b0;text-transform:uppercase;letter-spacing:.08em;font-weight:600;margin-bottom:.5rem">🔥 Cross-Attention Heatmap — realtime</p>', unsafe_allow_html=True)

        default2 = preset if preset != "— tự nhập —" else "Mẹ đang nấu cơm ở trong bếp"
        c3, c4 = st.columns([5, 1])
        with c3:
            sent2 = st.text_input("s2", value=default2, label_visibility="collapsed",
                                  placeholder="Nhập câu tiếng Việt...")
        with c4:
            run2 = st.button("🔍 Phân tích", key="run2", use_container_width=True)

        if run2 and sent2.strip():
            with st.spinner("Đang trích xuất cross-attention..."):
                # Dịch trước để lấy target
                tr2 = run_greedy_trace(tokenizer, model, device, sent2)
                tgt_text = tr2["output"]
                try:
                    src_tok, tgt_tok, matrix = get_cross_attention(
                        tokenizer, model, device, sent2, tgt_text)
                    st.session_state["attn"] = (src_tok, tgt_tok, matrix, sent2, tgt_text)
                except Exception as e:
                    st.error(f"Lỗi trích xuất attention: {e}")

        if "attn" in st.session_state:
            src_tok, tgt_tok, matrix, src_s, tgt_s = st.session_state["attn"]
            st.markdown(f"""
            <div style="display:flex;gap:8px;margin-bottom:8px;font-size:.72rem;font-family:'JetBrains Mono',monospace">
              <div style="background:#0d2040;border:1px solid #1c3354;border-radius:6px;padding:4px 10px;color:#38bdf8">📥 {src_s}</div>
              <div style="color:#364f6b;align-self:center">→</div>
              <div style="background:#0d3020;border:1px solid #1a4d1a;border-radius:6px;padding:4px 10px;color:#6ee7b7">📤 {tgt_s}</div>
            </div>""", unsafe_allow_html=True)
            components.html(attention_heatmap_html(src_tok, tgt_tok, matrix), height=460, scrolling=True)
        else:
            st.markdown("""
            <div style="margin-top:1rem;padding:2rem;background:#0f1829;border:1px dashed #1c3354;border-radius:12px;text-align:center;color:#364f6b">
              <p style="font-size:1.5rem;margin-bottom:.6rem">🔥</p>
              <p style="font-size:.82rem">Nhập câu và nhấn <b style="color:#6b90b0">Phân tích</b> để xem<br>ma trận cross-attention giữa token nguồn và đích</p>
            </div>""", unsafe_allow_html=True)

# ── Tab 3 : So sánh chiến lược ────────────────────────────────────────────
with tab3:
    st.markdown("""
    <div style="margin-bottom:.8rem;padding:.7rem 1rem;background:#0f1829;border:1px solid #1c3354;border-radius:10px;font-size:.8rem;color:#6b90b0;line-height:1.65">
      Chạy cùng một câu với 3 chiến lược giải mã song song để so sánh trực quan sự khác biệt.<br>
      <span style="color:#34d399;font-weight:600">Beam Search</span> — khám phá B chuỗi tốt nhất song song ·
      <span style="color:#fbbf24;font-weight:600">Greedy</span> — chọn token xác suất cao nhất mỗi bước ·
      <span style="color:#a78bfa;font-weight:600">Top-p Sampling</span> — lấy mẫu ngẫu nhiên từ nucleus
    </div>""", unsafe_allow_html=True)

    default3 = preset if preset != "— tự nhập —" else "Mẹ đang nấu cơm ở trong bếp"
    c5, c6 = st.columns([5, 1])
    with c5:
        sent3 = st.text_input("s3", value=default3, label_visibility="collapsed",
                              placeholder="Nhập câu tiếng Việt...")
    with c6:
        run3 = st.button("▶  So sánh", key="run3", use_container_width=True)

    if run3 and sent3.strip():
        with st.spinner("Đang chạy 3 chiến lược giải mã..."):
            bt = run_beam_trace(tokenizer, model, device, sent3,
                                top_k=top_k_candidates, num_beams=num_beams)
            gt = run_greedy_trace(tokenizer, model, device, sent3, top_k=top_k_candidates)
            sp = run_sampling_trace(tokenizer, model, device, sent3,
                                    top_k_show=top_k_candidates)
            st.session_state["cmp"] = (bt, gt, sp)

    if "cmp" in st.session_state:
        bt, gt, sp = st.session_state["cmp"]

        # Bảng tóm tắt
        beam_steps = len(bt["steps"]); greedy_steps = len(gt["steps"]); samp_steps = len(sp["steps"])
        avg_beam_prob  = sum(s["chosen_prob"] for s in bt["steps"]) / max(beam_steps,1)
        avg_greedy_prob= sum(s["chosen_prob"] for s in gt["steps"]) / max(greedy_steps,1)
        avg_samp_prob  = sum(s["chosen_prob"] for s in sp["steps"]) / max(samp_steps,1)

        st.markdown(f"""
        <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;margin-bottom:1rem">
          <div style="background:#0d1f0d;border:1px solid #1a4d1a;border-radius:10px;padding:.8rem 1rem">
            <div style="font-size:.65rem;color:#364f6b;text-transform:uppercase;letter-spacing:.08em;font-weight:600;margin-bottom:.3rem">🌲 Beam Search</div>
            <div style="font-size:.85rem;font-weight:700;color:#34d399;font-family:'JetBrains Mono',monospace;margin-bottom:.2rem">{bt['output']}</div>
            <div style="font-size:.68rem;color:#6b90b0">{beam_steps} bước · xác suất TB {avg_beam_prob:.3f}</div>
          </div>
          <div style="background:#1f1700;border:1px solid #3d2a00;border-radius:10px;padding:.8rem 1rem">
            <div style="font-size:.65rem;color:#364f6b;text-transform:uppercase;letter-spacing:.08em;font-weight:600;margin-bottom:.3rem">⚡ Greedy</div>
            <div style="font-size:.85rem;font-weight:700;color:#fbbf24;font-family:'JetBrains Mono',monospace;margin-bottom:.2rem">{gt['output']}</div>
            <div style="font-size:.68rem;color:#6b90b0">{greedy_steps} bước · xác suất TB {avg_greedy_prob:.3f}</div>
          </div>
          <div style="background:#0f0d1f;border:1px solid #2e1f5e;border-radius:10px;padding:.8rem 1rem">
            <div style="font-size:.65rem;color:#364f6b;text-transform:uppercase;letter-spacing:.08em;font-weight:600;margin-bottom:.3rem">🎲 Top-p Sampling</div>
            <div style="font-size:.85rem;font-weight:700;color:#a78bfa;font-family:'JetBrains Mono',monospace;margin-bottom:.2rem">{sp['output']}</div>
            <div style="font-size:.68rem;color:#6b90b0">{samp_steps} bước · xác suất TB {avg_samp_prob:.3f}</div>
          </div>
        </div>""", unsafe_allow_html=True)

        components.html(strategy_compare_html(bt, gt, sp, anim_speed_ms), height=600, scrolling=False)