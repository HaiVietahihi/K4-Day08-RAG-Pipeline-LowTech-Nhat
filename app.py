"""
Trợ Lý Hướng Dẫn Viên Du Lịch Thông Minh — RAG Chatbot
Streamlit app với GSAP animations + premium UI design.

Chạy:
    streamlit run app.py
"""

import os
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="Trợ Lý Du Lịch Việt Nam 🇻🇳",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# GSAP + CUSTOM CSS (taste-skill inspired premium design)
# =============================================================================

GSAP_HEADER = """
<script src="https://cdnjs.cloudflare.com/ajax/libs/gsap/3.12.5/gsap.min.js"></script>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Playfair+Display:wght@600;700&display=swap');

  :root {
    --bg-primary:    #0f1117;
    --bg-card:       rgba(255,255,255,0.04);
    --bg-card-hover: rgba(255,255,255,0.08);
    --border:        rgba(255,255,255,0.08);
    --accent-green:  #22c55e;
    --accent-teal:   #0ea5e9;
    --accent-amber:  #f59e0b;
    --text-primary:  #f1f5f9;
    --text-muted:    #94a3b8;
    --gradient-hero: linear-gradient(135deg, #0f1117 0%, #1a2535 50%, #0d1f2d 100%);
    --glow-green:    0 0 20px rgba(34,197,94,0.3);
    --glow-teal:     0 0 20px rgba(14,165,233,0.3);
    --radius:        16px;
    --radius-sm:     10px;
    --font-body:     'Inter', sans-serif;
    --font-display:  'Playfair Display', serif;
  }

  /* ---- HERO HEADER ---- */
  .hero-header {
    background: var(--gradient-hero);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 32px 40px 28px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
    opacity: 0;
    transform: translateY(-20px);
  }
  .hero-header::before {
    content: '';
    position: absolute;
    top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: radial-gradient(ellipse at 60% 40%, rgba(14,165,233,0.12) 0%, transparent 60%),
                radial-gradient(ellipse at 20% 80%, rgba(34,197,94,0.08) 0%, transparent 50%);
    pointer-events: none;
  }
  .hero-title {
    font-family: var(--font-display);
    font-size: 2.2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #f1f5f9 0%, #94d2bd 50%, #48cae4 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 8px;
    line-height: 1.2;
  }
  .hero-subtitle {
    font-family: var(--font-body);
    color: var(--text-muted);
    font-size: 0.95rem;
    font-weight: 400;
    margin: 0;
    letter-spacing: 0.01em;
  }
  .hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(34,197,94,0.12);
    border: 1px solid rgba(34,197,94,0.25);
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.75rem;
    font-weight: 600;
    color: var(--accent-green);
    margin-bottom: 12px;
    letter-spacing: 0.05em;
    text-transform: uppercase;
  }

  /* ---- STATUS PILLS ---- */
  .status-bar {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
    margin-top: 16px;
  }
  .status-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 5px 14px;
    font-size: 0.76rem;
    font-weight: 500;
    color: var(--text-muted);
    font-family: var(--font-body);
    opacity: 0;
    transform: translateX(-10px);
  }
  .status-pill .dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    flex-shrink: 0;
  }
  .dot-green  { background: var(--accent-green); box-shadow: var(--glow-green); }
  .dot-teal   { background: var(--accent-teal);  box-shadow: var(--glow-teal); }
  .dot-amber  { background: var(--accent-amber); }
  .dot-gray   { background: #475569; }

  /* ---- SUGGESTION CHIPS ---- */
  .chip-container {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin: 8px 0 4px;
  }
  .chip {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 6px 14px;
    font-size: 0.8rem;
    color: var(--text-muted);
    cursor: pointer;
    transition: all 0.2s ease;
    font-family: var(--font-body);
  }
  .chip:hover {
    background: var(--bg-card-hover);
    border-color: rgba(14,165,233,0.4);
    color: var(--text-primary);
    transform: translateY(-1px);
  }

  /* ---- SOURCE CARD ---- */
  .source-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    padding: 14px 16px;
    margin-bottom: 10px;
    transition: all 0.25s ease;
    opacity: 0;
    transform: translateY(8px);
  }
  .source-card:hover {
    background: var(--bg-card-hover);
    border-color: rgba(14,165,233,0.3);
    transform: translateY(-1px);
    box-shadow: var(--glow-teal);
  }
  .source-tag {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 6px;
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
  }
  .tag-legal { background: rgba(245,158,11,0.15); color: var(--accent-amber); }
  .tag-news  { background: rgba(14,165,233,0.15);  color: var(--accent-teal); }
  .tag-pageindex { background: rgba(34,197,94,0.12); color: var(--accent-green); }

  /* ---- SCORE BAR ---- */
  .score-bar-wrap {
    height: 4px;
    background: rgba(255,255,255,0.08);
    border-radius: 2px;
    margin: 8px 0 4px;
    overflow: hidden;
  }
  .score-bar {
    height: 100%;
    border-radius: 2px;
    background: linear-gradient(90deg, var(--accent-teal), var(--accent-green));
    transition: width 0.8s cubic-bezier(0.4,0,0.2,1);
  }

  /* ---- TYPING INDICATOR ---- */
  .typing-indicator {
    display: flex;
    align-items: center;
    gap: 5px;
    padding: 12px 16px;
  }
  .typing-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: var(--accent-teal);
    animation: typing-bounce 1.2s infinite ease-in-out;
  }
  .typing-dot:nth-child(2) { animation-delay: 0.2s; }
  .typing-dot:nth-child(3) { animation-delay: 0.4s; }
  @keyframes typing-bounce {
    0%, 80%, 100% { transform: scale(0.6); opacity: 0.4; }
    40%            { transform: scale(1.1); opacity: 1; }
  }

  /* ---- STREAMLIT OVERRIDES ---- */
  .stChatMessage {
    border-radius: var(--radius) !important;
    border: 1px solid var(--border) !important;
    background: var(--bg-card) !important;
  }
  .stButton > button {
    border-radius: 10px !important;
    font-family: var(--font-body) !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    transition: all 0.2s !important;
  }
  .stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.3) !important;
  }
  div[data-testid="stSidebar"] {
    background: #0d1117 !important;
    border-right: 1px solid var(--border) !important;
  }
  .stSlider > div > div > div {
    background: linear-gradient(90deg, var(--accent-teal), var(--accent-green)) !important;
  }
</style>

<div class="hero-header" id="heroHeader">
  <div class="hero-badge">⚡ RAG Pipeline Active</div>
  <h1 class="hero-title">🧭 Trợ Lý Du Lịch Việt Nam</h1>
  <p class="hero-subtitle">Hệ thống AI hỗ trợ lập kế hoạch du lịch · Hybrid Retrieval + LLM Generation</p>
  <div class="status-bar">
    <span class="status-pill" id="pill1"><span class="dot dot-green"></span>ChromaDB Online</span>
    <span class="status-pill" id="pill2"><span class="dot dot-teal"></span>Semantic Search</span>
    <span class="status-pill" id="pill3"><span class="dot dot-teal"></span>BM25 Lexical</span>
    <span class="status-pill" id="pill4"><span class="dot dot-green"></span>RRF Reranking</span>
    <span class="status-pill" id="pill5"><span class="dot dot-amber"></span>PageIndex Fallback</span>
  </div>
</div>

<script>
// GSAP entrance animations
gsap.to("#heroHeader", {
  opacity: 1, y: 0, duration: 0.7, ease: "power3.out", delay: 0.1
});
gsap.to(".status-pill", {
  opacity: 1, x: 0, duration: 0.5, ease: "power2.out",
  stagger: 0.1, delay: 0.5
});

// Animate source cards when they appear
function animateSourceCards() {
  gsap.utils.toArray(".source-card").forEach((card, i) => {
    if (card.dataset.animated) return;
    card.dataset.animated = "1";
    gsap.to(card, { opacity:1, y:0, duration:0.4, delay: i * 0.08, ease:"power2.out" });
  });
}
// Poll for new source cards
setInterval(animateSourceCards, 500);
</script>
"""

# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 8px 0 16px;">
      <div style="font-size:2.5rem;">🧭</div>
      <div style="font-family:'Inter',sans-serif; font-size:1rem; font-weight:600; color:#f1f5f9;">
        Du Lịch AI Assistant
      </div>
      <div style="font-size:0.75rem; color:#64748b; margin-top:4px;">
        Powered by Hybrid RAG Pipeline
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.markdown("**💡 Câu hỏi gợi ý**")
    suggestions = [
        "🏔️ Lịch trình Hà Giang 3 ngày 2 đêm?",
        "🌊 Đà Nẵng có gì vui chơi?",
        "🍜 Đặc sản Hải Phòng là gì?",
        "🏛️ Di tích ở Huế nào đẹp nhất?",
        "🌿 Kiên Giang có điểm du lịch nào?",
        "🦀 Cà Mau nên ăn gì?",
    ]
    for s in suggestions:
        if st.button(s, use_container_width=True, key=f"sug_{s[:15]}"):
            st.session_state["pending_query"] = s

    st.divider()
    st.markdown("**⚙️ Cài đặt**")
    top_k = st.slider("Số chunks retrieval (top_k)", 3, 10, 5,
                      help="Số đoạn văn bản sẽ được dùng làm context cho LLM")
    show_score_threshold = st.checkbox("Hiện ngưỡng fallback", value=False)
    if show_score_threshold:
        st.caption("🔀 Fallback threshold: `cosine < 0.48`")
        st.caption("Dưới ngưỡng → PageIndex Vectorless")

    st.divider()
    st.markdown("**🏗️ Kiến trúc**")
    st.markdown("""
    <div style="font-size:0.75rem; color:#64748b; line-height:1.8;">
    <code>Query</code><br>
    ↓ Semantic Search (Task 5)<br>
    ↓ BM25 Lexical (Task 6)<br>
    ↓ RRF Rerank (Task 7)<br>
    ↓ PageIndex Fallback (Task 8)<br>
    ↓ LLM Generation (Task 10)
    </div>
    """, unsafe_allow_html=True)

    if st.button("🗑️ Xoá lịch sử chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# =============================================================================
# SESSION STATE
# =============================================================================

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

# =============================================================================
# SOURCE RENDERER FUNCTION (define before use)
# =============================================================================

def _render_sources(sources: list[dict]):
    if not sources:
        return
    with st.expander(f"📚 Tài liệu tham khảo ({len(sources)} nguồn)", expanded=False):
        for i, src in enumerate(sources, 1):
            meta = src.get("metadata", {})
            src_name = meta.get("source", meta.get("source_file", f"Nguồn {i}"))
            doc_type = meta.get("type", src.get("source", "unknown"))
            score = src.get("score", 0)
            cosine = src.get("cosine_score")
            retrieval_src = src.get("source", "hybrid")

            tag_class = "tag-legal" if "legal" in doc_type else (
                "tag-pageindex" if retrieval_src == "pageindex" else "tag-news"
            )
            tag_label = doc_type.upper() if doc_type != "unknown" else retrieval_src.upper()
            score_pct = min(int(score * 3000), 100)
            if cosine:
                score_pct = min(int(cosine * 100), 100)

            st.markdown(f"""
            <div class="source-card">
              <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                <span style="font-size:0.85rem; font-weight:600; color:#f1f5f9;">[{i}] {src_name}</span>
                <span class="source-tag {tag_class}">{tag_label}</span>
              </div>
              <div class="score-bar-wrap">
                <div class="score-bar" style="width:{score_pct}%"></div>
              </div>
              <div style="font-size:0.72rem; color:#64748b; margin-bottom:8px;">
                score: {score:.4f}{f" | cosine: {cosine:.3f}" if cosine else ""}
              </div>
              <div style="font-size:0.82rem; color:#94a3b8; line-height:1.6; background:rgba(0,0,0,0.2);
                          border-radius:8px; padding:10px; font-family:'Inter',sans-serif;">
                {src.get('content', '')[:280]}...
              </div>
            </div>
            """, unsafe_allow_html=True)


# =============================================================================
# MAIN AREA
# =============================================================================

# Inject GSAP header
st.components.v1.html(GSAP_HEADER, height=200)

# Welcome message nếu chưa có chat
if not st.session_state.messages:
    st.markdown("""
    <div style="
      background: rgba(14,165,233,0.05);
      border: 1px solid rgba(14,165,233,0.15);
      border-radius: 14px;
      padding: 24px 28px;
      margin: 8px 0 24px;
      font-family: 'Inter', sans-serif;
    ">
      <div style="font-size:1.05rem; font-weight:600; color:#f1f5f9; margin-bottom:10px;">
        👋 Xin chào! Tôi là Trợ Lý Du Lịch AI
      </div>
      <div style="color:#94a3b8; font-size:0.9rem; line-height:1.7;">
        Tôi có thể giúp bạn:<br>
        • 🗺️ Lập lịch trình du lịch theo ngày<br>
        • 🍽️ Gợi ý đặc sản và nhà hàng địa phương<br>
        • 🏨 Tư vấn nơi lưu trú và phương tiện di chuyển<br>
        • 📸 Điểm đến nổi tiếng và kinh nghiệm du lịch<br><br>
        <em>Chọn câu hỏi gợi ý bên trái hoặc nhập câu hỏi phía dưới để bắt đầu!</em>
      </div>
    </div>
    """, unsafe_allow_html=True)

# Hiển thị lịch sử chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            _render_sources(msg["sources"])


# =============================================================================
# QUERY HANDLING
# =============================================================================

user_input = st.chat_input("Hỏi về du lịch Việt Nam... (Hà Giang, Huế, Đà Nẵng, Hải Phòng...)")
query = user_input or st.session_state.get("pending_query")

if query:
    st.session_state.pending_query = None

    # Hiển thị user message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Generate response
    with st.chat_message("assistant"):
        # Typing indicator
        typing_placeholder = st.empty()
        typing_placeholder.markdown("""
        <div class="typing-indicator">
          <div class="typing-dot"></div>
          <div class="typing-dot"></div>
          <div class="typing-dot"></div>
          <span style="font-size:0.8rem; color:#64748b; margin-left:4px;">Đang tìm kiếm tài liệu...</span>
        </div>
        """, unsafe_allow_html=True)

        try:
            from src.task10_generation import generate_with_citation
            response = generate_with_citation(query, top_k=top_k)
            answer = response.get("answer", "Không thể trả lời.")
            sources = response.get("sources", [])
            retrieval_src = response.get("retrieval_source", "hybrid")
        except NotImplementedError:
            answer = "⚠️ **Task 10 chưa được implement.**"
            sources = []
            retrieval_src = "none"
        except Exception as e:
            answer = f"❌ **Lỗi pipeline:** {e}"
            sources = []
            retrieval_src = "error"

        typing_placeholder.empty()

        # Hiển thị retrieval badge
        badge_color = "#22c55e" if retrieval_src == "hybrid" else "#f59e0b"
        badge_icon = "🔍" if retrieval_src == "hybrid" else "📑"
        st.markdown(f"""
        <div style="display:inline-flex; align-items:center; gap:6px; background:rgba(255,255,255,0.04);
                    border:1px solid rgba(255,255,255,0.08); border-radius:8px; padding:3px 10px;
                    margin-bottom:10px; font-size:0.72rem; color:{badge_color}; font-weight:500;">
          {badge_icon} via {retrieval_src.upper()} · {len(sources)} nguồn
        </div>
        """, unsafe_allow_html=True)

        st.markdown(answer)

        if sources:
            _render_sources(sources)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
    })
    st.rerun()
