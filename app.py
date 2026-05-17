"""
Streamlit frontend for the Multi-Opinion Debate Engine.

Place this file in the same directory as debate_engine.py, then run:
    streamlit run app.py
"""

import re
import json
import streamlit as st

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Debate Engine",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500&display=swap');

  html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: #0b0c10;
    color: #e0e0e0;
  }
  #MainMenu, footer, header { visibility: hidden; }

  .hero-title {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 2.6rem;
    letter-spacing: -0.02em;
    background: linear-gradient(135deg, #f5a623 0%, #e83e3e 50%, #a855f7 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.15;
    margin-bottom: 0;
  }
  .hero-sub {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    color: #4b5563;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    margin-top: 0.3rem;
    margin-bottom: 1.8rem;
  }

  /* Cards */
  .card {
    background: #13151c;
    border: 1px solid #1f2333;
    border-radius: 10px;
    padding: 1.3rem 1.5rem;
    margin-bottom: 1.1rem;
  }
  .card-label {
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 0.72rem;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    margin-bottom: 0.8rem;
  }
  .label-tech   { color: #38bdf8; }
  .label-prac   { color: #4ade80; }
  .label-critic { color: #fb923c; }
  .label-synth  { color: #c084fc; }
  .label-meta   { color: #94a3b8; }

  .prose {
    font-size: 0.9rem;
    line-height: 1.8;
    color: #cbd5e1;
    white-space: pre-wrap;
  }

  /* Confidence bar */
  .conf-wrap { display:flex; align-items:center; gap:1rem; }
  .conf-bar-bg {
    flex:1; height:7px; background:#1f2333;
    border-radius:99px; overflow:hidden;
  }
  .conf-bar-fill { height:100%; border-radius:99px; }
  .conf-pct {
    font-family:'Syne',sans-serif;
    font-weight:700; font-size:1.15rem; min-width:3.2rem; text-align:right;
  }

  /* Iteration pills */
  .pill {
    display:inline-flex; align-items:center; gap:0.35rem;
    font-family:'IBM Plex Mono',monospace; font-size:0.7rem;
    padding:0.22rem 0.65rem; border-radius:99px;
    margin:0.15rem 0.15rem 0.15rem 0;
  }
  .pill-ok   { background:#14291e; color:#4ade80; border:1px solid #166534; }
  .pill-warn { background:#2d1a0e; color:#fb923c; border:1px solid #7c2d12; }

  /* Trace rows */
  .trace-row {
    font-family:'IBM Plex Mono',monospace; font-size:0.7rem;
    color:#6b7280; padding:0.22rem 0;
    border-bottom:1px solid #1a1d28;
  }
  .trace-row:last-child { border-bottom:none; }

  /* Stat cards */
  .stat-card {
    background:#13151c; border:1px solid #1f2333;
    border-radius:10px; padding:1rem 1.3rem;
    margin-bottom:1rem;
  }
  .stat-card .stat-val {
    font-family:'Syne',sans-serif; font-weight:800;
    font-size:2rem; color:#e0e0e0; line-height:1.1;
  }
  .stat-card .stat-sub {
    font-family:'IBM Plex Mono',monospace;
    font-size:0.68rem; color:#4b5563; margin-top:0.2rem;
  }

  /* Streamlit widget overrides */
  .stTextArea textarea {
    background:#13151c !important;
    border:1px solid #2a2f42 !important;
    color:#e0e0e0 !important;
    font-family:'IBM Plex Sans',sans-serif !important;
    border-radius:8px !important;
    font-size:0.92rem !important;
  }
  .stTextArea textarea:focus {
    border-color:#a855f7 !important;
    box-shadow:0 0 0 2px rgba(168,85,247,0.18) !important;
  }
  .stButton > button {
    width:100%;
    background:linear-gradient(135deg,#a855f7,#e83e3e) !important;
    color:white !important; border:none !important;
    border-radius:8px !important;
    font-family:'Syne',sans-serif !important;
    font-weight:700 !important; font-size:0.92rem !important;
    letter-spacing:0.07em !important;
    padding:0.6rem 1.2rem !important;
  }
  .stButton > button:disabled { opacity:0.35 !important; }
  .stButton > button:hover:not(:disabled) { opacity:0.85 !important; }

  [data-testid="stSidebar"] {
    background:#0d0f16 !important;
    border-right:1px solid #1f2333 !important;
  }
  .stExpander {
    background:#13151c !important;
    border:1px solid #1f2333 !important;
    border-radius:8px !important;
  }
  hr { border:none; border-top:1px solid #1f2333; margin:1.4rem 0; }

  /* Spinner status badge */
  .status-badge {
    display:inline-flex; align-items:center; gap:0.5rem;
    font-family:'IBM Plex Mono',monospace; font-size:0.73rem;
    background:#1a1228; color:#c084fc;
    border:1px solid #6d28d9; border-radius:99px;
    padding:0.28rem 0.85rem; margin-bottom:0.8rem;
  }
  .pulse { width:7px; height:7px; border-radius:50%; background:#c084fc;
           animation:blink 1.1s infinite; }
  @keyframes blink { 0%,100%{opacity:1} 50%{opacity:.3} }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def strip_json_block(text: str) -> str:
    """Remove trailing ```json … ``` fence the LLMs append."""
    return re.sub(r"```json[\s\S]*?```", "", text).strip()


def conf_color(score: float) -> str:
    if score >= 0.85: return "#4ade80"
    if score >= 0.65: return "#f5a623"
    return "#e83e3e"


def render_card(label_html: str, label_class: str, body: str):
    clean = strip_json_block(body).replace("<", "&lt;").replace(">", "&gt;")
    st.markdown(f"""
    <div class="card">
      <div class="card-label {label_class}">{label_html}</div>
      <div class="prose">{clean}</div>
    </div>""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="font-family:'Syne',sans-serif;font-weight:800;font-size:1.25rem;
                color:#f5a623;margin-bottom:0.15rem;">⚖️ Debate Engine</div>
    <div style="font-family:'IBM Plex Mono',monospace;font-size:0.66rem;
                color:#4b5563;letter-spacing:0.1em;text-transform:uppercase;
                margin-bottom:1.4rem;">Configuration</div>
    """, unsafe_allow_html=True)

    max_iter = st.slider(
        "Max Critique Iterations", min_value=1, max_value=5, value=2,
        help="How many critique–refine loops before forcing synthesis."
    )

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-family:'IBM Plex Mono',monospace;font-size:0.7rem;
                color:#4b5563;line-height:1.8;">
      <b style="color:#6b7280;">Models</b><br>
      🔵 Gemini 1.5 Pro<br>
      &nbsp;&nbsp;&nbsp;Technical + Critic<br>
      🟢 Llama 3.3 70B (Groq)<br>
      &nbsp;&nbsp;&nbsp;Practical + Synthesizer
    </div>""", unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    with st.expander("💡 Example queries"):
        examples = [
            "Should a 50-engineer startup adopt Kubernetes or stick with Heroku/Railway?",
            "Should we rewrite our Python monolith in Rust for better performance?",
            "Should we adopt GitHub Copilot across the engineering team?",
            "Microservices vs monolith for a B2B SaaS reaching 100k users?",
            "Should we self-host our LLM or use a managed API like OpenAI?",
        ]
        for ex in examples:
            if st.button(ex, key=f"ex_{ex[:24]}"):
                st.session_state["prefill"] = ex
                st.rerun()


# ── Main area ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-title">Multi-Opinion<br>Debate Engine</div>
<div class="hero-sub">Technical × Practical &nbsp;·&nbsp; Critique Loop &nbsp;·&nbsp; Synthesis</div>
""", unsafe_allow_html=True)

prefill = st.session_state.pop("prefill", "")
query = st.text_area(
    "question",
    value=prefill,
    height=108,
    placeholder="Ask a question that has both technical and practical dimensions…",
    label_visibility="collapsed",
)

run_btn = st.button("⚡  Run Debate", disabled=not query.strip())

# ── Execute ───────────────────────────────────────────────────────────────────
if run_btn and query.strip():
    try:
        from debate_engine import run_debate, print_results
    except ImportError as err:
        st.error(
            f"**Could not import `debate_engine.py`.**\n\n"
            f"Make sure it lives in the same folder as `app.py`.\n\n`{err}`"
        )
        st.stop()

    status = st.empty()
    stages = [
        ("🔧", "Gemini generating technical perspective…"),
        ("💼", "Groq generating practical perspective…"),
        ("🔍", "Critic analysing both perspectives…"),
        ("✨", "Synthesizer building final answer…"),
    ]

    # Show a simple animated badge while the engine runs
    status.markdown("""
    <div class="status-badge"><div class="pulse"></div>Running debate engine…</div>
    """, unsafe_allow_html=True)

    with st.spinner(""):
        raw = run_debate(query=query.strip(), max_iterations=max_iter)
        result = print_results(raw)   # print_results now returns the structured dict

    status.empty()
    st.session_state["result"] = result


# ── Render results ────────────────────────────────────────────────────────────
if "result" in st.session_state:
    r = st.session_state["result"]

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Stats row ─────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)

    score = r["confidence_score"]
    pct   = int(score * 100)
    color = conf_color(score)

    with c1:
        st.markdown(f"""
        <div class="stat-card">
          <div class="card-label label-meta">Confidence Score</div>
          <div class="conf-wrap">
            <div class="conf-bar-bg">
              <div class="conf-bar-fill"
                   style="width:{pct}%;background:{color};"></div>
            </div>
            <div class="conf-pct" style="color:{color};">{pct}%</div>
          </div>
        </div>""", unsafe_allow_html=True)

    with c2:
        n = r["iteration"]
        st.markdown(f"""
        <div class="stat-card">
          <div class="card-label label-meta">Iterations</div>
          <div class="stat-val">{n}</div>
          <div class="stat-sub">critique–refine loop{"s" if n != 1 else ""}</div>
        </div>""", unsafe_allow_html=True)

    with c3:
        s = len(r["reasoning_trace"])
        st.markdown(f"""
        <div class="stat-card">
          <div class="card-label label-meta">Trace Steps</div>
          <div class="stat-val">{s}</div>
          <div class="stat-sub">reasoning steps logged</div>
        </div>""", unsafe_allow_html=True)

    # ── Iteration history pills ───────────────────────────────────────────
    if r.get("iteration_history"):
        pills = ""
        for snap in r["iteration_history"]:
            cls  = "pill-warn" if snap["has_major_gaps"] else "pill-ok"
            icon = "⚠️" if snap["has_major_gaps"] else "✅"
            pills += f'<span class="pill {cls}">{icon} Iter {snap["iteration"]}: {snap["gap_summary"]}</span>'
        st.markdown(f'<div style="margin-bottom:1rem;">{pills}</div>', unsafe_allow_html=True)

    # ── Perspectives side-by-side ─────────────────────────────────────────
    left, right = st.columns(2)
    with left:
        render_card(
            "🔧 Technical Perspective &nbsp;<span style='font-size:.6rem;color:#334155;'>(Gemini)</span>",
            "label-tech", r["technical_perspective"]
        )
    with right:
        render_card(
            "💼 Practical Perspective &nbsp;<span style='font-size:.6rem;color:#334155;'>(Groq)</span>",
            "label-prac", r["practical_perspective"]
        )

    # ── Critic ────────────────────────────────────────────────────────────
    render_card("🔍 Critic's Analysis", "label-critic", r["critique"])

    # ── Final synthesis ───────────────────────────────────────────────────
    render_card("✨ Synthesized Final Answer", "label-synth", r["final_answer"])

    # ── Reasoning trace ───────────────────────────────────────────────────
    with st.expander("🧵 Reasoning Trace", expanded=False):
        rows = "".join(
            f'<div class="trace-row">→ {t}</div>'
            for t in r["reasoning_trace"]
        )
        st.markdown(f'<div style="padding:0.4rem 0;">{rows}</div>', unsafe_allow_html=True)

    # ── Download ──────────────────────────────────────────────────────────
    st.download_button(
        label="⬇  Download full result (JSON)",
        data=json.dumps(r, indent=2, default=str),
        file_name="debate_result.json",
        mime="application/json",
    )
