"""Professional Streamlit UI for the Day 8 Vietnamese legal RAG pipeline."""

import json
import html
import re
import time
from pathlib import Path

import streamlit as st

from src.task9_retrieval_pipeline import retrieve
from src.task10_generation import generate_with_citation
from src.security_guardrails import REFUSAL_MESSAGE, should_refuse_query

PROJECT_DIR = Path(__file__).parent
MANIFEST_PATH = PROJECT_DIR / "data" / "index" / "manifest.json"

st.set_page_config(
    page_title="DrugLaw RAG Workspace",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)


def _load_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        return {}
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def _init_state() -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("last_sources", [])
    st.session_state.setdefault("last_latency", None)
    st.session_state.setdefault("pending_prompt", None)


def _source_name(source: dict) -> str:
    metadata = source.get("metadata", {}) or {}
    return metadata.get("source") or metadata.get("path") or "Unknown source"


def _source_caption(source: dict) -> str:
    metadata = source.get("metadata", {}) or {}
    source_type = metadata.get("type", "unknown")
    chunk_index = metadata.get("chunk_index", "n/a")
    score = float(source.get("score", 0.0) or 0.0)
    retrieval_source = source.get("source", "unknown")
    return (
        f"{source_type.upper()}  |  chunk {chunk_index}  |  "
        f"{retrieval_source}  |  score {score:.3f}"
    )


def _clean_answer_text(text: str) -> str:
    """Remove inline evidence markers from the visible assistant answer."""
    cleaned = re.sub(r"\s*\[[^\]]+?,\s*chunk\s*\d+\]", "", text)
    cleaned = re.sub(r"\s+\.", ".", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def _render_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
        
        :root {
            --ink: #1e293b;
            --muted: #64748b;
            --line: #e2e8f0;
            --panel: #f8fafc;
            --legal: #e11d48;
            --teal: #0d9488;
            --amber: #d97706;
            --glass-bg: rgba(255, 255, 255, 0.7);
        }
        
        * {
            font-family: 'Outfit', sans-serif !important;
        }
        
        .stApp { background: linear-gradient(135deg, #fdfbfb 0%, #ebedee 100%); color: var(--ink); }
        
        [data-testid="stHeader"] { 
            background: var(--glass-bg);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border-bottom: 1px solid rgba(255,255,255,0.3);
        }
        
        [data-testid="stSidebar"] {
            background: rgba(248, 250, 252, 0.85);
            backdrop-filter: blur(10px);
            border-right: 1px solid rgba(226, 232, 240, 0.8);
            box-shadow: 4px 0 15px rgba(0,0,0,0.03);
        }
        
        /* Typography */
        h1, h2, h3 { letter-spacing: -0.02em; color: var(--ink); font-weight: 700; }
        h1 { font-size: 2.2rem !important; margin-bottom: .2rem !important; 
             background: linear-gradient(90deg, #1e293b, #3b82f6); 
             -webkit-background-clip: text; 
             -webkit-text-fill-color: transparent; }
             
        .app-kicker {
            background: linear-gradient(90deg, #e11d48, #f43f5e);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: .85rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            margin-bottom: .5rem;
            display: inline-block;
        }
        .app-subtitle {
            color: var(--muted);
            font-size: .95rem;
            margin: 0 0 1rem 0;
            font-weight: 400;
        }
        
        /* Status Strip */
        .status-strip {
            display: flex;
            flex-wrap: wrap;
            gap: .55rem 1.2rem;
            padding: 1rem 1.2rem;
            border: 1px solid rgba(255,255,255,0.8);
            border-left: 4px solid var(--teal);
            background: rgba(255,255,255,0.65);
            backdrop-filter: blur(8px);
            border-radius: 12px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.02);
            font-size: .85rem;
            margin-bottom: 1.5rem;
            transition: all 0.3s ease;
        }
        .status-strip:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 20px rgba(0,0,0,0.05);
        }
        .status-strip strong { color: var(--ink); font-weight: 600; }
        
        /* Chat Messages */
        [data-testid="stChatMessage"] {
            border: 1px solid rgba(226, 232, 240, 0.6);
            border-radius: 16px;
            padding: 1rem 1.2rem;
            background: rgba(255,255,255,0.8);
            box-shadow: 0 2px 10px rgba(0,0,0,0.02);
            margin-bottom: 1.2rem;
            animation: fadeIn 0.4s ease-out;
            transition: all 0.2s ease;
        }
        [data-testid="stChatMessage"]:hover {
            box-shadow: 0 5px 20px rgba(0,0,0,0.06);
            transform: translateY(-1px);
        }
        [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
            border-left: 4px solid var(--legal);
            background: linear-gradient(to right, rgba(225,29,72,0.02), rgba(255,255,255,0.9));
        }
        [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
            border-left: 4px solid var(--teal);
            background: linear-gradient(to right, rgba(13,148,136,0.02), rgba(255,255,255,0.9));
        }
        
        /* Expanders */
        [data-testid="stExpander"] {
            border: 1px solid var(--line);
            border-radius: 12px;
            background: rgba(255,255,255,0.9);
            box-shadow: 0 2px 8px rgba(0,0,0,0.02);
            transition: all 0.3s ease;
        }
        [data-testid="stExpander"]:hover {
            box-shadow: 0 6px 15px rgba(0,0,0,0.05);
        }
        
        /* Buttons */
        .stButton > button {
            border-radius: 8px;
            border: 1px solid var(--line);
            background: white;
            font-weight: 600;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            box-shadow: 0 2px 4px rgba(0,0,0,0.02);
        }
        .stButton > button:hover {
            border-color: var(--teal);
            color: var(--teal);
            box-shadow: 0 4px 12px rgba(13, 148, 136, 0.15);
            transform: translateY(-2px);
        }
        
        /* Input */
        [data-testid="stChatInput"] {
            background: transparent;
        }
        [data-testid="stChatInput"] textarea { 
            border-radius: 16px; 
            border: 1px solid var(--line);
            box-shadow: 0 4px 20px rgba(0,0,0,0.04);
            transition: all 0.3s ease;
            font-size: 1rem;
        }
        [data-testid="stChatInput"] textarea:focus {
            border-color: var(--teal);
            box-shadow: 0 4px 20px rgba(13, 148, 136, 0.15);
        }
        
        .empty-state {
            border-radius: 16px;
            background: rgba(255,255,255,0.6);
            backdrop-filter: blur(8px);
            border: 1px solid rgba(255,255,255,0.8);
            padding: 2rem;
            margin: .5rem 0 1.5rem 0;
            text-align: center;
            box-shadow: 0 4px 15px rgba(0,0,0,0.03);
        }
        .empty-state strong { color: var(--ink); font-size: 1.1rem; }
        .empty-state p { color: var(--muted); margin: .5rem 0 0 0; }
        
        .source-meta {
            color: var(--teal);
            font-size: .75rem;
            font-weight: 600;
            text-transform: uppercase;
            margin-bottom: .4rem;
            letter-spacing: 0.05em;
        }
        .source-snippet {
            color: #334155;
            font-size: .85rem;
            line-height: 1.6;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_sidebar(manifest: dict) -> tuple[str, int]:
    with st.sidebar:
        st.markdown("### DrugLaw RAG")
        st.caption("Vietnamese legal and news intelligence")

        mode = st.radio(
            "Workspace mode",
            ["Ask with citations", "Inspect retrieval"],
            captions=["Generate a grounded answer", "Review ranked evidence only"],
        )
        top_k = st.slider("Evidence chunks", min_value=3, max_value=8, value=5)

        st.divider()
        st.markdown("#### Pipeline")
        left, right = st.columns(2)
        left.metric("Documents", manifest.get("document_count", 0))
        right.metric("Chunks", manifest.get("chunk_count", 0))
        st.caption(f"Embedding: `{manifest.get('embedding_model', 'not indexed')}`")
        st.caption(f"Vector shape: `{manifest.get('embedding_dim', 0)} dimensions`")

        st.divider()
        st.markdown("#### Session")
        message_count = len(st.session_state.messages)
        st.caption(f"{message_count} messages in this conversation")
        if st.button(
            "Clear conversation",
            use_container_width=True,
            icon=":material/delete_sweep:",
        ):
            st.session_state.messages = []
            st.session_state.last_sources = []
            st.session_state.last_latency = None
            st.rerun()

        st.divider()
        st.caption(
            "Answers are generated from the indexed documents and may require "
            "professional legal review."
        )
    return mode, top_k


def _render_sources(sources: list[dict]) -> None:
    if not sources:
        st.info("No evidence has been retrieved in this session.")
        return

    for index, source in enumerate(sources, 1):
        name = _source_name(source)
        with st.expander(f"{index}. {name}", expanded=index == 1):
            st.markdown(
                f'<div class="source-meta">{_source_caption(source)}</div>',
                unsafe_allow_html=True,
            )
            snippet = source.get("content", "").strip()
            st.markdown(
                f'<div class="source-snippet">{html.escape(snippet)}</div>',
                unsafe_allow_html=True,
            )


def _render_answer_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander("Citations used in this answer", expanded=False):
        for index, source in enumerate(sources, 1):
            metadata = source.get("metadata", {}) or {}
            label = metadata.get("source") or metadata.get("path") or f"Source {index}"
            chunk_index = metadata.get("chunk_index", "n/a")
            score = float(source.get("score", 0.0) or 0.0)
            st.markdown(
                f"- **{index}. {label}** `chunk {chunk_index}`  "
                f"`score {score:.3f}`"
            )


def _render_empty_state() -> None:
    st.markdown(
        """
        <div class="empty-state">
          <strong>Start with a legal question or inspect a reported case.</strong>
          <p>The workspace retrieves from Vietnamese legal documents and curated news, then cites the evidence used.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    examples = [
        "Luật Phòng, chống ma túy quy định gì về cai nghiện?",
        "Những người nổi tiếng nào bị bắt vì liên quan ma túy?",
        "Hình phạt cho hành vi tàng trữ trái phép chất ma túy là gì?",
    ]
    columns = st.columns(3)
    for column, example in zip(columns, examples):
        if column.button(example, use_container_width=True):
            st.session_state.pending_prompt = example
            st.rerun()


def _run_query(prompt: str, mode: str, top_k: int) -> dict:
    started = time.perf_counter()
    guardrail = should_refuse_query(prompt)
    if guardrail.blocked:
        return {
            "answer": REFUSAL_MESSAGE,
            "sources": [],
            "retrieval_source": "blocked",
            "latency": time.perf_counter() - started,
        }
    if mode == "Inspect retrieval":
        sources = retrieve(prompt, top_k=top_k)
        answer = (
            f"Retrieved {len(sources)} evidence chunks. Review their ranking "
            "and source metadata in the evidence panel."
        )
        retrieval_source = sources[0].get("source", "none") if sources else "none"
    else:
        result = generate_with_citation(prompt, top_k=top_k)
        answer = result["answer"]
        sources = result["sources"]
        retrieval_source = result["retrieval_source"]

    return {
        "answer": answer,
        "sources": sources,
        "retrieval_source": retrieval_source,
        "latency": time.perf_counter() - started,
    }


_init_state()
_render_css()
manifest = _load_manifest()
mode, top_k = _render_sidebar(manifest)

st.markdown('<div class="app-kicker">Evidence-grounded workspace</div>', unsafe_allow_html=True)
st.title("Vietnam Drug Law Research")
st.markdown(
    '<p class="app-subtitle">Hybrid retrieval, vectorless fallback, and cited Gemini responses.</p>',
    unsafe_allow_html=True,
)

latency_text = (
    f"{st.session_state.last_latency:.2f}s"
    if st.session_state.last_latency is not None
    else "not run"
)
st.markdown(
    f"""
    <div class="status-strip">
      <span><strong>Mode</strong> {mode}</span>
      <span><strong>Index</strong> {manifest.get('chunk_count', 0)} chunks ready</span>
      <span><strong>Last response</strong> {latency_text}</span>
      <span><strong>Citation policy</strong> required</span>
    </div>
    """,
    unsafe_allow_html=True,
)

chat_col, evidence_col = st.columns([1.7, 1], gap="large")

with chat_col:
    st.markdown("### Conversation")
    if not st.session_state.messages:
        _render_empty_state()
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                st.caption(
                    f"{message.get('retrieval_source', 'unknown')} retrieval  |  "
                    f"{message.get('latency', 0):.2f}s"
                )
                _render_answer_sources(message.get("sources", []))

with evidence_col:
    st.markdown("### Evidence")
    _render_sources(st.session_state.last_sources)

prompt = st.chat_input("Ask about Vietnamese drug law or related cases")
if st.session_state.pending_prompt:
    prompt = st.session_state.pending_prompt
    st.session_state.pending_prompt = None

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.spinner("Retrieving evidence and preparing response..."):
        try:
            result = _run_query(prompt, mode, top_k)
        except Exception as exc:
            result = {
                "answer": f"The request could not be completed: {exc}",
                "sources": [],
                "retrieval_source": "error",
                "latency": 0.0,
            }

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": _clean_answer_text(result["answer"]),
            "raw_answer": result["answer"],
            "sources": result["sources"],
            "retrieval_source": result["retrieval_source"],
            "latency": result["latency"],
        }
    )
    st.session_state.last_sources = result["sources"]
    st.session_state.last_latency = result["latency"]
    st.rerun()
