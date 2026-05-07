import streamlit as st
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.embeddings.vector_store import load_index, index_exists
from src.ingestion.pdf_loader import load_all_documents
from src.ingestion.chunker import pages_to_langchain_docs, chunk_documents
from src.pipeline.rag_chain import build_advanced_rag_chain, format_response

st.set_page_config(page_title="AeroManual RAG", page_icon="✈️", layout="wide")

st.title("✈️ AeroManual — Aerospace Document Assistant")
st.caption("Advanced RAG: Query Rewriting → Hybrid Search → RRF Fusion → Cross-Encoder Reranking")

# ── Sidebar ──
with st.sidebar:
    st.header("📚 Loaded Documents")
    st.markdown("""
    - FAA AMT Handbook — Powerplant
    - FAA AMT Handbook — Airframe
    - FAA AMT Handbook — General
    """)
    st.divider()
    st.header("⚙️ RAG Pipeline")
    st.markdown("""
    1. **Query Rewriting** — 3 query variants
    2. **Hybrid Search** — BM25 + FAISS
    3. **RRF Fusion** — Reciprocal Rank Fusion
    4. **Reranking** — BGE Cross-Encoder
    """)
    st.divider()
    st.header("💡 Example Questions")
    examples = [
        "What are inspection intervals for turbine hot sections?",
        "Describe the magneto timing procedure",
        "What are hydraulic system maintenance checks?",
        "Explain fuel system safety precautions",
        "How is a compression test performed?",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state["prefill"] = ex

# ── Load Pipeline ──
@st.cache_resource(show_spinner="Loading RAG pipeline (first time takes ~2 mins)...")
def load_pipeline():
    if not index_exists():
        return None, None
    vs = load_index()
    pages = load_all_documents()
    docs = pages_to_langchain_docs(pages)
    chunks = chunk_documents(docs)
    chain, chunks = build_advanced_rag_chain(vs, chunks)
    return chain, chunks

chain, chunks = load_pipeline()

if chain is None:
    st.error("""
    ⚠️ FAISS index not found. Run this first:
    ```
    python -m src.embeddings.vector_store
    ```
    Then refresh this page.
    """)
    st.stop()

# ── Chat History ──
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg["role"] == "assistant" and "sources" in msg:
            with st.expander("📌 Sources"):
                for src in msg["sources"]:
                    st.markdown(f"- `{src}`")

# ── Input ──
prefill = st.session_state.pop("prefill", "")
query = st.chat_input("Ask a maintenance question...") or prefill

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        with st.spinner("Running advanced retrieval pipeline..."):
            result = chain.invoke({"query": query})
            resp = format_response(result)

        st.write(resp["answer"])

        col1, col2 = st.columns([1, 1])
        with col1:
            with st.expander(f"📌 Sources ({resp['num_sources']})"):
                for src in resp["sources"]:
                    st.markdown(f"- `{src}`")
        with col2:
            with st.expander("🔎 Retrieved Chunks"):
                for doc in result["source_documents"]:
                    st.markdown(f"**{doc.metadata.get('source')} — Page {doc.metadata.get('page')}**")
                    st.text(doc.page_content[:400] + "...")
                    st.divider()

    st.session_state.messages.append({
        "role": "assistant",
        "content": resp["answer"],
        "sources": resp["sources"]
    })