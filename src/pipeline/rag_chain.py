import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from typing import Dict
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
# PROMPT — Structured for aerospace Q&A
# Forces citation, handles missing info cleanly
# ─────────────────────────────────────────────
PROMPT_TEMPLATE = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are an expert aerospace maintenance assistant with deep knowledge of FAA regulations and aircraft systems.

INSTRUCTIONS:
- Answer ONLY using the provided context below
- Be specific and technical — this is for trained aviation maintenance technicians
- Always cite the source document name and page number
- If the answer spans multiple sources, cite all of them
- If the answer is NOT in the context, say exactly: "This information is not covered in the provided FAA manuals."
- Never guess or add information not present in the context

Context:
{context}

Question: {question}

Answer (with citations):"""
)


def get_llm():
    """Returns LLM — Groq (free) preferred, falls back to Anthropic."""
    groq_key = os.getenv("GROQ_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")

    if groq_key:
        from langchain_groq import ChatGroq
        print("Using Groq — llama-3.1-70b (free)")
        return ChatGroq(
            model="llama-3.3-70b-versatile",
            api_key=groq_key,
            temperature=0,
            max_tokens=1024
        )
    elif anthropic_key:
        from langchain_anthropic import ChatAnthropic
        print("Using Anthropic Claude")
        return ChatAnthropic(
            model="claude-sonnet-4-20250514",
            api_key=anthropic_key,
            max_tokens=1024
        )
    else:
        raise ValueError(
            "No API key found.\n"
            "Get a free Groq key at https://console.groq.com\n"
            "Then add GROQ_API_KEY=your_key to your .env file"
        )


def build_rag_chain(retriever):
    """Build RAG chain with given retriever (basic or advanced)."""
    llm = get_llm()
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": PROMPT_TEMPLATE},
        return_source_documents=True
    )
    return chain


def build_advanced_rag_chain(vectorstore, chunks):
    """
    Full advanced pipeline:
    Query Rewriting → Hybrid Search → RRF Fusion → Cross-Encoder Reranking
    """
    from src.retrieval.advanced_retriever import AdvancedRetriever
    llm = get_llm()
    retriever = AdvancedRetriever(
        vectorstore=vectorstore,
        chunks=chunks,
        llm=llm,
        initial_k=20,
        final_k=5,
        use_query_rewriting=True,
        use_reranking=True
    )
    lc_retriever = retriever.as_langchain_retriever()
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=lc_retriever,
        chain_type_kwargs={"prompt": PROMPT_TEMPLATE},
        return_source_documents=True
    )
    return chain, chunks  # return chunks so streamlit can use them


def format_response(result: Dict) -> Dict:
    """Format output with clean deduplicated citations."""
    sources = []
    for doc in result.get("source_documents", []):
        m = doc.metadata
        src = f"{m.get('source', 'unknown')} — Page {m.get('page', '?')}"
        if src not in sources:
            sources.append(src)
    return {
        "answer": result["result"],
        "sources": sources,
        "num_sources": len(sources)
    }


if __name__ == "__main__":
    from src.embeddings.vector_store import load_index
    from src.ingestion.pdf_loader import load_all_documents
    from src.ingestion.chunker import pages_to_langchain_docs, chunk_documents

    print("Loading index and chunks...")
    vs = load_index()
    pages = load_all_documents()
    docs = pages_to_langchain_docs(pages)
    chunks = chunk_documents(docs)

    print("Building advanced RAG chain...")
    chain, _ = build_advanced_rag_chain(vs, chunks)

    query = "What are the inspection requirements for turbine engine hot section?"
    print(f"\nQuery: {query}\n")
    result = chain.invoke({"query": query})
    resp = format_response(result)
    print("Answer:", resp["answer"])
    print("\nSources:", resp["sources"])