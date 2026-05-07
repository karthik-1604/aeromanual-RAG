import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from src.embeddings.vector_store import load_index, index_exists
from src.ingestion.pdf_loader import load_all_documents
from src.ingestion.chunker import pages_to_langchain_docs, chunk_documents
from src.retrieval.hybrid_search import HybridRetriever


@pytest.fixture(scope="module")
def setup():
    assert index_exists(), "FAISS index must exist — run vector_store.py first"
    vs = load_index()
    pages = load_all_documents()
    docs = pages_to_langchain_docs(pages)
    chunks = chunk_documents(docs)
    return vs, chunks


def test_index_exists():
    assert index_exists(), "FAISS index should exist"


def test_dense_retrieval_returns_results(setup):
    vs, chunks = setup
    retriever = vs.as_retriever(search_kwargs={"k": 5})
    results = retriever.invoke("turbine engine inspection")
    assert len(results) == 5
    for doc in results:
        assert len(doc.page_content) > 0
        assert "source" in doc.metadata


def test_hybrid_retriever_returns_results(setup):
    vs, chunks = setup
    hybrid = HybridRetriever(vs, chunks, k=5)
    results = hybrid.retrieve("magneto timing reciprocating engine")
    assert len(results) > 0
    assert len(results) <= 5


def test_hybrid_retriever_has_metadata(setup):
    vs, chunks = setup
    hybrid = HybridRetriever(vs, chunks, k=5)
    results = hybrid.retrieve("fuel system safety")
    for doc in results:
        assert "source" in doc.metadata
        assert "page" in doc.metadata


def test_retrieval_relevance(setup):
    vs, chunks = setup
    retriever = vs.as_retriever(search_kwargs={"k": 5})
    results = retriever.invoke("aircraft hydraulic system")
    sources = [doc.metadata["source"] for doc in results]
    aerospace_sources = ["faa_powerplant", "faa_airframe", "faa_general"]
    assert any(s in aerospace_sources for s in sources)