"""
Advanced Retrieval Pipeline — 4 techniques used in production RAG systems:

1. Hybrid Search      — BM25 (keyword) + Dense (semantic) with RRF fusion
2. Query Rewriting    — LLM expands query into multiple sub-queries
3. Reranking          — Cross-encoder reranks top-k candidates
4. Contextual Compression — Strips irrelevant parts from retrieved chunks
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

MODEL_CACHE_PATH = "data/models"
os.environ["HF_HOME"] = MODEL_CACHE_PATH  # Keep all models inside project

from rank_bm25 import BM25Okapi
from langchain.schema import Document
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers.document_compressors import CrossEncoderReranker
from typing import List
import numpy as np
import logging

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# TECHNIQUE 1: Reciprocal Rank Fusion (RRF)
# Better than simple score averaging — used by
# Microsoft, Cohere, and most production systems
# ─────────────────────────────────────────────
def reciprocal_rank_fusion(results_lists: List[List[Document]], k: int = 60) -> List[Document]:
    """
    RRF merges multiple ranked lists without needing score normalization.
    Formula: score(d) = sum(1 / (k + rank(d))) across all lists
    k=60 is the standard constant used in the original RRF paper.
    """
    scores = {}
    doc_map = {}

    for results in results_lists:
        for rank, doc in enumerate(results):
            doc_id = doc.metadata.get("chunk_id", hash(doc.page_content))
            if doc_id not in scores:
                scores[doc_id] = 0.0
                doc_map[doc_id] = doc
            scores[doc_id] += 1.0 / (k + rank + 1)

    sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)
    return [doc_map[i] for i in sorted_ids]


# ─────────────────────────────────────────────
# TECHNIQUE 2: Query Rewriting / Decomposition
# LLM generates multiple versions of the query
# to improve recall — used in HyDE, RAG-Fusion
# ─────────────────────────────────────────────
def rewrite_query(query: str, llm) -> List[str]:
    """
    Generate 3 alternative phrasings of the query.
    This improves recall by covering different ways
    the same information might be expressed in docs.
    """
    prompt = f"""You are an expert at reformulating technical aerospace maintenance questions.
Given this question, generate 3 different versions that capture the same intent
but use different terminology. This helps retrieve more relevant documents.

Original question: {query}

Output exactly 3 alternative questions, one per line, no numbering or bullets:"""

    response = llm.invoke(prompt)
    lines = [l.strip() for l in response.content.strip().split("\n") if l.strip()]
    rewrites = lines[:3] if len(lines) >= 3 else lines
    # Always include the original
    all_queries = [query] + rewrites
    logger.info(f"Query rewriting: {len(all_queries)} queries generated")
    return all_queries


# ─────────────────────────────────────────────
# TECHNIQUE 3: Cross-Encoder Reranking
# Two-stage retrieval: fast retriever gets top-20,
# slow reranker picks best 5. Used by Cohere, BGE.
# ─────────────────────────────────────────────
def get_reranker(top_n: int = 5):
    """
    BGE reranker — free, runs locally, very effective.
    Scores query-document pairs jointly (not separately).
    """
    model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
    return CrossEncoderReranker(model=model, top_n=top_n)


# ─────────────────────────────────────────────
# TECHNIQUE 4: Contextual Compression
# Instead of passing full chunks to LLM,
# extract only the sentences relevant to query.
# Reduces noise and saves tokens.
# ─────────────────────────────────────────────
def get_compression_retriever(base_retriever, llm):
    compressor = LLMChainExtractor.from_llm(llm)
    return ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever
    )


# ─────────────────────────────────────────────
# MAIN: Advanced Hybrid Retriever
# Combines all 4 techniques into one pipeline
# ─────────────────────────────────────────────
class AdvancedRetriever:
    def __init__(
        self,
        vectorstore,
        chunks: List[Document],
        llm,
        initial_k: int = 20,   # retrieve more initially
        final_k: int = 5,      # rerank down to this many
        use_query_rewriting: bool = True,
        use_reranking: bool = True
    ):
        self.vectorstore = vectorstore
        self.chunks = chunks
        self.llm = llm
        self.initial_k = initial_k
        self.final_k = final_k
        self.use_query_rewriting = use_query_rewriting
        self.use_reranking = use_reranking

        # Build BM25 index
        logger.info("Building BM25 index...")
        tokenized = [doc.page_content.lower().split() for doc in chunks]
        self.bm25 = BM25Okapi(tokenized)

        # Load reranker if needed
        if use_reranking:
            logger.info("Loading BGE reranker...")
            self.reranker = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")

        logger.info("AdvancedRetriever ready")

    def _dense_search(self, query: str) -> List[Document]:
        return self.vectorstore.similarity_search(query, k=self.initial_k)

    def _bm25_search(self, query: str) -> List[Document]:
        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)
        top_idx = np.argsort(scores)[::-1][:self.initial_k]
        return [self.chunks[i] for i in top_idx if scores[i] > 0]

    def _rerank(self, query: str, docs: List[Document]) -> List[Document]:
        if not docs:
            return docs
        pairs = [(query, doc.page_content) for doc in docs]
        scores = self.reranker.score(pairs)
        scored = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored[:self.final_k]]

    def retrieve(self, query: str) -> List[Document]:
        # Step 1: Query rewriting
        if self.use_query_rewriting:
            queries = rewrite_query(query, self.llm)
        else:
            queries = [query]

        # Step 2: Hybrid search for each query → RRF fusion
        all_ranked = []
        for q in queries:
            dense = self._dense_search(q)
            sparse = self._bm25_search(q)
            fused = reciprocal_rank_fusion([dense, sparse])
            all_ranked.append(fused)

        # Step 3: RRF across all query results
        final_candidates = reciprocal_rank_fusion(all_ranked)[:self.initial_k]

        # Step 4: Rerank with cross-encoder
        if self.use_reranking:
            final_docs = self._rerank(query, final_candidates)
        else:
            final_docs = final_candidates[:self.final_k]

        logger.info(f"Retrieved {len(final_docs)} docs for: '{query[:60]}'")
        return final_docs

    def as_langchain_retriever(self):
        """Wrap as LangChain-compatible retriever for use in chains."""
        from langchain.schema import BaseRetriever
        from langchain.callbacks.manager import CallbackManagerForRetrieverRun

        adv = self

        class _Retriever(BaseRetriever):
            def _get_relevant_documents(
                self, query: str, *, run_manager: CallbackManagerForRetrieverRun
            ) -> List[Document]:
                return adv.retrieve(query)

        return _Retriever()