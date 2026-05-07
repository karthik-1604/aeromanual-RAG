from rank_bm25 import BM25Okapi
from langchain.schema import Document
from typing import List
import numpy as np
import logging

logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    Combines BM25 (sparse/keyword) + FAISS (dense/semantic) retrieval.
    Critical for technical documents — exact terms like 'magneto timing'
    or 'torque wrench' are missed by pure semantic search.
    alpha=0.5 means equal weight. Increase for more semantic, decrease for more keyword.
    """

    def __init__(self, vectorstore, chunks: List[Document], k: int = 5, alpha: float = 0.5):
        self.vectorstore = vectorstore
        self.chunks = chunks
        self.k = k
        self.alpha = alpha

        logger.info("Building BM25 index...")
        tokenized = [doc.page_content.lower().split() for doc in chunks]
        self.bm25 = BM25Okapi(tokenized)
        logger.info("HybridRetriever ready")

    def retrieve(self, query: str) -> List[Document]:
        # --- Dense retrieval via FAISS ---
        dense_results = self.vectorstore.similarity_search_with_score(query, k=self.k * 2)
        # Convert L2 distance to a similarity score
        dense_map = {}
        for doc, score in dense_results:
            cid = doc.metadata.get("chunk_id", id(doc))
            dense_map[cid] = (doc, 1 / (1 + score))

        # --- Sparse retrieval via BM25 ---
        tokens = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokens)
        top_bm25_idx = np.argsort(bm25_scores)[::-1][: self.k * 2]
        bm25_map = {}
        for idx in top_bm25_idx:
            if bm25_scores[idx] > 0:
                cid = self.chunks[idx].metadata.get("chunk_id", idx)
                bm25_map[cid] = (self.chunks[idx], float(bm25_scores[idx]))

        # --- Normalize scores to [0, 1] ---
        def normalize(score_map):
            if not score_map:
                return score_map
            vals = [v for _, v in score_map.values()]
            mn, mx = min(vals), max(vals)
            if mx == mn:
                return {k: (doc, 1.0) for k, (doc, _) in score_map.items()}
            return {k: (doc, (v - mn) / (mx - mn)) for k, (doc, v) in score_map.items()}

        dense_map = normalize(dense_map)
        bm25_map = normalize(bm25_map)

        # --- Combine scores ---
        all_ids = set(dense_map) | set(bm25_map)
        combined = {}
        for cid in all_ids:
            d_score = dense_map.get(cid, (None, 0.0))[1]
            b_score = bm25_map.get(cid, (None, 0.0))[1]
            doc = (dense_map.get(cid) or bm25_map.get(cid))[0]
            combined[cid] = (doc, self.alpha * d_score + (1 - self.alpha) * b_score)

        sorted_results = sorted(combined.values(), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in sorted_results[: self.k]]

    def as_langchain_retriever(self):
        """Wrap as a LangChain-compatible retriever."""
        from langchain.schema import BaseRetriever
        from langchain.callbacks.manager import CallbackManagerForRetrieverRun

        hybrid = self

        class _Retriever(BaseRetriever):
            def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun):
                return hybrid.retrieve(query)

        return _Retriever()