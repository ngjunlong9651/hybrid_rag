"""Toy hybrid retriever: BM25 + dense embeddings fused with Reciprocal Rank Fusion (RRF).

This is the compact, script version of what hybrid_rag_learning.ipynb builds step by step.
Read the notebook first for the theory; read this file to see the whole idea in about 80 lines.

Run: python hybrid_retriever.py "your query"
"""
import re
import sys

import numpy as np
from rank_bm25 import BM25Okapi                      # sparse / keyword retrieval
from sentence_transformers import SentenceTransformer  # dense / semantic retrieval

# A tiny knowledge base. In a real system each string would be one chunk of a longer document.
DOCS = [
    "Error E4021: connection pool exhausted, increase max_connections.",
    "How to reset your password if you forgot your login credentials.",
    "SKU AX-9920 is a stainless steel water bottle, 750ml.",
    "Our refund policy allows returns within 30 days of purchase.",
    "Steps to recover access to your account after losing your password.",
    "The database refused new clients because too many sessions were open.",
    "Shipping typically takes 3 to 5 business days within the country.",
    "Kubernetes pods can be restarted with kubectl rollout restart.",
]

# RRF constant. 60 is the value from the original paper and the usual default.
# Smaller k makes the top ranks dominate; larger k flattens the difference between ranks.
RRF_K = 60


def tokenize(text: str) -> list[str]:
    # BM25 only sees tokens, so tokenization decides what it can match.
    # Lowercase, and keep identifiers like "ax-9920" or "e4021" in one piece.
    return re.findall(r"[a-z0-9_\-]+", text.lower())


class HybridRetriever:
    def __init__(self, docs: list[str], model_name: str = "all-MiniLM-L6-v2"):
        self.docs = docs

        # INDEXING, sparse side: build term statistics (tf, idf, doc lengths) over tokenized docs.
        self.bm25 = BM25Okapi([tokenize(d) for d in docs])

        # INDEXING, dense side: embed every doc once, up front.
        # normalize_embeddings=True gives unit-length vectors, so cosine similarity == dot product.
        self.model = SentenceTransformer(model_name)
        self.doc_vecs = self.model.encode(docs, normalize_embeddings=True)

    def bm25_ranking(self, query: str) -> list[int]:
        scores = self.bm25.get_scores(tokenize(query))
        # argsort is ascending, so negate to get best-first. Returns document positions.
        return list(np.argsort(-scores))

    def dense_ranking(self, query: str) -> list[int]:
        # Encode the query with the SAME model used for documents (mixing models breaks the geometry).
        q = self.model.encode([query], normalize_embeddings=True)[0]
        # One matrix-vector product scores all documents at once.
        return list(np.argsort(-(self.doc_vecs @ q)))

    @staticmethod
    def rrf(rankings: list[list[int]], k: int = RRF_K) -> list[tuple[int, float]]:
        """score(d) = sum over rankers of 1 / (k + rank), where rank starts at 1.

        Uses ranks only, never raw scores, so BM25 and cosine need no normalization.
        A doc that appears near the top of BOTH lists beats one that is first in just one.
        """
        scores: dict[int, float] = {}
        for ranking in rankings:
            for rank, doc_id in enumerate(ranking, start=1):
                scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
        return sorted(scores.items(), key=lambda kv: -kv[1])

    def search(self, query: str, top_k: int = 3, pool: int = 50) -> dict:
        # pool = how many candidates each retriever contributes before fusion (50 to 500 in production).
        sparse = self.bm25_ranking(query)[:pool]
        dense = self.dense_ranking(query)[:pool]
        fused = self.rrf([sparse, dense])[:top_k]
        # Return all three views so you can compare single retrievers against the hybrid result.
        return {"bm25": sparse[:top_k], "dense": dense[:top_k], "hybrid": [d for d, _ in fused]}


def show(label: str, ids: list[int], docs: list[str]) -> None:
    print(f"\n{label}")
    for i in ids:
        print(f"  [{i}] {docs[i]}")


if __name__ == "__main__":
    # Default query is an exact identifier, the case where BM25 should do well.
    query = " ".join(sys.argv[1:]) or "E4021"
    retriever = HybridRetriever(DOCS)
    result = retriever.search(query)
    print(f"Query: {query}")
    show("BM25 only", result["bm25"], DOCS)
    show("Dense only", result["dense"], DOCS)
    show("Hybrid (RRF)", result["hybrid"], DOCS)
