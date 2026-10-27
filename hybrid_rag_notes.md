# Hybrid RAG: Learning Notes

## 1. Core idea
Hybrid RAG retrieves with two complementary methods and merges the results before the LLM sees them.

| Method | Strength | Weakness |
|---|---|---|
| Sparse / lexical (BM25) | Exact terms, IDs, SKUs, error codes, rare names | No synonym or meaning awareness |
| Dense / vector (embeddings + ANN) | Paraphrases, conceptual queries | Misses exact identifiers, can drift |

Combining them typically lifts recall noticeably (one guide cites roughly 65-78% to 91% recall@10, treat vendor numbers as indicative).

## 2. The standard pipeline
1. **Chunk and index** documents twice: a BM25/inverted index and a vector index.
2. **Retrieve in parallel**: top 50-500 candidates from each. Parallel calls keep latency close to a single retrieval.
3. **Fuse** the two ranked lists (RRF or weighted score blend).
4. **Rerank** the top 50-200 with a cross-encoder (or late-interaction model like ColBERT).
5. **Generate**: pass the top 3-10 chunks to the LLM.

## 3. Key concepts to learn, in order
1. **BM25**: TF-IDF successor, with term-frequency saturation and length normalization.
2. **Embeddings and ANN indexes**: HNSW, IVF, cosine similarity.
3. **Learned sparse retrieval**: SPLADE, a middle ground between BM25 and dense.
4. **Fusion**:
   - **RRF**: `score(d) = sum over rankers 1 / (k + rank(d))`, k is usually 60. Uses ranks only, so no score normalization is needed. Robust default.
   - **Weighted/convex combination (alpha)**: needs score normalization, but is tunable when you have eval data.
5. **Reranking**: bi-encoder (fast, retrieval) vs cross-encoder (slow, accurate, applied to few candidates).
6. **Contextual Retrieval**: prepend an LLM-generated blurb to each chunk before embedding and BM25 indexing. Anthropic reports about 49% fewer failed retrievals, and about 67% with reranking.
7. **Evaluation**: recall@k, MRR, nDCG on a small labeled query set. Without this you cannot tune alpha, k, or chunk size.

## 4. Practical gotchas
- Tokenization and stemming for BM25 matter (code, multilingual text, identifiers).
- Always evaluate against dense-only and BM25-only baselines; hybrid is not guaranteed to win on your data.
- Reranker latency grows with candidate count, so cap it.
- Metadata filters should apply to both retrievers consistently.
- Chunk size interacts strongly with both retrieval methods.

## 5. Suggested learning path
1. Read an overview (denser.ai guide) to get the mental model.
2. Read Anthropic's Contextual Retrieval post for a concrete, measured pipeline.
3. Build a toy hybrid retriever in Python (`rank_bm25` + `sentence-transformers` + hand-written RRF).
4. Re-implement it in a real engine (Qdrant, Weaviate, or Elasticsearch).
5. Add a cross-encoder reranker and measure the gain.
6. Build a small eval set and compare configurations.

## 6. Resources
Overviews and tutorials
- [Hybrid Search for RAG: Combining BM25 and Dense Vector Search (denser.ai)](https://denser.ai/blog/hybrid-search-for-rag/)
- [Implementing Hybrid Semantic + Lexical Search in RAG (Machine Learning Mastery)](https://machinelearningmastery.com/implementing-hybrid-semantic-lexical-search-in-rag/)
- [Hybrid Search for RAG: BM25, SPLADE, and Vector Search Combined (Prem AI)](https://blog.premai.io/hybrid-search-for-rag-bm25-splade-and-vector-search-combined)
- [Hybrid search and reranking for RAG: BM25, vectors, cross-encoders (wz-it)](https://wz-it.com/en/knowledge/ki/hybrid-search-reranking/)
- [Hybrid Search guide (Supermemory)](https://blog.supermemory.ai/hybrid-search-guide/)

Contextual Retrieval
- Anthropic's original post: https://www.anthropic.com/news/contextual-retrieval (from memory, not fetched in this session, verify it loads)
- [InfoQ summary](https://www.infoq.com/news/2024/09/anthropic-contextual-retrieval/)
- [Analytics Vidhya walkthrough](https://www.analyticsvidhya.com/blog/2024/11/anthropics-contextual-rag/)

Engine documentation (hands-on)
- [Weaviate: hybrid search concepts](https://docs.weaviate.io/weaviate/concepts/search/hybrid-search)
- [Qdrant: hybrid search with reranking](https://qdrant.tech/documentation/advanced-tutorials/reranking-hybrid-search/)
- [Chroma: hybrid search](https://docs.trychroma.com/cloud/search-api/hybrid-search)
- Elasticsearch RRF retriever: see the "rrf retriever" page in the Elastic reference docs (the search only surfaced preview-build links, so search the official docs site instead).

Background papers (not fetched, search by title)
- Cormack et al., "Reciprocal Rank Fusion outperforms Condorcet and individual rank learning methods" (2009)
- Formal et al., "SPLADE" (2021)
- Khattab and Zaharia, "ColBERT" (2020)
