"""Cache-backed drop-in retriever with unchanged BM25 + semantic + RRF ranks."""
from __future__ import annotations

from pathlib import Path

from retrieval_embedding_cache import load_or_build_embeddings


class CachedHybridRetriever:
    """Wrap the existing HybridRetriever methods, changing only index setup.

    Production retrieval uses HybridRetriever.retrieve_all() and retrieve()
    unchanged; all candidate counts and RRF settings are preserved.
    """

    def __init__(self, documents: list[dict], model, *, cache_dir: str | Path,
                 encoder_key: str, encoder_revision: str = "default",
                 candidate_k: int = 100, rrf_constant: int = 60,
                 rebuild_cache: bool = False):
        # Lazy import to keep cache unit tests independent of transformer libs.
        from hybrid_retriever import HybridRetriever
        from bm25_retriever import tokenize
        from rank_bm25 import BM25Okapi
        if not documents:
            raise ValueError("Cannot build a hybrid retriever from an empty corpus")
        if type(candidate_k) is not int or candidate_k <= 0:
            raise ValueError("candidate_k must be a positive integer")
        if type(rrf_constant) is not int or rrf_constant <= 0:
            raise ValueError("rrf_constant must be a positive integer")
        self.documents = documents
        self.doc_ids = [int(d["doc_id"]) for d in documents]
        self.model = model
        self.candidate_k = candidate_k
        self.rrf_constant = rrf_constant
        self.bm25 = BM25Okapi([
            tokenize(doc["title"] + " " + " ".join(doc["abstract"]))
            for doc in documents
        ])
        self.cache_result = load_or_build_embeddings(
            documents, model, cache_dir=cache_dir,
            encoder_key=encoder_key, encoder_revision=encoder_revision,
            rebuild=rebuild_cache,
        )
        self.embeddings = self.cache_result.embeddings
        # The ranking algorithms come from the original implementation.
        self._retrieve_all = HybridRetriever.retrieve_all
        self._retrieve = HybridRetriever.retrieve

    def retrieve_all(self, query: str, k: int = 20) -> dict[str, list[int]]:
        return self._retrieve_all(self, query, k=k)

    def retrieve(self, query: str, k: int = 5) -> list[int]:
        return self._retrieve(self, query, k=k)
