"""Hybrid SciFact retrieval using BM25, semantic embeddings, and RRF.

Install: pip install rank-bm25 sentence-transformers numpy
Run one SciFact claim: python hybrid_retriever.py

The BM25 index and semantic document embeddings are constructed once per
HybridRetriever instance, then reused for all queries.
"""

import argparse
from pathlib import Path

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from bm25_retriever import CLAIM, EVIDENCE_IDS, load_corpus, tokenize
from retrieval_metrics import recall_at_k
from semantic_retriever import MODEL_NAME, build_embeddings, retrieve_semantic


BASE_DIR = Path(__file__).resolve().parent


def reciprocal_rank_fusion(
    rankings: list[list[int]],
    k: int = 20,
    rrf_constant: int = 60,
) -> list[int]:
    """Fuse ranked document IDs without requiring comparable model scores.

    Each document receives sum(1 / (rrf_constant + rank)), where ranks
    begin at 1. A duplicate within the SAME ranking is counted only once;
    a document in both rankings receives contributions from both.
    Ties are resolved deterministically by first occurrence in the inputs.
    """
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer")
    if (isinstance(rrf_constant, bool) or not isinstance(rrf_constant, int)
            or rrf_constant <= 0):
        raise ValueError("rrf_constant must be a positive integer")

    scores: dict[int, float] = {}
    first_seen: dict[int, int] = {}
    for ranking in rankings:
        seen_in_ranking: set[int] = set()
        for rank, doc_id in enumerate(ranking, start=1):
            if doc_id in seen_in_ranking:
                continue
            seen_in_ranking.add(doc_id)
            if doc_id not in first_seen:
                first_seen[doc_id] = len(first_seen)
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rrf_constant + rank)

    return sorted(scores, key=lambda doc_id: (-scores[doc_id], first_seen[doc_id]))[:k]


class HybridRetriever:
    """Build both retrieval representations once, then answer many queries.

    candidate_k controls the number of candidates supplied BY EACH model
    to RRF, not the number returned by the final hybrid ranking.
    """

    def __init__(
        self,
        documents: list[dict],
        model: SentenceTransformer,
        candidate_k: int = 100,
        rrf_constant: int = 60,
    ) -> None:
        if not documents:
            raise ValueError("Cannot build a hybrid retriever from an empty corpus")
        if isinstance(candidate_k, bool) or not isinstance(candidate_k, int) or candidate_k <= 0:
            raise ValueError("candidate_k must be a positive integer")
        if (isinstance(rrf_constant, bool) or not isinstance(rrf_constant, int)
                or rrf_constant <= 0):
            raise ValueError("rrf_constant must be a positive integer")

        self.documents = documents
        self.doc_ids = [int(doc["doc_id"]) for doc in documents]
        self.model = model
        self.candidate_k = candidate_k
        self.rrf_constant = rrf_constant

        # BM25 and document embeddings are built ONCE, not per query.
        tokens = [
            tokenize(doc["title"] + " " + " ".join(doc["abstract"]))
            for doc in documents
        ]
        self.bm25 = BM25Okapi(tokens)
        self.embeddings = build_embeddings(documents, model)

    def retrieve_all(self, query: str, k: int = 20) -> dict[str, list[int]]:
        """Return BM25, semantic, and RRF top-k IDs for the same query."""
        if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
            raise ValueError("k must be a positive integer")

        # Candidates from both methods are drawn from the same document set.
        depth = min(len(self.documents), max(k, self.candidate_k))

        scores = self.bm25.get_scores(tokenize(query))
        bm25_indices = sorted(
            range(len(self.documents)), key=lambda i: -scores[i]
        )[:depth]
        bm25_ids = [self.doc_ids[i] for i in bm25_indices]

        # This calls our existing semantic_retriever.py implementation.
        semantic_ids = retrieve_semantic(
            query, self.documents, self.embeddings, self.model, k=depth
        )

        hybrid_ids = reciprocal_rank_fusion(
            [bm25_ids, semantic_ids], k=k, rrf_constant=self.rrf_constant
        )
        return {
            "bm25": bm25_ids[:k],
            "semantic": semantic_ids[:k],
            "hybrid": hybrid_ids,
        }

    def retrieve(self, query: str, k: int = 5) -> list[int]:
        """Return only the hybrid top-k document IDs."""
        return self.retrieve_all(query, k=k)["hybrid"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default=str(BASE_DIR / "data" / "corpus.jsonl"))
    parser.add_argument("--candidate-k", type=int, default=100)
    parser.add_argument("--rrf-constant", type=int, default=60)
    args = parser.parse_args()

    documents = load_corpus(args.corpus)
    model = SentenceTransformer(MODEL_NAME)
    retriever = HybridRetriever(
        documents, model, candidate_k=args.candidate_k, rrf_constant=args.rrf_constant
    )
    rankings = retriever.retrieve_all(CLAIM, k=5)
    print(f"Scientific abstracts: {len(documents)}")
    for name, ids in rankings.items():
        print(f"{name.capitalize()} top 5 document IDs: {ids}")
        print(f"{name.capitalize()} Recall@5: {recall_at_k(ids, EVIDENCE_IDS, 5):.4f}")


if __name__ == "__main__":
    main()
