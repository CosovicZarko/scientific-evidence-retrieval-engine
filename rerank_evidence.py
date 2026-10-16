"""Rerank a fixed list of SciFact candidate papers with a pretrained cross-encoder.

Install: pip install sentence-transformers numpy rank-bm25
Run the claim-47 demo: python rerank_evidence.py

This is inference only: the cross-encoder is NOT fine-tuned on SciFact here.
It scores query/document pairs jointly. Unlike a retriever, it cannot add
papers that are missing from the supplied candidates.
"""

import argparse
from pathlib import Path

import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer

from bm25_retriever import CLAIM, EVIDENCE_IDS, load_corpus
from hybrid_retriever import HybridRetriever
from retrieval_metrics import recall_at_k
from semantic_retriever import MODEL_NAME


BASE_DIR = Path(__file__).resolve().parent
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"


def document_text(document: dict) -> str:
    """Use the same title + abstract source text as our retrieval baselines."""
    return document["title"] + " " + " ".join(document["abstract"])


def rerank_from_scores(
    candidate_ids: list[int],
    scores: list[float] | np.ndarray,
    k: int = 5,
) -> list[int]:
    """Return candidates sorted by descending cross-encoder score, stable on ties.

    Does not introduce any new document IDs. `scores[i]` corresponds to
    `candidate_ids[i]`. This helper is separately testable without a model.
    """
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer")
    if len(candidate_ids) != len(scores):
        raise ValueError("One score is required for each candidate document")
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("Candidate document IDs must be unique")

    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("Scores must be a finite one-dimensional array")

    ranked_indices = sorted(range(len(candidate_ids)), key=lambda i: (-values[i], i))
    return [candidate_ids[i] for i in ranked_indices[:k]]


def rerank_documents(
    query: str,
    candidate_ids: list[int],
    documents_by_id: dict[int, dict],
    reranker: CrossEncoder,
    k: int = 5,
    batch_size: int = 16,
) -> list[int]:
    """Score query–document pairs jointly, then rerank only those candidates."""
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if not candidate_ids:
        return []
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("Candidate document IDs must be unique")

    # Fail loudly rather than silently losing a candidate due to an ID mismatch.
    pairs = [(query, document_text(documents_by_id[doc_id])) for doc_id in candidate_ids]
    scores = reranker.predict(
        pairs, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=False
    )
    return rerank_from_scores(candidate_ids, np.asarray(scores).reshape(-1), k=k)


def reciprocal_rank_at_k(
    retrieved_ids: list[int], relevant_ids: set[int], k: int
) -> float:
    """First relevant document's reciprocal rank (0 if none in the first k)."""
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer")
    if not relevant_ids:
        raise ValueError("relevant_ids must not be empty")
    for rank, doc_id in enumerate(retrieved_ids[:k], start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default=str(BASE_DIR / "data" / "corpus.jsonl"))
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    documents = load_corpus(args.corpus)
    documents_by_id = {int(doc["doc_id"]): doc for doc in documents}
    if len(documents_by_id) != len(documents):
        raise ValueError("Corpus contains duplicate document IDs")

    # Reuse the same hybrid retrieval settings as our prior benchmark.
    hybrid = HybridRetriever(
        documents, SentenceTransformer(MODEL_NAME),
        candidate_k=100, rrf_constant=60
    )
    candidates = hybrid.retrieve(CLAIM, k=20)

    # This model scores query/document pairs, not standalone embeddings.
    reranker = CrossEncoder(RERANKER_MODEL)
    reranked = rerank_documents(
        CLAIM, candidates, documents_by_id, reranker,
        k=20, batch_size=args.batch_size
    )

    print(f"Scientific abstracts: {len(documents)}")
    print(f"Hybrid top 5 IDs:     {candidates[:5]}")
    print(f"Reranked top 5 IDs:   {reranked[:5]}")
    print(f"Evidence IDs:         {sorted(EVIDENCE_IDS)}")
    print(f"Hybrid Recall@5:      {recall_at_k(candidates, EVIDENCE_IDS, 5):.4f}")
    print(f"Reranked Recall@5:    {recall_at_k(reranked, EVIDENCE_IDS, 5):.4f}")
    print(f"Hybrid MRR@5:         {reciprocal_rank_at_k(candidates, EVIDENCE_IDS, 5):.4f}")
    print(f"Reranked MRR@5:       {reciprocal_rank_at_k(reranked, EVIDENCE_IDS, 5):.4f}")
    assert set(candidates) == set(reranked), "Reranking changed the candidate pool"
    assert recall_at_k(candidates, EVIDENCE_IDS, 20) == recall_at_k(
        reranked, EVIDENCE_IDS, 20
    ), "Recall@20 must not change when the same 20 documents are reordered"


if __name__ == "__main__":
    main()
