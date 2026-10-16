"""Evaluate a single BM25 index on SciFact development claims.

Usage:
    python evaluate_bm25.py
    python evaluate_bm25.py --corpus data/corpus.jsonl --claims data/claims_dev.jsonl

Requires:
    pip install rank-bm25 numpy

Place this file next to bm25_retriever.py and retrieval_metrics.py.
"""

import argparse
import json
from pathlib import Path

from rank_bm25 import BM25Okapi

from bm25_retriever import load_corpus, tokenize
from retrieval_metrics import recall_at_k


KS = (5, 10, 20)
BASE_DIR = Path(__file__).resolve().parent


def load_claims(claims_path: str) -> list[dict]:
    """Load SciFact claims, one JSON object per line."""
    claims = []
    with open(claims_path, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                claims.append(json.loads(line))
    return claims


def evaluate_bm25(corpus_path: str, claims_path: str) -> dict:
    """Measure mean Recall@5/10/20 for development claims with evidence."""
    documents = load_corpus(corpus_path)
    claims = load_claims(claims_path)

    if not documents:
        raise ValueError("The SciFact corpus is empty; cannot build BM25 index")

    # Build the BM25 index ONCE. Each claim below only calls get_scores().
    tokenized_documents = [
        tokenize(doc["title"] + " " + " ".join(doc["abstract"]))
        for doc in documents
    ]
    bm25 = BM25Okapi(tokenized_documents)
    doc_ids = [int(doc["doc_id"]) for doc in documents]

    total_recall = {k: 0.0 for k in KS}
    evaluated = 0
    skipped = 0

    for claim in claims:
        # SciFact evidence keys are document IDs stored as JSON strings.
        # Do NOT use cited_doc_ids here: citations are not evidence labels.
        relevant_ids = {int(doc_id) for doc_id in (claim.get("evidence") or {})}
        if not relevant_ids:
            skipped += 1
            continue

        scores = bm25.get_scores(tokenize(claim["claim"]))
        ranked_indices = sorted(
            range(len(doc_ids)), key=lambda i: -scores[i]
        )[:max(KS)]
        retrieved_ids = [doc_ids[i] for i in ranked_indices]

        for k in KS:
            total_recall[k] += recall_at_k(retrieved_ids, relevant_ids, k)
        evaluated += 1

    mean_recall = (
        {k: total_recall[k] / evaluated for k in KS} if evaluated else {}
    )

    print(f"Scientific abstracts:   {len(documents)}")
    print(f"Development claims:     {len(claims)}")
    print(f"Claims evaluated:       {evaluated}")
    print(f"Claims skipped:         {skipped} (no annotated evidence)")

    if not evaluated:
        print("No annotated evidence found; mean Recall@k is undefined.")
    else:
        print("\nBM25 development results:")
        for k in KS:
            print(f"Mean Recall@{k:<2}: {mean_recall[k]:.4f}")

    return {
        "documents": len(documents),
        "claims": len(claims),
        "evaluated": evaluated,
        "skipped": skipped,
        "mean_recall": mean_recall,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus", default=str(BASE_DIR / "data" / "corpus.jsonl"),
        help="Path to SciFact corpus.jsonl",
    )
    parser.add_argument(
        "--claims", default=str(BASE_DIR / "data" / "claims_dev.jsonl"),
        help="Path to SciFact claims_dev.jsonl",
    )
    args = parser.parse_args()
    evaluate_bm25(args.corpus, args.claims)


if __name__ == "__main__":
    main()
