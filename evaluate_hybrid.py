"""Compare BM25, semantic retrieval, and RRF hybrid on SciFact development.

All three are evaluated on EXACTLY the same claims with annotated evidence.
The corpus indexes and embeddings are built only once.

Run: python evaluate_hybrid.py
Or:  python evaluate_hybrid.py --corpus data/corpus.jsonl --claims data/claims_dev.jsonl
"""

import argparse
from pathlib import Path

from sentence_transformers import SentenceTransformer

from bm25_retriever import load_corpus
from evaluate_bm25 import load_claims
from hybrid_retriever import HybridRetriever
from retrieval_metrics import recall_at_k
from semantic_retriever import MODEL_NAME

KS = (5, 10, 20)
METHODS = ("bm25", "semantic", "hybrid")
BASE_DIR = Path(__file__).resolve().parent


def evaluate_hybrid(
    corpus_path: str,
    claims_path: str,
    candidate_k: int = 100,
    rrf_constant: int = 60,
) -> dict:
    """Compute macro-average Recall@k across evidence-annotated dev claims."""
    documents = load_corpus(corpus_path)
    claims = load_claims(claims_path)

    # One neural encoder and one shared BM25/semantic index, not per claim.
    model = SentenceTransformer(MODEL_NAME)
    retriever = HybridRetriever(
        documents, model, candidate_k=candidate_k, rrf_constant=rrf_constant
    )

    totals = {method: {k: 0.0 for k in KS} for method in METHODS}
    evaluated = 0
    skipped = 0

    for claim in claims:
        # Exactly the same eligibility rule as our previous evaluators.
        relevant_ids = {int(doc_id) for doc_id in (claim.get("evidence") or {})}
        if not relevant_ids:
            skipped += 1
            continue

        rankings = retriever.retrieve_all(claim["claim"], k=max(KS))
        for method in METHODS:
            for k in KS:
                totals[method][k] += recall_at_k(rankings[method], relevant_ids, k)
        evaluated += 1

    mean_recall = (
        {method: {k: totals[method][k] / evaluated for k in KS} for method in METHODS}
        if evaluated else {}
    )

    print(f"Scientific abstracts: {len(documents)}")
    print(f"Development claims:   {len(claims)}")
    print(f"Claims evaluated:     {evaluated}")
    print(f"Claims skipped:       {skipped} (no annotated evidence)")
    print(f"RRF candidate depth:  {candidate_k} per retriever")
    print(f"RRF rank constant:    {rrf_constant}")

    if not evaluated:
        print("No annotated evidence; mean Recall@k is undefined.")
    else:
        print("\nMean per-claim Recall@k (same claims for every method):")
        print(f"{'Metric':<12}{'BM25':>12}{'Semantic':>12}{'Hybrid':>12}")
        for k in KS:
            print(
                f"Recall@{k:<5}"
                f"{mean_recall['bm25'][k]:>12.4f}"
                f"{mean_recall['semantic'][k]:>12.4f}"
                f"{mean_recall['hybrid'][k]:>12.4f}"
            )
        print("\nHybrid minus BM25 / semantic (percentage points):")
        for k in KS:
            versus_bm25 = 100 * (mean_recall["hybrid"][k] - mean_recall["bm25"][k])
            versus_semantic = 100 * (mean_recall["hybrid"][k] - mean_recall["semantic"][k])
            print(f"Recall@{k:<2}: {versus_bm25:+.2f} pp / {versus_semantic:+.2f} pp")

    return {
        "documents": len(documents),
        "claims": len(claims),
        "evaluated": evaluated,
        "skipped": skipped,
        "mean_recall": mean_recall,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default=str(BASE_DIR / "data" / "corpus.jsonl"))
    parser.add_argument("--claims", default=str(BASE_DIR / "data" / "claims_dev.jsonl"))
    parser.add_argument("--candidate-k", type=int, default=100)
    parser.add_argument("--rrf-constant", type=int, default=60)
    args = parser.parse_args()
    evaluate_hybrid(args.corpus, args.claims, args.candidate_k, args.rrf_constant)


if __name__ == "__main__":
    main()
