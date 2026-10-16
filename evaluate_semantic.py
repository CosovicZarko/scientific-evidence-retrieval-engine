"""Evaluate pretrained semantic retrieval on SciFact development claims.

Install: pip install sentence-transformers rank-bm25 numpy
Run:     python evaluate_semantic.py
Or:      python evaluate_semantic.py --corpus data/corpus.jsonl --claims data/claims_dev.jsonl

Uses the same evidence-only claim filtering and macro-averaged Recall@k
as evaluate_bm25.py for a fair comparison.
"""

import argparse
from pathlib import Path

from sentence_transformers import SentenceTransformer

from bm25_retriever import load_corpus
from evaluate_bm25 import load_claims
from retrieval_metrics import recall_at_k
from semantic_retriever import MODEL_NAME, build_embeddings, retrieve_semantic


KS = (5, 10, 20)
BASE_DIR = Path(__file__).resolve().parent


def evaluate_semantic(corpus_path: str, claims_path: str) -> dict:
    """Measure mean Recall@5, @10, and @20 on annotated dev claims."""
    documents = load_corpus(corpus_path)
    claims = load_claims(claims_path)

    if not documents:
        raise ValueError("The SciFact corpus is empty; cannot build embeddings")

    # Initialize the pretrained neural encoder and encode the corpus ONCE.
    model = SentenceTransformer(MODEL_NAME)
    embeddings = build_embeddings(documents, model)

    recall_totals = {k: 0.0 for k in KS}
    evaluated = 0
    skipped = 0

    for claim in claims:
        # Identical eligibility rule to evaluate_bm25.py. SciFact's
        # evidence keys are annotated document IDs (not cited_doc_ids).
        relevant_ids = {
            int(doc_id) for doc_id in (claim.get("evidence") or {})
        }
        if not relevant_ids:
            skipped += 1
            continue

        retrieved_ids = retrieve_semantic(
            claim["claim"], documents, embeddings, model, k=max(KS)
        )

        for k in KS:
            recall_totals[k] += recall_at_k(retrieved_ids, relevant_ids, k)
        evaluated += 1

    mean_recall = (
        {k: recall_totals[k] / evaluated for k in KS}
        if evaluated else {}
    )

    print(f"Scientific abstracts:   {len(documents)}")
    print(f"Development claims:     {len(claims)}")
    print(f"Claims evaluated:       {evaluated}")
    print(f"Claims skipped:         {skipped} (no annotated evidence)")

    if not evaluated:
        print("No annotated evidence found; mean Recall@k is undefined.")
    else:
        print("\nSemantic development results:")
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
    evaluate_semantic(args.corpus, args.claims)


if __name__ == "__main__":
    main()
