"""Evaluate a pretrained cross-encoder on the exact hybrid top-20 candidates.

This REUSES retrieval_claim_analysis.csv created by analyze_retrieval.py,
so the BM25/semantic indexes do not need to be recomputed. All 188 eligible
SciFact development claims are evaluated using their cached hybrid top-20.

Run: python evaluate_reranker.py
Optional: python evaluate_reranker.py --batch-size 8

The script is an INFERENCE benchmark; no SciFact labels are used for training.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sentence_transformers import CrossEncoder

from bm25_retriever import load_corpus
from evaluate_bm25 import load_claims
from rerank_evidence import (
    RERANKER_MODEL,
    document_text,
    reciprocal_rank_at_k,
    rerank_from_scores,
)
from retrieval_metrics import recall_at_k


BASE_DIR = Path(__file__).resolve().parent
KS = (5, 10, 20)


def load_cached_candidates(csv_path: str, claims: list[dict]) -> list[dict]:
    """Validate that the candidate CSV exactly matches eligible dev claims."""
    eligible = {
        int(claim["id"]): claim
        for claim in claims
        if claim.get("evidence")
    }

    records = []
    seen = set()
    with open(csv_path, "r", newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            claim_id = int(row["claim_id"])
            if claim_id in seen:
                raise ValueError(f"Duplicate claim ID in cache: {claim_id}")
            seen.add(claim_id)
            if claim_id not in eligible:
                raise ValueError(f"Unexpected/non-eligible claim ID: {claim_id}")

            claim = eligible[claim_id]
            evidence = {int(x) for x in json.loads(row["evidence_doc_ids"])}
            expected_evidence = {int(x) for x in claim["evidence"]}
            if evidence != expected_evidence or row["claim"] != claim["claim"]:
                raise ValueError(f"Cached annotations differ for claim {claim_id}")

            candidates = [int(x) for x in json.loads(row["hybrid_top20"])]
            if len(candidates) != 20 or len(set(candidates)) != 20:
                raise ValueError(f"Claim {claim_id} does not have 20 unique candidates")

            # Check the exported baseline against our current metric code.
            for k in KS:
                cached = float(row[f"hybrid_recall_at_{k}"])
                measured = recall_at_k(candidates, evidence, k)
                if not np.isclose(cached, measured, atol=1e-10):
                    raise ValueError(f"Cached Recall@{k} differs for claim {claim_id}")

            records.append({
                "claim_id": claim_id,
                "claim": claim["claim"],
                "relevant_ids": evidence,
                "hybrid_ids": candidates,
            })

    if seen != eligible.keys():
        missing = set(eligible) - seen
        raise ValueError(f"Candidate cache does not match development set; missing IDs: {sorted(missing)}")
    return records


def evaluate_reranker(
    corpus_path: str,
    claims_path: str,
    cache_path: str,
    output_path: str,
    model_name: str = RERANKER_MODEL,
    batch_size: int = 16,
) -> dict:
    """Rerank exact hybrid candidates and compare per-claim recall and MRR."""
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")

    documents = load_corpus(corpus_path)
    claims = load_claims(claims_path)
    document_lookup = {int(doc["doc_id"]): doc for doc in documents}
    if len(document_lookup) != len(documents):
        raise ValueError("Corpus contains duplicate document IDs")
    records = load_cached_candidates(cache_path, claims)
    skipped = len(claims) - len(records)

    print(f"Scientific abstracts: {len(documents)}")
    print(f"Development claims:   {len(claims)}")
    print(f"Claims evaluated:     {len(records)}")
    print(f"Claims skipped:       {skipped} (no annotated evidence)")
    print(f"Candidate pool:       cached hybrid top 20")
    print(f"Cross-encoder model:  {model_name}")

    if not records:
        print("No annotated evidence found; Recall@k is undefined.")
        return {"evaluated": 0, "skipped": skipped, "mean": {}}

    # Assemble all claim–paper pairs once. This avoids rebuilding the encoder
    # or invoking it separately for every scientific claim.
    pairs = []
    for rec in records:
        for doc_id in rec["hybrid_ids"]:
            if doc_id not in document_lookup:
                raise ValueError(f"Hybrid candidate {doc_id} is absent from corpus")
            pairs.append((rec["claim"], document_text(document_lookup[doc_id])))

    print(f"Claim–document pairs: {len(pairs)}")
    model = CrossEncoder(model_name)
    scores = np.asarray(model.predict(
        pairs, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=True
    )).reshape(-1)
    if len(scores) != len(pairs):
        raise ValueError("Cross-encoder did not return one score per input pair")

    totals = {method: {k: 0.0 for k in KS} for method in ("hybrid", "reranked")}
    mrr_totals = {method: 0.0 for method in ("hybrid", "reranked")}
    comparison = []
    cursor = 0
    for rec in records:
        original = rec["hybrid_ids"]
        n = len(original)
        reranked = rerank_from_scores(original, scores[cursor:cursor + n], k=n)
        cursor += n
        relevant_ids = rec["relevant_ids"]
        assert set(reranked) == set(original), "Reranker changed the candidate set"

        row = {
            "claim_id": rec["claim_id"],
            "claim": rec["claim"],
            "evidence_doc_ids": json.dumps(sorted(relevant_ids)),
            "hybrid_top20": json.dumps(original),
            "reranked_top20": json.dumps(reranked),
        }
        for k in KS:
            base = recall_at_k(original, relevant_ids, k)
            new = recall_at_k(reranked, relevant_ids, k)
            totals["hybrid"][k] += base
            totals["reranked"][k] += new
            row[f"hybrid_recall_at_{k}"] = base
            row[f"reranked_recall_at_{k}"] = new
            if k == 20 and base != new:
                raise AssertionError("Reranking fixed top-20 candidates must preserve Recall@20")

        for method, ranking in (("hybrid", original), ("reranked", reranked)):
            mrr = reciprocal_rank_at_k(ranking, relevant_ids, 5)
            mrr_totals[method] += mrr
            row[f"{method}_rr_at_5"] = mrr
        comparison.append(row)

    if cursor != len(scores):
        raise AssertionError("Not all reranker scores were consumed")

    count = len(records)
    means = {
        method: {k: totals[method][k] / count for k in KS}
        for method in ("hybrid", "reranked")
    }
    mean_mrr5 = {m: mrr_totals[m] / count for m in ("hybrid", "reranked")}

    print("\nMean per-claim Recall@k (same 188 eligible claims):")
    print(f"{'Metric':<14}{'Hybrid':>12}{'Reranked':>12}{'Delta (pp)':>13}")
    for k in KS:
        baseline = means["hybrid"][k]
        reranked = means["reranked"][k]
        print(f"Recall@{k:<7}{baseline:>12.4f}{reranked:>12.4f}{100 * (reranked - baseline):>+13.2f}")
    print(f"MRR@5          {mean_mrr5['hybrid']:>12.4f}{mean_mrr5['reranked']:>12.4f}"
          f"{100 * (mean_mrr5['reranked'] - mean_mrr5['hybrid']):>+13.2f}")
    print("\nRecall@20 must remain unchanged: only candidate ordering changed.")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=comparison[0].keys())
        writer.writeheader()
        writer.writerows(comparison)
    print(f"Saved per-claim reranking results: {path.resolve()}")

    return {
        "documents": len(documents), "claims": len(claims),
        "evaluated": count, "skipped": skipped,
        "mean_recall": means, "mean_mrr_at_5": mean_mrr5,
        "output_path": str(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default=str(BASE_DIR / "data" / "corpus.jsonl"))
    parser.add_argument("--claims", default=str(BASE_DIR / "data" / "claims_dev.jsonl"))
    parser.add_argument("--cache", default=str(BASE_DIR / "docs/retrieval_claim_analysis.csv"))
    parser.add_argument("--output", default=str(BASE_DIR / "reports/reranking_claim_analysis.csv"))
    parser.add_argument("--model", default=RERANKER_MODEL)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    evaluate_reranker(args.corpus, args.claims, args.cache, args.output, args.model, args.batch_size)


if __name__ == "__main__":
    main()
