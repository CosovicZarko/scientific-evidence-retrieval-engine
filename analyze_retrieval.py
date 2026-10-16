"""Analyze paired SciFact development retrieval results and RRF improvements.

Run:
    python analyze_retrieval.py

Install dependencies if needed:
    pip install rank-bm25 sentence-transformers numpy

Builds BM25 and document embeddings once. All comparisons use the exact
same development claims with annotated evidence. No models are trained.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


KS = (5, 10, 20)
METHODS = ("bm25", "semantic", "hybrid")
BASE_DIR = Path(__file__).resolve().parent


def evaluate_claims(claims: list[dict], retriever) -> tuple[list[dict], int]:
    """Collect per-claim rankings and recall, skipping no-evidence claims."""
    # Delayed import makes the statistical helpers testable without a model.
    from retrieval_metrics import recall_at_k

    rows = []
    skipped = 0

    for claim in claims:
        relevant = {int(doc_id) for doc_id in (claim.get("evidence") or {})}
        if not relevant:
            skipped += 1
            continue

        rankings = retriever.retrieve_all(claim["claim"], k=max(KS))
        row = {
            "claim_id": claim["id"],
            "claim": claim["claim"],
            "evidence_doc_ids": json.dumps(sorted(relevant)),
            "evidence_count": len(relevant),
        }
        for method in METHODS:
            ids = [int(doc_id) for doc_id in rankings[method]]
            if len(ids) != len(set(ids)):
                raise ValueError(f"Duplicate IDs from {method} for claim {claim['id']}")
            row[f"{method}_top20"] = json.dumps(ids)
            for k in KS:
                row[f"{method}_recall_at_{k}"] = recall_at_k(ids, relevant, k)
        rows.append(row)

    return rows, skipped


def paired_bootstrap_ci(
    deltas: np.ndarray,
    iterations: int = 10000,
    seed: int = 42,
) -> tuple[float, float]:
    """95% percentile CI for mean paired (hybrid - baseline) recall.

    Resample *claims*, not individual retrieved documents, preserving
    the pairing of retrieval models for every claim.
    """
    values = np.asarray(deltas, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("deltas must be a nonempty 1-D array")
    if iterations <= 0:
        raise ValueError("iterations must be positive")

    rng = np.random.default_rng(seed)
    means = np.empty(iterations, dtype=float)
    # Looping avoids allocating a potentially huge (iterations x claims) array.
    for i in range(iterations):
        sample = rng.integers(0, len(values), size=len(values))
        means[i] = values[sample].mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def summarize(rows: list[dict], iterations: int, seed: int) -> dict:
    """Produce means, paired CIs, and evidence-level overlap diagnostics."""
    if not rows:
        raise ValueError("No development claims with annotated evidence")

    summary = {"evaluated": len(rows), "metrics": {}}
    for k in KS:
        per_method = {
            method: np.asarray(
                [r[f"{method}_recall_at_{k}"] for r in rows], dtype=float
            )
            for method in METHODS
        }
        means = {name: float(scores.mean()) for name, scores in per_method.items()}

        comparisons = {}
        for baseline in ("bm25", "semantic"):
            deltas = per_method["hybrid"] - per_method[baseline]
            lo, hi = paired_bootstrap_ci(deltas, iterations, seed)
            comparisons[baseline] = {
                "difference": float(deltas.mean()),
                "ci_low": lo,
                "ci_high": hi,
                "claims_better": int(np.count_nonzero(deltas > 1e-12)),
                "claims_worse": int(np.count_nonzero(deltas < -1e-12)),
                "claims_tied": int(np.count_nonzero(np.abs(deltas) <= 1e-12)),
            }

        # Evidence-level counts to see whether BM25 and semantic genuinely
        # retrieve complementary annotated documents.
        overlap = {"both": 0, "bm25_only": 0, "semantic_only": 0, "neither": 0}
        for row in rows:
            evidence = set(json.loads(row["evidence_doc_ids"]))
            b = set(json.loads(row["bm25_top20"])[:k]) & evidence
            s = set(json.loads(row["semantic_top20"])[:k]) & evidence
            overlap["both"] += len(b & s)
            overlap["bm25_only"] += len(b - s)
            overlap["semantic_only"] += len(s - b)
            overlap["neither"] += len(evidence - (b | s))

        summary["metrics"][str(k)] = {
            "means": means,
            "hybrid_vs": comparisons,
            "evidence_overlap_bm25_semantic": overlap,
        }
    return summary


def save_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def print_results(summary: dict, rows: list[dict], skipped: int) -> None:
    print(f"Claims evaluated: {summary['evaluated']}")
    print(f"Claims skipped:   {skipped} (no annotated evidence)")
    print("\nMean per-claim Recall@k:")
    print(f"{'Metric':<12}{'BM25':>12}{'Semantic':>12}{'Hybrid':>12}")
    for k in KS:
        means = summary["metrics"][str(k)]["means"]
        print(
            f"Recall@{k:<5}"
            f"{means['bm25']:>12.4f}"
            f"{means['semantic']:>12.4f}"
            f"{means['hybrid']:>12.4f}"
        )

    print("\nPaired bootstrap: hybrid improvement (percentage points, 95% CI):")
    for k in KS:
        item = summary["metrics"][str(k)]
        for baseline, c in item["hybrid_vs"].items():
            print(
                f"  Recall@{k:<2} vs {baseline:<8} "
                f"{100*c['difference']:+6.2f} pp "
                f"[95% CI {100*c['ci_low']:+.2f}, {100*c['ci_high']:+.2f}] "
                f"(better/worse/tied claims: "
                f"{c['claims_better']}/{c['claims_worse']}/{c['claims_tied']})"
            )

    print("\nAnnotated evidence overlap between BM25 and semantic retrieval:")
    for k in KS:
        o = summary["metrics"][str(k)]["evidence_overlap_bm25_semantic"]
        print(
            f"  @ {k:<2}: found by both={o['both']}, "
            f"BM25 only={o['bm25_only']}, semantic only={o['semantic_only']}, "
            f"neither={o['neither']}"
        )

    print("\nExamples where hybrid improves Recall@10 over BM25:")
    improved = sorted(
        (r for r in rows if r["hybrid_recall_at_10"] > r["bm25_recall_at_10"]),
        key=lambda r: (r["hybrid_recall_at_10"] - r["bm25_recall_at_10"]),
        reverse=True,
    )
    if not improved:
        print("  No such claims in this run.")
    for r in improved[:3]:
        relevant = set(json.loads(r["evidence_doc_ids"]))
        bhits = set(json.loads(r["bm25_top20"])[:10]) & relevant
        hhits = set(json.loads(r["hybrid_top20"])[:10]) & relevant
        print(f"  Claim {r['claim_id']}: {r['claim'][:115]}")
        print(f"    Evidence recovered by hybrid but not BM25 @10: {sorted(hhits - bhits)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default=str(BASE_DIR / "data" / "corpus.jsonl"))
    parser.add_argument("--claims", default=str(BASE_DIR / "data" / "claims_dev.jsonl"))
    parser.add_argument("--candidate-k", type=int, default=100)
    parser.add_argument("--rrf-constant", type=int, default=60)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=str(BASE_DIR / "reports/retrieval_claim_analysis.csv"))
    args = parser.parse_args()

    # These imports are inside main so statistical functions remain independently testable.
    from sentence_transformers import SentenceTransformer
    from bm25_retriever import load_corpus
    from evaluate_bm25 import load_claims
    from hybrid_retriever import HybridRetriever
    from semantic_retriever import MODEL_NAME

    documents = load_corpus(args.corpus)
    claims = load_claims(args.claims)
    print(f"Scientific abstracts: {len(documents)}")
    print(f"Development claims:   {len(claims)}")
    print(f"RRF candidate depth:  {args.candidate_k} per retriever")
    print(f"RRF rank constant:    {args.rrf_constant}")

    model = SentenceTransformer(MODEL_NAME)
    retriever = HybridRetriever(
        documents, model,
        candidate_k=args.candidate_k,
        rrf_constant=args.rrf_constant,
    )
    rows, skipped = evaluate_claims(claims, retriever)
    if not rows:
        print("No annotated evidence found; no recall statistics available.")
        return

    summary = summarize(rows, iterations=args.bootstrap, seed=args.seed)
    print_results(summary, rows, skipped)
    output = Path(args.output)
    save_csv(rows, output)
    print(f"\nSaved per-claim results: {output}")


if __name__ == "__main__":
    main()
