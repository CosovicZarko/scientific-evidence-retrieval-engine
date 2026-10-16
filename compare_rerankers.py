"""Compare pretrained and SciFact-fine-tuned rerankers on the SAME dev candidates.

Prerequisites:
  - retrieval_claim_analysis.csv from analyze_retrieval.py
  - reranking_claim_analysis.csv from the original evaluate_reranker.py run
  - checkpoints/scifact_reranker/ from fine_tune_reranker.py

Run:
    python compare_rerankers.py

The script only evaluates the fine-tuned model (the original results are cached),
then compares their per-claim Recall@5/10/20 and reciprocal rank at 5. No training
occurs and the SciFact dev annotations are used only for evaluation.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from retrieval_metrics import recall_at_k


BASE_DIR = Path(__file__).resolve().parent
CUTOFFS = (5, 10, 20)


def reciprocal_rank_at_5(ranking: list[int], relevant: set[int]) -> float:
    for rank, doc_id in enumerate(ranking[:5], start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def load_ranking_csv(path: Path) -> dict[int, dict]:
    """Read one ranking per annotated claim; reject duplicate or invalid rows."""
    rows = {}
    with path.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            claim_id = int(row["claim_id"])
            if claim_id in rows:
                raise ValueError(f"Duplicate claim ID in {path}: {claim_id}")
            relevant = {int(x) for x in json.loads(row["evidence_doc_ids"])}
            hybrid = [int(x) for x in json.loads(row["hybrid_top20"])]
            ranked = [int(x) for x in json.loads(row["reranked_top20"])]
            if not relevant:
                raise ValueError(f"Claim {claim_id} has no evidence")
            if len(hybrid) != 20 or len(set(hybrid)) != 20:
                raise ValueError(f"Invalid hybrid top 20 for claim {claim_id}")
            if len(ranked) != 20 or set(ranked) != set(hybrid):
                raise ValueError(f"Reranking changed candidates for claim {claim_id}")
            rows[claim_id] = {
                "claim": row["claim"],
                "relevant": relevant,
                "hybrid": hybrid,
                "ranked": ranked,
            }
    if not rows:
        raise ValueError(f"No evaluated claims in {path}")
    return rows


def paired_ci(deltas: np.ndarray, bootstrap_samples: int, seed: int) -> tuple[float, float]:
    """Paired percentile bootstrap over CLAIMS (not claim-document pairs)."""
    rng = np.random.default_rng(seed)
    n = len(deltas)
    means = np.empty(bootstrap_samples, dtype=float)
    for i in range(bootstrap_samples):
        means[i] = deltas[rng.integers(0, n, size=n)].mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def compare(
    pretrained_csv: Path,
    finetuned_csv: Path,
    output_csv: Path,
    bootstrap_samples: int = 10000,
    seed: int = 42,
) -> dict:
    """Compare two rankings, validating identical queries and candidate pools."""
    if bootstrap_samples < 100:
        raise ValueError("bootstrap_samples must be at least 100")
    pretrained = load_ranking_csv(pretrained_csv)
    finetuned = load_ranking_csv(finetuned_csv)
    if pretrained.keys() != finetuned.keys():
        missing = sorted(set(pretrained) - set(finetuned))
        extra = sorted(set(finetuned) - set(pretrained))
        raise ValueError(f"Claim IDs do not match (missing={missing[:5]}, extra={extra[:5]})")

    claim_ids = sorted(pretrained)
    metrics = [f"Recall@{k}" for k in CUTOFFS] + ["MRR@5"]
    values = {model: {metric: [] for metric in metrics} for model in ("pretrained", "finetuned")}
    rows = []

    for claim_id in claim_ids:
        old, new = pretrained[claim_id], finetuned[claim_id]
        if (old["claim"] != new["claim"] or old["relevant"] != new["relevant"]
                or old["hybrid"] != new["hybrid"]):
            raise ValueError(f"Non-identical query, annotations, or candidate pool: claim {claim_id}")
        row = {"claim_id": claim_id, "claim": old["claim"]}
        for name, record in (("pretrained", old), ("finetuned", new)):
            ranked = record["ranked"]
            for k in CUTOFFS:
                metric = f"Recall@{k}"
                score = recall_at_k(ranked, old["relevant"], k)
                values[name][metric].append(score)
                row[f"{name}_recall_at_{k}"] = score
                if k == 20 and score != recall_at_k(old["hybrid"], old["relevant"], 20):
                    raise AssertionError(f"Recall@20 changed for claim {claim_id}")
            mrr = reciprocal_rank_at_5(ranked, old["relevant"])
            values[name]["MRR@5"].append(mrr)
            row[f"{name}_mrr_at_5"] = mrr
        rows.append(row)

    print(f"SciFact development claims evaluated: {len(claim_ids)}")
    print("Candidate pool: identical cached hybrid top-20 for each claim")
    print(f"{'Metric':<12}{'Pretrained':>13}{'Fine-tuned':>13}{'Delta (pp)':>13}{'95% paired CI (pp)':>24}")
    summary = {}
    for metric in metrics:
        before = np.asarray(values["pretrained"][metric], dtype=float)
        after = np.asarray(values["finetuned"][metric], dtype=float)
        delta = after - before
        lower, upper = paired_ci(delta, bootstrap_samples, seed)
        better = int(np.count_nonzero(delta > 1e-12))
        worse = int(np.count_nonzero(delta < -1e-12))
        tied = len(delta) - better - worse
        print(f"{metric:<12}{before.mean():>13.4f}{after.mean():>13.4f}{100*delta.mean():>+13.2f}"
              f"  [{100*lower:+.2f}, {100*upper:+.2f}]")
        print(f"  Better / worse / tied claims: {better} / {worse} / {tied}")
        summary[metric] = {
            "pretrained": float(before.mean()),
            "finetuned": float(after.mean()),
            "delta": float(delta.mean()),
            "ci_95": (lower, upper),
            "better": better, "worse": worse, "tied": tied,
        }

    if output_csv:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with output_csv.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved paired comparison: {output_csv.resolve()}")
    print("Note: exploratory development-set CIs; not an untouched test set.")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=BASE_DIR / "checkpoints/scifact_reranker")
    parser.add_argument("--pretrained-results", type=Path, default=BASE_DIR / "docs/reranking_claim_analysis.csv")
    parser.add_argument("--finetuned-results", type=Path, default=BASE_DIR / "docs/reranking_finetuned_analysis.csv")
    parser.add_argument("--comparison-output", type=Path, default=BASE_DIR / "reports/reranker_comparison.csv")
    parser.add_argument("--corpus", type=Path, default=BASE_DIR / "data/corpus.jsonl")
    parser.add_argument("--claims", type=Path, default=BASE_DIR / "data/claims_dev.jsonl")
    parser.add_argument("--cache", type=Path, default=BASE_DIR / "docs/retrieval_claim_analysis.csv")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-eval", action="store_true", help="Only compare existing evaluation CSVs")
    args = parser.parse_args()

    if not args.pretrained_results.is_file():
        parser.error(
            f"Missing pretrained evaluation: {args.pretrained_results}. "
            "Run 'python evaluate_reranker.py' first."
        )
    if args.pretrained_results.resolve() == args.finetuned_results.resolve():
        parser.error("Pretrained and fine-tuned CSV paths must differ")

    if not args.skip_eval:
        if not (args.checkpoint / "config.json").is_file():
            parser.error(f"Missing checkpoint config.json: {args.checkpoint}")
        from evaluate_reranker import evaluate_reranker
        print("Evaluating the fine-tuned model against the SAME cached hybrid top-20 ...")
        evaluate_reranker(
            str(args.corpus), str(args.claims), str(args.cache),
            str(args.finetuned_results), model_name=str(args.checkpoint.resolve()),
            batch_size=args.batch_size,
        )
        print("\nPaired pretrained vs fine-tuned comparison:\n")
    elif not args.finetuned_results.is_file():
        parser.error(f"Missing fine-tuned results: {args.finetuned_results}")

    compare(args.pretrained_results, args.finetuned_results,
            args.comparison_output, args.bootstrap_samples, args.seed)


if __name__ == "__main__":
    main()
