"""Compare a pretrained cross-encoder sentence selector against TF-IDF on SciFact.

Usage:
    python evaluate_neural_sentence_selector.py

Evaluates only annotated claim/document pairs on SciFact development data.
Gold sentence indices are used ONLY for evaluation, not sentence ranking.
This is not end-to-end document retrieval or stance classification.

The cross-encoder runs once over all claim/sentence pairs (batched internally).
It does not train or update any model weights.
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from evidence_sentence_selector import select_evidence_sentences
from evaluate_sentence_selector import evidence_sets_for_pair
from prepare_evidence_classification_data import load_jsonl

BASE_DIR = Path(__file__).resolve().parent
MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L6-v2"
CUTOFFS = (1, 2)


def prepare_pairs(corpus_path: Path, claims_path: Path):
    """Collect (claim, sentence) inputs and claim/document evaluation units."""
    corpus = load_jsonl(corpus_path)
    claims = load_jsonl(claims_path)
    documents = {int(doc["doc_id"]): doc for doc in corpus}
    if len(documents) != len(corpus):
        raise ValueError("Duplicate document IDs in corpus")
    if len({int(claim["id"]) for claim in claims}) != len(claims):
        raise ValueError("Duplicate claim IDs in claims file")

    sentence_pairs = []
    units = []
    skipped = 0
    total_groups = 0
    for claim in claims:
        evidence = claim.get("evidence") or {}
        if not evidence:
            skipped += 1
            continue
        for doc_key, annotations in evidence.items():
            doc_id = int(doc_key)
            if doc_id not in documents:
                raise ValueError(f"Missing document {doc_id} for claim {claim['id']}")
            abstract = documents[doc_id]["abstract"]
            if not isinstance(abstract, list) or not abstract:
                raise ValueError(f"Missing/invalid abstract for document {doc_id}")
            groups = evidence_sets_for_pair(annotations, len(abstract))
            if not groups:
                raise ValueError(f"No evidence sets for claim {claim['id']} / document {doc_id}")
            offset = len(sentence_pairs)
            sentence_pairs.extend((claim["claim"], text) for text in abstract)
            units.append({
                "claim_id": int(claim["id"]),
                "doc_id": doc_id,
                "claim": claim["claim"],
                "abstract": abstract,
                "groups": groups,
                "offset": offset,
            })
            total_groups += len(groups)
    if not units:
        raise ValueError("No annotated claim-document pairs to evaluate")
    return len(corpus), len(claims), skipped, total_groups, units, sentence_pairs


def ranking_metrics(ranked_indices: list[int], groups: list[set[int]], k: int) -> dict:
    """Evaluate the SAME union and complete-group definitions as TF-IDF."""
    gold_union = set().union(*groups)
    if not gold_union:
        raise ValueError("An annotated pair must have at least one gold sentence")
    selected = set(ranked_indices[:k])
    return {
        "recall": len(selected & gold_union) / len(gold_union),
        "any_hit": int(bool(selected & gold_union)),
        "complete_set_hit": int(any(group <= selected for group in groups)),
    }


def evaluate(corpus_path: Path, claims_path: Path, output_path: Path,
             model, batch_size: int = 16) -> dict:
    """Evaluate a cross-encoder-style model with .predict() on all sentences."""
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")

    n_corpus, n_claims, skipped, n_groups, units, sentence_pairs = prepare_pairs(
        corpus_path, claims_path
    )
    scores = np.asarray(model.predict(
        sentence_pairs,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=True,
    ))
    if scores.size != len(sentence_pairs) or not np.all(np.isfinite(scores)):
        raise ValueError("Model must return one finite relevance score per sentence pair")
    scores = scores.reshape(-1)

    methods = ("tfidf", "neural")
    totals = {
        method: {k: Counter() for k in CUTOFFS}
        for method in methods
    }
    rows = []
    for unit in units:
        abstract = unit["abstract"]
        offset = unit["offset"]
        neural_ranking = sorted(
            range(len(abstract)),
            key=lambda i: (-float(scores[offset + i]), i),
        )[:max(CUTOFFS)]
        tfidf_ranking = select_evidence_sentences(
            unit["claim"], abstract, k=max(CUTOFFS)
        )
        rankings = {"tfidf": tfidf_ranking, "neural": neural_ranking}
        groups = unit["groups"]
        gold_union = sorted(set().union(*groups))
        row = {
            "claim_id": unit["claim_id"],
            "doc_id": unit["doc_id"],
            "gold_sentence_indices": json.dumps(gold_union),
            "annotated_evidence_sets": json.dumps([sorted(g) for g in groups]),
            "tfidf_selected_indices": json.dumps(tfidf_ranking),
            "neural_selected_indices": json.dumps(neural_ranking),
        }
        for method, ranking in rankings.items():
            for k in CUTOFFS:
                metrics = ranking_metrics(ranking, groups, k)
                totals[method][k].update(metrics)
                row[f"{method}_recall_at_{k}"] = round(metrics["recall"], 6)
                row[f"{method}_any_hit_at_{k}"] = metrics["any_hit"]
                row[f"{method}_complete_set_hit_at_{k}"] = metrics["complete_set_hit"]
        rows.append(row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    n = len(units)
    print(f"Scientific abstracts: {n_corpus}")
    print(f"Development claims:   {n_claims}")
    print(f"Claims with evidence: {n_claims - skipped}")
    print(f"Claims skipped:       {skipped}")
    print(f"Annotated claim-doc pairs: {n}")
    print(f"Distinct evidence groups:  {n_groups}")
    print(f"Claim-sentence pairs scored: {len(sentence_pairs)}")
    print("\nSentence selection comparison (macro-averaged over claim-doc pairs):")
    print(f"{'Metric':<29} {'TF-IDF':>10} {'Neural':>10} {'Delta pp':>11}")
    for k in CUTOFFS:
        for field, label in (
            ("recall", "Mean sentence Recall"),
            ("any_hit", "Any gold sentence Hit"),
            ("complete_set_hit", "Complete evidence set"),
        ):
            baseline = totals["tfidf"][k][field] / n
            neural = totals["neural"][k][field] / n
            print(f"{label + '@' + str(k):<29} {baseline:>10.4f} {neural:>10.4f} {(neural - baseline)*100:>+10.2f}")
    print("\nGold-annotated claim/document pairs only; not end-to-end performance.")
    print("Multiple pairs from one claim are dependent; results are exploratory.")
    print("This pretrained cross-encoder was NOT fine-tuned for sentence selection.")
    print(f"Saved per-pair comparison: {output_path}")
    return {
        "evaluated_pairs": n, "annotated_groups": n_groups,
        "scored_sentences": len(sentence_pairs), "totals": totals,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=BASE_DIR / "data" / "corpus.jsonl")
    parser.add_argument("--claims", type=Path, default=BASE_DIR / "data" / "claims_dev.jsonl")
    parser.add_argument("--output", type=Path, default=BASE_DIR / "sentence_selection_neural_analysis.csv")
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    from sentence_transformers import CrossEncoder
    print(f"Loading cross-encoder: {args.model}")
    model = CrossEncoder(args.model)
    evaluate(args.corpus, args.claims, args.output, model, args.batch_size)


if __name__ == "__main__":
    main()
