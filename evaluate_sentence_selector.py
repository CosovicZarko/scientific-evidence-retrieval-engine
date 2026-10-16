"""Evaluate a TF-IDF evidence sentence selector on SciFact development annotations.

Run from the scientific-evidence-engine project directory:
    python evaluate_sentence_selector.py

Only gold-annotated claim/document pairs are scored. The gold annotations are
used for evaluation, never for sentence selection. An abstract can contain
multiple alternative annotated evidence sets.
"""

import argparse
import csv
import json
from pathlib import Path

from evidence_sentence_selector import select_evidence_sentences
from prepare_evidence_classification_data import load_jsonl


BASE_DIR = Path(__file__).resolve().parent
CUTOFFS = (1, 2)


def evidence_sets_for_pair(annotations: list[dict], sentence_count: int) -> list[set[int]]:
    """Return distinct nonempty annotated sets, validating zero-based indices."""
    unique: set[tuple[int, ...]] = set()
    for annotation in annotations:
        indices = annotation.get("sentences")
        if not isinstance(indices, list) or not indices:
            raise ValueError(f"Invalid evidence annotation: {annotation!r}")
        if any(type(i) is not int or i < 0 or i >= sentence_count for i in indices):
            raise ValueError(f"Invalid evidence sentence index: {indices!r}")
        unique.add(tuple(sorted(set(indices))))
    return [set(group) for group in sorted(unique)]


def evaluate_sentence_selector(corpus_path: Path, claims_path: Path, output_path: Path) -> dict:
    corpus = load_jsonl(corpus_path)
    claims = load_jsonl(claims_path)
    documents = {int(doc["doc_id"]): doc for doc in corpus}
    if len(documents) != len(corpus):
        raise ValueError("Duplicate document IDs in corpus")
    if len({int(c["id"]) for c in claims}) != len(claims):
        raise ValueError("Duplicate claim IDs in development set")

    totals = {k: {"recall": 0.0, "any_hit": 0, "complete_set_hit": 0} for k in CUTOFFS}
    evaluated_pairs = 0
    evidence_groups = 0
    skipped = 0
    rows = []

    for claim in claims:
        evidence = claim.get("evidence") or {}
        if not evidence:
            skipped += 1
            continue

        for doc_key, annotations in evidence.items():
            doc_id = int(doc_key)
            if doc_id not in documents:
                raise ValueError(f"Missing evidence document {doc_id} for claim {claim['id']}")

            abstract = documents[doc_id]["abstract"]
            if not isinstance(abstract, list):
                raise ValueError(f"Invalid abstract for document {doc_id}")
            groups = evidence_sets_for_pair(annotations, len(abstract))
            if not groups:
                raise ValueError(f"No evidence sets for claim {claim['id']} document {doc_id}")
            gold_union = set().union(*groups)
            selected = select_evidence_sentences(claim["claim"], abstract, k=max(CUTOFFS))
            if len(set(selected)) != len(selected) or any(i < 0 or i >= len(abstract) for i in selected):
                raise ValueError(f"Invalid selected sentence indices for claim {claim['id']} document {doc_id}")

            row = {
                "claim_id": int(claim["id"]),
                "doc_id": doc_id,
                "gold_sentence_indices": json.dumps(sorted(gold_union)),
                "annotated_evidence_sets": json.dumps([sorted(group) for group in groups]),
                "selected_indices": json.dumps(selected),
            }

            for k in CUTOFFS:
                selected_set = set(selected[:k])
                recall = len(selected_set & gold_union) / len(gold_union)
                any_hit = bool(selected_set & gold_union)
                complete_set_hit = any(group <= selected_set for group in groups)
                totals[k]["recall"] += recall
                totals[k]["any_hit"] += int(any_hit)
                totals[k]["complete_set_hit"] += int(complete_set_hit)
                row[f"recall_at_{k}"] = round(recall, 6)
                row[f"any_hit_at_{k}"] = int(any_hit)
                row[f"complete_set_hit_at_{k}"] = int(complete_set_hit)

            evaluated_pairs += 1
            evidence_groups += len(groups)
            rows.append(row)

    if not evaluated_pairs:
        raise ValueError("No annotated claim-document pairs: sentence recall is undefined")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Scientific abstracts: {len(corpus)}")
    print(f"Development claims:   {len(claims)}")
    print(f"Claims with evidence: {len(claims) - skipped}")
    print(f"Claims skipped:       {skipped} (no annotated evidence)")
    print(f"Evaluated claim-doc pairs: {evaluated_pairs}")
    print(f"Distinct evidence groups within pairs: {evidence_groups}")
    print("\nSentence selection results (macro-averaged over claim-doc pairs):")
    for k in CUTOFFS:
        t = totals[k]
        print(f"Recall@{k}:                 {t['recall'] / evaluated_pairs:.4f}")
        print(f"Any gold sentence Hit@{k}: {t['any_hit'] / evaluated_pairs:.4f} ({t['any_hit']}/{evaluated_pairs})")
        print(f"Complete evidence set@{k}: {t['complete_set_hit'] / evaluated_pairs:.4f} ({t['complete_set_hit']}/{evaluated_pairs})")

    print("\nRecall uses the UNION of annotated sentences per claim-document pair.")
    print("Complete evidence set means at least one entire annotated evidence group was selected.")
    print("These are gold document pairs, not end-to-end document retrieval results.")
    print(f"Saved per-pair analysis: {output_path}")
    return {"claims": len(claims), "evaluated_pairs": evaluated_pairs, "skipped": skipped, "totals": totals}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=BASE_DIR / "data" / "corpus.jsonl")
    parser.add_argument("--claims", type=Path, default=BASE_DIR / "data" / "claims_dev.jsonl")
    parser.add_argument("--output", type=Path, default=BASE_DIR / "sentence_selection_analysis.csv")
    args = parser.parse_args()
    evaluate_sentence_selector(args.corpus, args.claims, args.output)


if __name__ == "__main__":
    main()
