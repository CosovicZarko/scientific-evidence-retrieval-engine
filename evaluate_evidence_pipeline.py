"""Evaluate end-to-end SciFact evidence localization on development claims.

Use saved *fine-tuned* reranker rankings rather than recomputing retrieval:
    python evaluate_evidence_pipeline.py

For every development claim with gold evidence:
    cached reranked top 3 documents
      -> pretrained neural sentence selector, top 2 sentences per document
      -> score Document Hit@3 and Complete Evidence Hit@3x2

Gold annotations are consulted ONLY after inference to calculate the metrics.
This evaluates evidence localization, NOT SUPPORT/CONTRADICT classification.
"""

import argparse
import csv
import json
from pathlib import Path

from evidence_pipeline import SENTENCE_MODEL_NAME, select_sentences_for_documents
from evaluate_sentence_selector import evidence_sets_for_pair
from prepare_evidence_classification_data import load_jsonl

BASE_DIR = Path(__file__).resolve().parent
TOP_DOCUMENTS = 3
TOP_SENTENCES = 2


def load_cached_rankings(path: Path) -> dict[int, dict]:
    """Load a complete set of cached *fine-tuned* reranker results by claim ID."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Run 'python compare_rerankers.py' first, "
            "or pass --rankings PATH."
        )

    results = {}
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        required = {"claim_id", "claim", "reranked_top20", "evidence_doc_ids"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Rankings CSV must have columns: {sorted(required)}")

        for row in reader:
            claim_id = int(row["claim_id"])
            if claim_id in results:
                raise ValueError(f"Duplicate cached ranking for claim {claim_id}")
            ids = json.loads(row["reranked_top20"])
            if not isinstance(ids, list) or len(ids) < TOP_DOCUMENTS:
                raise ValueError(f"Claim {claim_id}: expected at least 3 ranked documents")
            if any(type(doc_id) is not int for doc_id in ids):
                raise ValueError(f"Claim {claim_id}: invalid document IDs")
            if len(ids) != len(set(ids)):
                raise ValueError(f"Claim {claim_id}: duplicate documents in ranking")

            cached_gold = json.loads(row["evidence_doc_ids"])
            if not isinstance(cached_gold, list) or any(type(d) is not int for d in cached_gold):
                raise ValueError(f"Claim {claim_id}: invalid cached evidence IDs")

            results[claim_id] = {
                "claim": row["claim"],
                "reranked_top20": ids,
                "cached_evidence_doc_ids": set(cached_gold),
            }

    return results


def build_eval_units(
    corpus_path: Path, claims_path: Path, rankings_path: Path,
) -> tuple[int, int, int, dict[int, dict], list[dict]]:
    """Validate source and cache alignment before any sentence selection."""
    documents = load_jsonl(corpus_path)
    claims = load_jsonl(claims_path)
    documents_by_id = {int(d["doc_id"]): d for d in documents}
    if len(documents_by_id) != len(documents):
        raise ValueError("Duplicate document IDs in corpus")
    if len({int(c["id"]) for c in claims}) != len(claims):
        raise ValueError("Duplicate claim IDs in development set")

    rankings = load_cached_rankings(rankings_path)
    skipped = 0
    units = []
    for claim in claims:
        claim_id = int(claim["id"])
        evidence = claim.get("evidence") or {}
        if not evidence:
            skipped += 1
            continue

        cached = rankings.get(claim_id)
        if cached is None:
            raise ValueError(f"Missing cached reranked results for claim {claim_id}")
        if cached["claim"] != claim["claim"]:
            raise ValueError(f"Claim text differs between development data and cache: {claim_id}")

        groups_by_doc = {}
        for raw_doc_id, annotations in evidence.items():
            doc_id = int(raw_doc_id)
            if doc_id not in documents_by_id:
                raise ValueError(f"Missing gold evidence document {doc_id} for claim {claim_id}")
            abstract = documents_by_id[doc_id]["abstract"]
            if not isinstance(abstract, list) or not abstract:
                raise ValueError(f"Invalid gold abstract for document {doc_id}")
            groups = evidence_sets_for_pair(annotations, len(abstract))
            if not groups:
                raise ValueError(f"Claim {claim_id}, document {doc_id}: empty evidence groups")
            groups_by_doc[doc_id] = groups

        if cached["cached_evidence_doc_ids"] != set(groups_by_doc):
            raise ValueError(f"Cached evidence IDs disagree with development annotations: {claim_id}")

        top3 = cached["reranked_top20"][:TOP_DOCUMENTS]
        for doc_id in top3:
            if doc_id not in documents_by_id:
                raise ValueError(f"Cached retrieved document {doc_id} missing from corpus")

        units.append({
            "claim_id": claim_id,
            "claim": claim["claim"],
            "top3": top3,
            "gold_groups_by_doc": groups_by_doc,
        })

    if not units:
        raise ValueError("No annotated development claims; metrics are undefined")
    expected_ids = {unit["claim_id"] for unit in units}
    if set(rankings) != expected_ids:
        extra = sorted(set(rankings) - expected_ids)
        raise ValueError(f"Unexpected cached claim IDs (possibly wrong split): {extra[:10]}")

    return len(documents), len(claims), skipped, documents_by_id, units


def evaluate(
    corpus_path: Path,
    claims_path: Path,
    rankings_path: Path,
    output_path: Path,
    sentence_reranker,
    batch_size: int = 16,
) -> dict:
    """Evaluate two claim-level binary metrics with gold used only for scoring."""
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")

    n_documents, n_claims, skipped, documents_by_id, units = build_eval_units(
        corpus_path, claims_path, rankings_path
    )

    doc_hits = 0
    complete_hits = 0
    doc_misses = 0
    sentence_misses = 0
    rows = []

    for unit in units:
        # Inference sees only the claim, cached ranked document IDs, and corpus.
        # NO gold indices, labels, or evidence document IDs are passed in.
        selected = select_sentences_for_documents(
            unit["claim"], unit["top3"], documents_by_id,
            sentence_reranker, k=TOP_SENTENCES, batch_size=batch_size,
        )
        if [result["doc_id"] for result in selected] != unit["top3"]:
            raise ValueError(f"Sentence results do not align with claim {unit['claim_id']}")

        selected_indices_by_doc = {}
        for result in selected:
            doc_id = result["doc_id"]
            abstract = documents_by_id[doc_id]["abstract"]
            indices = [item["index"] for item in result["sentences"]]
            if len(indices) != min(TOP_SENTENCES, len(abstract)) or len(indices) != len(set(indices)):
                raise ValueError(f"Invalid selected sentence count/indices for document {doc_id}")
            for item in result["sentences"]:
                if type(item["index"]) is not int or not 0 <= item["index"] < len(abstract):
                    raise ValueError(f"Invalid selected sentence index for document {doc_id}")
                if item["text"] != abstract[item["index"]]:
                    raise ValueError(f"Selected sentence does not match document {doc_id}")
            selected_indices_by_doc[doc_id] = set(indices)

        # GOLD used only here, after all selections are fixed.
        gold_groups = unit["gold_groups_by_doc"]
        doc_hit = any(doc_id in gold_groups for doc_id in unit["top3"])
        complete_hit = any(
            any(group.issubset(selected_indices_by_doc[doc_id]) for group in groups)
            for doc_id, groups in gold_groups.items()
            if doc_id in selected_indices_by_doc
        )
        if complete_hit and not doc_hit:
            raise AssertionError("A complete evidence hit implies a document hit")

        doc_hits += int(doc_hit)
        complete_hits += int(complete_hit)
        doc_misses += int(not doc_hit)
        sentence_misses += int(doc_hit and not complete_hit)
        stage = (
            "complete_evidence_found" if complete_hit else
            "sentence_selection_miss" if doc_hit else
            "document_retrieval_miss"
        )
        rows.append({
            "claim_id": unit["claim_id"],
            "claim": unit["claim"],
            "reranked_top3": json.dumps(unit["top3"]),
            "selected_sentences": json.dumps(selected, ensure_ascii=False),
            "gold_evidence_doc_ids": json.dumps(sorted(gold_groups)),
            "gold_evidence_groups_by_doc": json.dumps({
                str(doc_id): [sorted(group) for group in groups]
                for doc_id, groups in gold_groups.items()
            }),
            "document_hit_at_3": int(doc_hit),
            "complete_evidence_hit_at_3x2": int(complete_hit),
            "outcome": stage,
        })

    evaluated = len(units)
    assert doc_hits + doc_misses == evaluated
    assert complete_hits + sentence_misses + doc_misses == evaluated

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Scientific abstracts:               {n_documents}")
    print(f"Development claims:                 {n_claims}")
    print(f"Claims with annotated evidence:     {evaluated}")
    print(f"Claims skipped:                     {skipped}")
    print("Candidate documents per claim:      3 (cached fine-tuned rankings)")
    print("Selected sentences per document:    2 (neural cross-encoder)")
    print()
    print(f"Document Hit@3:                     {doc_hits / evaluated:.4f} ({doc_hits}/{evaluated})")
    print(f"Complete Evidence Hit@3x2:         {complete_hits / evaluated:.4f} ({complete_hits}/{evaluated})")
    print()
    print(f"Claims missed at document stage:    {doc_misses}")
    print(f"Document found, no complete group:  {sentence_misses}")
    print(f"Claims with complete evidence:      {complete_hits}")
    print()
    print("Gold annotations were used ONLY to score retrieved/selected evidence.")
    print("Missing annotations do not establish irrelevance; no stance predictions made.")
    print(f"Saved per-claim analysis: {output_path}")

    return {
        "evaluated": evaluated,
        "skipped": skipped,
        "document_hits": doc_hits,
        "complete_evidence_hits": complete_hits,
        "document_misses": doc_misses,
        "sentence_misses": sentence_misses,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=BASE_DIR / "data/corpus.jsonl")
    parser.add_argument("--claims", type=Path, default=BASE_DIR / "data/claims_dev.jsonl")
    parser.add_argument(
        "--rankings", type=Path,
        default=BASE_DIR / "docs/reranking_finetuned_analysis.csv",
        help="Cached rankings made with the fine-tuned document reranker",
    )
    parser.add_argument(
        "--output", type=Path,
        default=BASE_DIR / "reports/evidence_pipeline_analysis.csv",
    )
    parser.add_argument("--sentence-model", default=SENTENCE_MODEL_NAME)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    # Check input alignment before downloading/loading the sentence model.
    _, _, _, _, units = build_eval_units(args.corpus, args.claims, args.rankings)
    print(f"Validated cached fine-tuned rankings for {len(units)} annotated claims.")
    from sentence_transformers import CrossEncoder
    print(f"Loading neural sentence selector: {args.sentence_model}")
    selector = CrossEncoder(args.sentence_model)
    evaluate(
        args.corpus, args.claims, args.rankings, args.output,
        selector, args.batch_size,
    )


if __name__ == "__main__":
    main()
