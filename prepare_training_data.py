"""Prepare SciFact claim-document pairs for *future* reranker fine-tuning.

Usage:
    python prepare_training_data.py
    python prepare_training_data.py --val-fraction 0.2 --hard-negatives 2 --random-negatives 2

Requires: pip install rank-bm25 numpy

Only data/claims_train.jsonl is used for supervised pairs and splitting.
Data/dev and data/test are deliberately NOT read.

IMPORTANT: Zero labels are UNVERIFIED presumed negatives: a missing SciFact
annotation does not establish irrelevance. Both SUPPORT and CONTRADICT
annotations become positive relevance labels (1).
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

from rank_bm25 import BM25Okapi

from bm25_retriever import load_corpus, tokenize
from evaluate_bm25 import load_claims


BASE_DIR = Path(__file__).resolve().parent


def document_text(doc: dict) -> str:
    """Match the title + abstract representation used by earlier baselines."""
    return doc["title"] + " " + " ".join(doc["abstract"])


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def split_annotated_claims(
    claims: list[dict], val_fraction: float, seed: int
) -> tuple[list[dict], list[dict], int]:
    """Split on claim IDs BEFORE creating positives/negatives."""
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1")

    annotated = [claim for claim in claims if claim.get("evidence")]
    skipped = len(claims) - len(annotated)
    if len(annotated) < 2:
        raise ValueError("Need at least two annotated claims for train/validation")

    ids = [int(claim["id"]) for claim in claims]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate claim IDs in the training file")

    rng = random.Random(seed)
    rng.shuffle(annotated)
    n_val = min(len(annotated) - 1, max(1, round(len(annotated) * val_fraction)))
    validation = annotated[:n_val]
    training = annotated[n_val:]
    assert not {c["id"] for c in training} & {c["id"] for c in validation}
    return training, validation, skipped


def make_pairs(
    claims: list[dict],
    documents: list[dict],
    bm25: BM25Okapi,
    *,
    hard_negatives: int,
    random_negatives: int,
    hard_pool: int,
    hard_skip: int,
    seed: int,
) -> list[dict]:
    """Produce positive evidence pairs and cautiously sampled *presumed* negatives.

    The BM25 index is built once by the caller. We exclude cited documents
    as well as all annotated positive evidence when mining negatives.
    """
    doc_by_id = {int(doc["doc_id"]): doc for doc in documents}
    doc_ids = [int(doc["doc_id"]) for doc in documents]
    if len(doc_by_id) != len(documents):
        raise ValueError("Duplicate document IDs in corpus")

    rng = random.Random(seed)
    pairs = []
    for claim in claims:
        claim_id = int(claim["id"])
        claim_text = claim["claim"]
        annotations = claim.get("evidence") or {}
        positive_ids = {int(doc_id) for doc_id in annotations}
        cited_ids = {int(doc_id) for doc_id in claim.get("cited_doc_ids", [])}
        excluded_ids = positive_ids | cited_ids

        missing = positive_ids - doc_by_id.keys()
        if missing:
            raise ValueError(f"Claim {claim_id} has unknown evidence IDs: {sorted(missing)}")

        for doc_id in sorted(positive_ids):
            labels = sorted({entry["label"] for entry in annotations[str(doc_id)]})
            pairs.append({
                "claim_id": claim_id,
                "doc_id": doc_id,
                "claim": claim_text,
                "document": document_text(doc_by_id[doc_id]),
                "label": 1,
                "label_source": "annotated_evidence",
                "evidence_labels": labels,
            })

        # Hard negatives: high-ranked BM25 results without known annotations
        # or citations. These are NOT human-verified negative labels.
        scores = bm25.get_scores(tokenize(claim_text))
        top_indices = sorted(
            range(len(doc_ids)), key=lambda i: -float(scores[i])
        )[:hard_pool]
        hard_candidates = [
            doc_ids[i] for rank, i in enumerate(top_indices, start=1)
            if rank > hard_skip and doc_ids[i] not in excluded_ids
        ]
        if len(hard_candidates) < hard_negatives:
            raise ValueError(
                f"Not enough hard-negative candidates for claim {claim_id}; "
                "increase --hard-pool or reduce --hard-negatives"
            )
        selected_hard = hard_candidates[:hard_negatives]
        excluded_ids.update(selected_hard)

        # Random negatives come from the full corpus, excluding known
        # positives, citations, and already-selected hard negatives.
        random_candidates = [doc_id for doc_id in doc_ids if doc_id not in excluded_ids]
        if len(random_candidates) < random_negatives:
            raise ValueError(f"Not enough random negatives for claim {claim_id}")
        selected_random = rng.sample(random_candidates, random_negatives)

        for source, selected_ids in (
            ("presumed_negative_bm25", selected_hard),
            ("presumed_negative_random", selected_random),
        ):
            for doc_id in selected_ids:
                pairs.append({
                    "claim_id": claim_id,
                    "doc_id": doc_id,
                    "claim": claim_text,
                    "document": document_text(doc_by_id[doc_id]),
                    "label": 0,
                    "label_source": source,
                    "evidence_labels": [],
                })

    rng.shuffle(pairs)
    pair_keys = [(pair["claim_id"], pair["doc_id"]) for pair in pairs]
    if len(set(pair_keys)) != len(pair_keys):
        raise ValueError("Duplicate claim-document pair was generated")
    return pairs


def write_jsonl(path: Path, pairs: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for pair in pairs:
            file.write(json.dumps(pair, ensure_ascii=False) + "\n")


def summarize(pairs: list[dict]) -> dict:
    return {
        "claims": len({p["claim_id"] for p in pairs}),
        "pairs": len(pairs),
        "positives": sum(p["label"] == 1 for p in pairs),
        "presumed_negatives": sum(p["label"] == 0 for p in pairs),
        "by_source": dict(sorted(Counter(p["label_source"] for p in pairs).items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=BASE_DIR / "data/corpus.jsonl")
    parser.add_argument("--claims", type=Path, default=BASE_DIR / "data/claims_train.jsonl")
    parser.add_argument("--output-dir", type=Path, default=BASE_DIR / "data/derived")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--hard-negatives", type=int, default=2)
    parser.add_argument("--random-negatives", type=int, default=2)
    parser.add_argument("--hard-pool", type=int, default=100)
    parser.add_argument("--hard-skip", type=int, default=5)
    args = parser.parse_args()

    if args.hard_negatives < 0 or args.random_negatives < 0:
        parser.error("Negative sample counts must be nonnegative")
    if args.hard_pool <= 0 or args.hard_skip < 0 or args.hard_skip >= args.hard_pool:
        parser.error("Require 0 <= hard-skip < hard-pool")
    if args.hard_negatives + args.random_negatives == 0:
        parser.error("Specify at least one negative per annotated claim")

    documents = load_corpus(str(args.corpus))
    claims = load_claims(str(args.claims))
    if not documents:
        raise ValueError("Empty corpus")

    training, validation, skipped = split_annotated_claims(
        claims, args.val_fraction, args.seed
    )

    # Unsupervised index: corpus-only. No development/test claims or labels.
    bm25 = BM25Okapi([tokenize(document_text(doc)) for doc in documents])

    training_pairs = make_pairs(
        training, documents, bm25,
        hard_negatives=args.hard_negatives,
        random_negatives=args.random_negatives,
        hard_pool=args.hard_pool, hard_skip=args.hard_skip, seed=args.seed,
    )
    validation_pairs = make_pairs(
        validation, documents, bm25,
        hard_negatives=args.hard_negatives,
        random_negatives=args.random_negatives,
        hard_pool=args.hard_pool, hard_skip=args.hard_skip, seed=args.seed + 1,
    )

    assert not ({p["claim_id"] for p in training_pairs} &
                {p["claim_id"] for p in validation_pairs})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / "reranker_train.jsonl"
    val_path = args.output_dir / "reranker_val.jsonl"
    write_jsonl(train_path, training_pairs)
    write_jsonl(val_path, validation_pairs)

    manifest = {
        "purpose": "binary claim-document relevance reranker fine-tuning",
        "caution": "label=0 means an unverified presumed negative, not proven irrelevance",
        "positive_labels": ["SUPPORT", "CONTRADICT"],
        "data_source": "SciFact claims_train only; dev/test labels untouched",
        "corpus_sha256": sha256_file(args.corpus),
        "claims_train_sha256": sha256_file(args.claims),
        "seed": args.seed,
        "validation_fraction": args.val_fraction,
        "hard_negatives_per_claim": args.hard_negatives,
        "random_negatives_per_claim": args.random_negatives,
        "hard_pool": args.hard_pool,
        "hard_skip": args.hard_skip,
        "excluded_from_negatives": ["annotated evidence IDs", "all cited_doc_ids"],
        "total_training_claims": len(claims),
        "claims_skipped_without_annotations": skipped,
        "train": summarize(training_pairs),
        "validation": summarize(validation_pairs),
    }
    manifest_path = args.output_dir / "training_data_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"Scientific abstracts: {len(documents)}")
    print(f"SciFact training claims: {len(claims)}")
    print(f"Claims with annotated evidence: {len(training) + len(validation)}")
    print(f"Claims skipped (no evidence): {skipped}")
    print(f"Train: {manifest['train']['claims']} claims, "
          f"{manifest['train']['positives']} positive pairs, "
          f"{manifest['train']['presumed_negatives']} presumed negative pairs")
    print(f"Validation: {manifest['validation']['claims']} claims, "
          f"{manifest['validation']['positives']} positive pairs, "
          f"{manifest['validation']['presumed_negatives']} presumed negative pairs")
    print("Negative sampling: excluded annotated evidence and ALL cited document IDs")
    print("CAUTION: negatives are assumed, not verified irrelevant")
    print(f"Saved: {train_path}")
    print(f"Saved: {val_path}")
    print(f"Saved: {manifest_path}")


if __name__ == "__main__":
    main()
