"""Prepare SciFact SUPPORT/CONTRADICT sentence-level classification data.

Usage:
    python prepare_evidence_classification_data.py
    python prepare_evidence_classification_data.py --val-fraction 0.2 --seed 42

Only SciFact TRAINING claims are used. The train/validation split is made by
claim (and grouped by identical claim text) BEFORE extracting annotations.
No unannotated document is assigned a label; there is no artificial
INSUFFICIENT_EVIDENCE class.

The output contains one record per distinct *annotated evidence sentence set*.
It is not one example per sentence or one example per claim-document pair.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random


BASE_DIR = Path(__file__).resolve().parent
LABEL_TO_ID = {"CONTRADICT": 0, "SUPPORT": 1}


def load_jsonl(path: Path) -> list[dict]:
    """Read SciFact JSONL with the standard library, one object per line."""
    records = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_number}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Expected a JSON object at {path}:{line_number}")
            records.append(record)
    return records


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_claim_text(text: str) -> str:
    """Keep verbatim-duplicate claims together even if their IDs differ."""
    return " ".join(text.lower().split())


def split_claims(
    claims: list[dict], val_fraction: float, seed: int
) -> tuple[list[dict], list[dict], int]:
    """Split annotated claims before creating examples; preserve text groups."""
    if not 0 < val_fraction < 1:
        raise ValueError("--val-fraction must be strictly between 0 and 1")

    claim_ids = [int(c["id"]) for c in claims]
    if len(set(claim_ids)) != len(claim_ids):
        raise ValueError("Duplicate claim IDs in training claims")

    annotated = [c for c in claims if c.get("evidence")]
    skipped = len(claims) - len(annotated)
    groups_by_text = defaultdict(list)
    for claim in annotated:
        groups_by_text[normalized_claim_text(claim["claim"])].append(claim)

    groups = list(groups_by_text.values())
    if len(groups) < 2:
        raise ValueError("Need at least two distinct annotated claims/groups")
    rng = random.Random(seed)
    rng.shuffle(groups)
    target_validation = min(
        len(annotated) - 1,
        max(1, round(len(annotated) * val_fraction)),
    )
    validation, training = [], []
    for group_index, group in enumerate(groups):
        # Keep at least one group for training.
        if len(validation) < target_validation and group_index < len(groups) - 1:
            validation.extend(group)
        else:
            training.extend(group)

    assert training and validation
    train_ids = {int(c["id"]) for c in training}
    val_ids = {int(c["id"]) for c in validation}
    assert train_ids.isdisjoint(val_ids)
    assert not {normalized_claim_text(c["claim"]) for c in training} & {
        normalized_claim_text(c["claim"]) for c in validation
    }
    return training, validation, skipped


def extract_examples(claims: list[dict], documents_by_id: dict[int, dict]) -> list[dict]:
    """Turn annotated sentence-index groups into binary stance examples.

    Reject bad annotations instead of silently producing misleading labels.
    Repeat annotations with the *same* claim/doc/sentence-set/label are removed.
    Conflicting labels for a claim/document are rejected as ambiguous.
    """
    examples = []
    seen = set()
    for claim in claims:
        claim_id = int(claim["id"])
        claim_text = claim["claim"]
        for doc_id_raw, annotations in (claim.get("evidence") or {}).items():
            doc_id = int(doc_id_raw)
            doc = documents_by_id.get(doc_id)
            if doc is None:
                raise ValueError(f"Claim {claim_id}: evidence document {doc_id} is missing")
            if not isinstance(annotations, list) or not annotations:
                raise ValueError(f"Claim {claim_id}, document {doc_id}: empty/invalid annotations")
            abstract = doc["abstract"]
            if not isinstance(abstract, list):
                raise ValueError(f"Document {doc_id}: abstract must be a sentence list")

            doc_labels = {entry.get("label") for entry in annotations}
            unknown = doc_labels - LABEL_TO_ID.keys()
            if unknown:
                raise ValueError(f"Claim {claim_id}, document {doc_id}: unknown labels {unknown}")
            if len(doc_labels) != 1:
                raise ValueError(
                    f"Claim {claim_id}, document {doc_id}: conflicting evidence labels {doc_labels}"
                )

            for annotation in annotations:
                indices = annotation.get("sentences")
                if not isinstance(indices, list) or not indices:
                    raise ValueError(
                        f"Claim {claim_id}, document {doc_id}: missing/empty sentence indices"
                    )
                if len(indices) != len(set(indices)):
                    raise ValueError(
                        f"Claim {claim_id}, document {doc_id}: repeated sentence indices {indices}"
                    )
                for index in indices:
                    if type(index) is not int or not 0 <= index < len(abstract):
                        raise ValueError(
                            f"Claim {claim_id}, document {doc_id}: invalid sentence index "
                            f"{index!r}; abstract has {len(abstract)} sentences"
                        )

                label_name = annotation["label"]
                # Canonicalize index ordering for consistent de-duplication/text.
                sentence_indices = sorted(indices)
                key = (claim_id, doc_id, tuple(sentence_indices), label_name)
                if key in seen:
                    continue
                seen.add(key)
                evidence_sentences = [abstract[index] for index in sentence_indices]
                examples.append({
                    "claim_id": claim_id,
                    "doc_id": doc_id,
                    "claim": claim_text,
                    "document_title": doc["title"],
                    "sentence_indices": sentence_indices,
                    "evidence_sentences": evidence_sentences,
                    "evidence": " ".join(evidence_sentences),
                    "label": LABEL_TO_ID[label_name],
                    "label_name": label_name,
                    "label_source": "scifact_annotated_evidence",
                })
    return examples


def write_jsonl(path: Path, examples: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for example in examples:
            file.write(json.dumps(example, ensure_ascii=False) + "\n")


def summarize(examples: list[dict]) -> dict:
    labels = Counter(item["label_name"] for item in examples)
    return {
        "claims": len({e["claim_id"] for e in examples}),
        "claim_document_pairs": len({(e["claim_id"], e["doc_id"]) for e in examples}),
        "evidence_groups": len(examples),
        "SUPPORT": labels["SUPPORT"],
        "CONTRADICT": labels["CONTRADICT"],
    }


def prepare(corpus_path: Path, claims_path: Path, output_dir: Path, *,
            val_fraction: float = 0.2, seed: int = 42) -> dict:
    documents = load_jsonl(corpus_path)
    claims = load_jsonl(claims_path)
    documents_by_id = {int(doc["doc_id"]): doc for doc in documents}
    if len(documents_by_id) != len(documents):
        raise ValueError("Duplicate document IDs in corpus")
    if not documents:
        raise ValueError("SciFact corpus is empty")

    training_claims, validation_claims, skipped = split_claims(
        claims, val_fraction, seed
    )
    train_examples = extract_examples(training_claims, documents_by_id)
    val_examples = extract_examples(validation_claims, documents_by_id)

    if not train_examples or not val_examples:
        raise ValueError("No evidence examples available in training or validation")
    if set(e["claim_id"] for e in train_examples) & set(e["claim_id"] for e in val_examples):
        raise ValueError("Claim leakage between training and validation")
    # Text duplicates must also be confined to one split.
    if {normalized_claim_text(e["claim"]) for e in train_examples} & {
        normalized_claim_text(e["claim"]) for e in val_examples
    }:
        raise ValueError("Duplicate claim text crosses the training/validation split")

    random.Random(seed).shuffle(train_examples)
    random.Random(seed + 1).shuffle(val_examples)
    manifest = {
        "purpose": "SciFact binary evidence stance classification",
        "data_source": "SciFact claims_train.jsonl only; dev/test untouched",
        "unit_of_example": "claim + one annotated evidence sentence group",
        "labels": LABEL_TO_ID,
        "no_insufficient_evidence_class": True,
        "corpus_sha256": sha256_file(corpus_path),
        "claims_train_sha256": sha256_file(claims_path),
        "seed": seed,
        "validation_fraction": val_fraction,
        "scientific_abstracts": len(documents),
        "training_claims_total": len(claims),
        "annotated_claims": len(training_claims) + len(validation_claims),
        "claims_skipped_without_evidence": skipped,
        "train": summarize(train_examples),
        "validation": summarize(val_examples),
    }

    train_path = output_dir / "evidence_classification_train.jsonl"
    val_path = output_dir / "evidence_classification_val.jsonl"
    manifest_path = output_dir / "evidence_classification_manifest.json"
    write_jsonl(train_path, train_examples)
    write_jsonl(val_path, val_examples)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"Scientific abstracts:          {len(documents)}")
    print(f"SciFact training claims:       {len(claims)}")
    print(f"Claims with evidence:          {manifest['annotated_claims']}")
    print(f"Claims skipped (no evidence):  {skipped}")
    for split in ("train", "validation"):
        counts = manifest[split]
        print(f"{split.capitalize():<11} {counts['claims']:>3} claims, "
              f"{counts['evidence_groups']:>3} evidence groups "
              f"(SUPPORT={counts['SUPPORT']}, CONTRADICT={counts['CONTRADICT']})")
    print("Labels: SUPPORT=1, CONTRADICT=0; no synthetic insufficient-evidence labels")
    print("Claim-level split; identical claim text never crosses splits")
    for file in (train_path, val_path, manifest_path):
        print(f"Saved: {file}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=BASE_DIR / "data" / "corpus.jsonl")
    parser.add_argument("--claims", type=Path, default=BASE_DIR / "data" / "claims_train.jsonl")
    parser.add_argument("--output-dir", type=Path, default=BASE_DIR / "data" / "derived")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    prepare(args.corpus, args.claims, args.output_dir,
            val_fraction=args.val_fraction, seed=args.seed)


if __name__ == "__main__":
    main()
