"""Simple TF-IDF evidence sentence selection baseline for SciFact.

Run:
    python evidence_sentence_selector.py

This scores sentences within one abstract. It does not predict SUPPORT vs.
CONTRADICT and does not use annotated evidence indices to pick the sentences.
"""

from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from prepare_evidence_classification_data import load_jsonl


BASE_DIR = Path(__file__).resolve().parent
EXAMPLE_CLAIM_ID = 1248
EXAMPLE_DOC_ID = 7209559


def select_evidence_sentences(
    claim: str,
    abstract: list[str],
    k: int = 2,
) -> list[int]:
    """Return zero-based sentence indices ranked by claim–sentence TF-IDF similarity.

    Fit a separate TF-IDF vectorizer on [claim, *abstract] for this one
    claim/document pair. Highest cosine similarity is ranked first; ties
    preserve original sentence order. No gold annotations are used.
    """
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer")
    if not abstract:
        return []

    vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
    texts = [claim, *abstract]

    # A vectorizer cannot build a vocabulary from wholly non-word text.
    if not any(vectorizer.build_tokenizer()(text.lower()) for text in texts):
        return list(range(min(k, len(abstract))))

    matrix = vectorizer.fit_transform(texts)
    similarities = cosine_similarity(matrix[0:1], matrix[1:]).ravel()

    indices = sorted(
        range(len(abstract)),
        key=lambda i: (-float(similarities[i]), i),
    )
    return indices[:k]


def main() -> None:
    corpus = load_jsonl(BASE_DIR / "data" / "corpus.jsonl")
    training_claims = load_jsonl(BASE_DIR / "data" / "claims_train.jsonl")

    claim_record = next(
        (c for c in training_claims if c["id"] == EXAMPLE_CLAIM_ID), None
    )
    document = next(
        (d for d in corpus if int(d["doc_id"]) == EXAMPLE_DOC_ID), None
    )
    if claim_record is None or document is None:
        raise ValueError("SciFact example claim or document was not found")

    claim = claim_record["claim"]
    abstract = document["abstract"]
    selected = select_evidence_sentences(claim, abstract, k=2)
    gold_indices = sorted({
        sentence_index
        for annotation in claim_record["evidence"][str(EXAMPLE_DOC_ID)]
        for sentence_index in annotation["sentences"]
    })

    print(f"Claim ID: {EXAMPLE_CLAIM_ID}")
    print(f"Claim: {claim}")
    print(f"Document ID: {EXAMPLE_DOC_ID}")
    print(f"Document title: {document['title']}")
    print(f"Annotated evidence indices: {gold_indices}")
    print(f"Selected top 2 indices: {selected}")
    for index in selected:
        print(f"  [{index}] {abstract[index].strip()}")
    print(f"Gold evidence found in top 2: {bool(set(selected) & set(gold_indices))}")


if __name__ == "__main__":
    main()
