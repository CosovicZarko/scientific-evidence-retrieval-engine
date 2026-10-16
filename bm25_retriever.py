"""A simple BM25 retrieval baseline for scientific evidence in SciFact."""

import json
import re
import sys

from rank_bm25 import BM25Okapi

from retrieval_metrics import recall_at_k


CLAIM = (
    "A strong bias in the phage genome locations where the spacers were "
    "derived has been observed in many CRISPR subtypes that confer the "
    "immunity to phage."
)
EVIDENCE_IDS = {26996935}


def tokenize(text: str) -> list[str]:
    """Split text into lowercase word tokens."""
    return re.findall(r"\b\w+\b", text.lower())


def load_corpus(corpus_path: str) -> list[dict]:
    """Load the SciFact corpus from JSONL, one document per line."""
    documents = []
    with open(corpus_path, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                documents.append(json.loads(line))
    return documents


def retrieve_bm25(
    query: str,
    documents: list[dict],
    k: int = 5,
) -> list[int]:
    """Return the top-k SciFact document IDs, ranked by BM25 score."""
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer")
    if not documents:
        return []

    tokenized_documents = [
        tokenize(doc["title"] + " " + " ".join(doc["abstract"]))
        for doc in documents
    ]
    bm25 = BM25Okapi(tokenized_documents)
    scores = bm25.get_scores(tokenize(query))

    # Highest score first; preserve corpus order for tied scores.
    ranked_indices = sorted(range(len(documents)), key=lambda i: -scores[i])[:k]
    return [int(documents[i]["doc_id"]) for i in ranked_indices]


if __name__ == "__main__":
    corpus_path = sys.argv[1] if len(sys.argv) > 1 else "data/corpus.jsonl"
    documents = load_corpus(corpus_path)
    retrieved_ids = retrieve_bm25(CLAIM, documents, k=5)

    print(f"Documents in corpus: {len(documents)}")
    print(f"Top 5 BM25 document IDs: {retrieved_ids}")
    print(f"Annotated evidence document IDs: {sorted(EVIDENCE_IDS)}")
    print(f"Recall@5: {recall_at_k(retrieved_ids, EVIDENCE_IDS, 5):.4f}")
