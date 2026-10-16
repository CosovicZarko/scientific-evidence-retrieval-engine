"""Semantic retrieval baseline for SciFact claim 47 using a pretrained encoder.

Install: pip install sentence-transformers numpy rank-bm25
Run:     python semantic_retriever.py
Or:      python semantic_retriever.py /path/to/corpus.jsonl
"""

import sys
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from bm25_retriever import CLAIM, EVIDENCE_IDS, load_corpus
from retrieval_metrics import recall_at_k

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def build_embeddings(
    documents: list[dict],
    model: SentenceTransformer,
) -> np.ndarray:
    """Encode each SciFact title and abstract into a normalized embedding."""
    if not documents:
        raise ValueError("Cannot build embeddings from an empty corpus")

    texts = [
        document["title"] + " " + " ".join(document["abstract"])
        for document in documents
    ]

    return model.encode(
        texts,
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )


def retrieve_semantic(
    query: str,
    documents: list[dict],
    embeddings: np.ndarray,
    model: SentenceTransformer,
    k: int = 5,
) -> list[int]:
    """Return the top-k document IDs by cosine similarity, best first."""
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer")
    if embeddings.ndim != 2 or embeddings.shape[0] != len(documents):
        raise ValueError("Embeddings must have one row per document")
    if not documents:
        return []

    query_embedding = model.encode(
        query,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    if query_embedding.ndim != 1 or query_embedding.shape[0] != embeddings.shape[1]:
        raise ValueError("Query and document embedding dimensions must match")

    # Normalized embeddings: dot product equals cosine similarity.
    similarities = embeddings @ query_embedding
    ranked_indices = np.argsort(-similarities, kind="stable")[:k]
    return [int(documents[i]["doc_id"]) for i in ranked_indices]


def main() -> None:
    corpus_path = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path(__file__).resolve().parent / "data" / "corpus.jsonl"
    )

    documents = load_corpus(str(corpus_path))
    print(f"Documents in corpus: {len(documents)}")
    print(f"Loading pretrained model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    embeddings = build_embeddings(documents, model)
    print(f"Document embeddings shape: {embeddings.shape}")

    retrieved_ids = retrieve_semantic(CLAIM, documents, embeddings, model, k=5)
    print(f"Top 5 semantic document IDs: {retrieved_ids}")
    print(f"Annotated evidence document IDs: {sorted(EVIDENCE_IDS)}")
    print(f"Recall@5: {recall_at_k(retrieved_ids, EVIDENCE_IDS, 5):.4f}")


if __name__ == "__main__":
    main()
