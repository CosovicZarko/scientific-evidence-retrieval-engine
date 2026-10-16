"""Retrieve SciFact papers and display candidate evidence sentences.

Pipeline (inference only):
    hybrid BM25 + semantic RRF top 20
    -> SciFact-fine-tuned document reranker top 3
    -> pretrained cross-encoder sentence selector top 2 per abstract

This script deliberately does NOT read gold evidence annotations, run stance
classification, or declare whether a scientific claim is true. The selected
sentences are *candidates*, not verified scientific evidence.

Run:
    python evidence_pipeline.py
    python evidence_pipeline.py --claim "Vitamin D reduces respiratory infections."

Requires the existing project files, SciFact data/corpus.jsonl, and the local
fine-tuned checkpoint checkpoints/scifact_reranker/.
"""

import argparse
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent
DEMO_CLAIM = (
    "A strong bias in the phage genome locations where the spacers were "
    "derived has been observed in many CRISPR subtypes that confer the "
    "immunity to phage."
)
SENTENCE_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L6-v2"


def select_sentences_for_documents(
    claim: str,
    ranked_doc_ids: list[int],
    documents_by_id: dict[int, dict],
    sentence_reranker,
    k: int = 2,
    batch_size: int = 16,
) -> list[dict]:
    """Rank original abstract sentences per document using one batched call.

    Returns one record per document with original zero-based sentence indices.
    No gold annotations are consulted.
    """
    if type(k) is not int or k <= 0:
        raise ValueError("k must be a positive integer")
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if len(ranked_doc_ids) != len(set(ranked_doc_ids)):
        raise ValueError("Document IDs must be unique")

    pairs: list[tuple[str, str]] = []
    entries = []
    for doc_id in ranked_doc_ids:
        document = documents_by_id[doc_id]
        abstract = document["abstract"]
        if not isinstance(abstract, list) or not abstract:
            raise ValueError(f"Missing or empty abstract for document {doc_id}")
        if any(not isinstance(sentence, str) or not sentence.strip() for sentence in abstract):
            raise ValueError(f"Blank or invalid sentence in document {doc_id}")
        entries.append((doc_id, document, len(pairs)))
        pairs.extend((claim, sentence) for sentence in abstract)

    if not pairs:
        return []

    raw_scores = np.asarray(sentence_reranker.predict(
        pairs,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=False,
    ))
    # CrossEncoder normally returns (N,) or (N, 1) scores.
    if raw_scores.size != len(pairs) or not np.all(np.isfinite(raw_scores)):
        raise ValueError("Sentence reranker must return one finite score per sentence")
    scores = raw_scores.reshape(-1)

    output = []
    for position, (doc_id, document, offset) in enumerate(entries):
        abstract = document["abstract"]
        indices = sorted(
            range(len(abstract)),
            key=lambda i: (-float(scores[offset + i]), i),
        )[:k]
        output.append({
            "rank": position + 1,
            "doc_id": doc_id,
            "title": document["title"],
            "sentences": [
                {"index": i, "text": abstract[i]}
                for i in indices
            ],
        })
    return output


def run_evidence_pipeline(
    claim: str,
    documents_by_id: dict[int, dict],
    hybrid_retriever,
    document_reranker,
    sentence_reranker,
    batch_size: int = 16,
) -> list[dict]:
    """Return three reranked papers with two candidate sentences per paper."""
    if not isinstance(claim, str) or not claim.strip():
        raise ValueError("claim must be non-empty text")

    # Import the actual project reranker; it scores claim + title + abstract.
    from rerank_evidence import rerank_documents

    candidate_ids = hybrid_retriever.retrieve(claim, k=20)
    if not candidate_ids:
        return []
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("Hybrid retrieval returned duplicate document IDs")

    ranked_ids = rerank_documents(
        claim, candidate_ids, documents_by_id, document_reranker,
        k=len(candidate_ids), batch_size=batch_size,
    )
    if len(ranked_ids) != len(candidate_ids) or set(ranked_ids) != set(candidate_ids):
        raise AssertionError("Document reranking changed the candidate set")

    return select_sentences_for_documents(
        claim,
        ranked_ids[:3],
        documents_by_id,
        sentence_reranker,
        k=2,
        batch_size=batch_size,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--claim", default=DEMO_CLAIM)
    parser.add_argument("--corpus", type=Path, default=BASE_DIR / "data/corpus.jsonl")
    parser.add_argument(
        "--checkpoint", type=Path,
        default=BASE_DIR / "checkpoints/scifact_reranker",
        help="Path to the SciFact fine-tuned document reranker",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    if not (args.checkpoint / "config.json").is_file() or not (
        args.checkpoint / "model.safetensors"
    ).is_file():
        parser.error(
            f"Missing fine-tuned document reranker checkpoint at {args.checkpoint}. "
            "Run fine_tune_reranker.py first or use --checkpoint."
        )

    # Lazy imports make the analysis functions testable without loading models.
    from sentence_transformers import CrossEncoder, SentenceTransformer
    from bm25_retriever import load_corpus
    from hybrid_retriever import HybridRetriever
    from semantic_retriever import MODEL_NAME as ENCODER_MODEL_NAME

    documents = load_corpus(str(args.corpus))
    documents_by_id = {int(doc["doc_id"]): doc for doc in documents}
    if len(documents_by_id) != len(documents):
        raise ValueError("Duplicate document IDs in corpus")

    print(f"Scientific abstracts: {len(documents)}")
    print("Loading hybrid retriever (BM25 + semantic embeddings)...")
    hybrid_retriever = HybridRetriever(
        documents,
        SentenceTransformer(ENCODER_MODEL_NAME),
        candidate_k=100,
        rrf_constant=60,
    )
    print(f"Loading fine-tuned document reranker: {args.checkpoint}")
    document_reranker = CrossEncoder(str(args.checkpoint.resolve()))
    print(f"Loading neural sentence selector: {SENTENCE_MODEL_NAME}")
    sentence_reranker = CrossEncoder(SENTENCE_MODEL_NAME)

    results = run_evidence_pipeline(
        args.claim,
        documents_by_id,
        hybrid_retriever,
        document_reranker,
        sentence_reranker,
        batch_size=args.batch_size,
    )
    print(f"\nClaim: {args.claim}")
    print("Pipeline: hybrid top 20 -> document reranked top 3 -> two sentences each")
    for result in results:
        print(f"\n--- Paper {result['rank']} ---")
        print(f"Document ID: {result['doc_id']}")
        print(f"Title: {result['title']}")
        for sentence in result["sentences"]:
            print(f"  [{sentence['index']}] {sentence['text']}")
    if not results:
        print("No candidate documents were returned.")
    print("\nNote: These are automatically selected candidate sentences, not verified "
          "evidence or scientific verdicts.")


if __name__ == "__main__":
    main()
