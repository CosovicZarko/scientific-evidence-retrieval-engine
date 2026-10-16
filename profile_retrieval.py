"""Profile local SciFact retrieval (no LLM, no stance inference, no gold labels).

Run twice with the same arguments: first run builds semantic embeddings, second
loads the persistent cache. Writes an optional JSON timing report.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

from retrieval_embedding_cache import InvalidEmbeddingCache

BASE_DIR = Path(__file__).resolve().parent


class TimedRetriever:
    def __init__(self, wrapped, timings: dict):
        self.wrapped, self.timings = wrapped, timings

    def retrieve(self, query, k=5):
        start = perf_counter()
        result = self.wrapped.retrieve(query, k=k)
        self.timings["hybrid_query_seconds"] = round(perf_counter() - start, 6)
        return result


class TimedCrossEncoder:
    def __init__(self, wrapped, timings: dict, name: str):
        self.wrapped, self.timings, self.name = wrapped, timings, name

    def predict(self, *args, **kwargs):
        start = perf_counter()
        result = self.wrapped.predict(*args, **kwargs)
        self.timings[self.name] = round(perf_counter() - start, 6)
        return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path,
                        default=BASE_DIR / "data" / "corpus.jsonl")
    parser.add_argument("--checkpoint", type=Path,
                        default=BASE_DIR / "checkpoints" / "scifact_reranker")
    parser.add_argument("--cache-dir", type=Path,
                        default=BASE_DIR / ".cache" / "semantic_index")
    parser.add_argument("--encoder-revision", default="default",
                        help="Change when updating model weights/revision")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--claim", default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    # Import heavy NLP libraries only after parsing arguments.
    from sentence_transformers import CrossEncoder, SentenceTransformer
    from bm25_retriever import load_corpus
    from cached_hybrid_retriever import CachedHybridRetriever
    from evidence_pipeline import (DEMO_CLAIM, SENTENCE_MODEL_NAME,
                                   run_evidence_pipeline)
    from semantic_retriever import MODEL_NAME

    if not args.corpus.is_file():
        parser.error(f"Corpus not found: {args.corpus}")
    if not (args.checkpoint / "config.json").is_file():
        parser.error(f"Fine-tuned reranker checkpoint missing: {args.checkpoint}")
    timings = {}
    start_all = perf_counter()

    started = perf_counter()
    documents = load_corpus(str(args.corpus))
    lookup = {int(doc["doc_id"]): doc for doc in documents}
    if len(lookup) != len(documents):
        raise ValueError("Duplicate corpus document IDs")
    timings["corpus_load_seconds"] = round(perf_counter() - started, 6)

    started = perf_counter()
    encoder = SentenceTransformer(MODEL_NAME)
    timings["encoder_load_seconds"] = round(perf_counter() - started, 6)

    started = perf_counter()
    hybrid = CachedHybridRetriever(
        documents, encoder, cache_dir=args.cache_dir,
        encoder_key=MODEL_NAME, encoder_revision=args.encoder_revision,
        candidate_k=100, rrf_constant=60, rebuild_cache=args.rebuild_cache,
    )
    timings["hybrid_index_total_seconds"] = round(perf_counter() - started, 6)
    cache = hybrid.cache_result

    started = perf_counter()
    document_reranker = CrossEncoder(str(args.checkpoint.resolve()))
    timings["document_reranker_load_seconds"] = round(perf_counter() - started, 6)

    started = perf_counter()
    sentence_reranker = CrossEncoder(SENTENCE_MODEL_NAME)
    timings["sentence_reranker_load_seconds"] = round(perf_counter() - started, 6)

    claim = args.claim or DEMO_CLAIM
    started = perf_counter()
    papers = run_evidence_pipeline(
        claim, lookup, TimedRetriever(hybrid, timings),
        TimedCrossEncoder(document_reranker, timings, "document_rerank_seconds"),
        TimedCrossEncoder(sentence_reranker, timings, "sentence_selection_seconds"),
    )
    timings["query_total_seconds"] = round(perf_counter() - started, 6)
    timings["total_process_seconds"] = round(perf_counter() - start_all, 6)
    summary = {
        "pipeline": "SciFact hybrid top20 -> fine-tuned rerank top3 -> selected sentences",
        "encoder": MODEL_NAME,
        "corpus_documents": len(documents),
        "cache_hit": cache.hit,
        "cache_key": cache.cache_key,
        "cache_file": str(cache.embeddings_path),
        "embedding_shape": list(cache.embeddings.shape),
        "cache_load_seconds": round(cache.load_seconds, 6),
        "embedding_build_seconds": round(cache.build_seconds, 6),
        "timings": timings,
        "ranked_papers": [
            {"rank": paper["rank"], "doc_id": paper["doc_id"],
             "sentence_indices": [s["index"] for s in paper["sentences"]]}
            for paper in papers
        ],
    }
    print(json.dumps(summary, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.exists():
            parser.error(f"Will not overwrite an existing report: {args.output}")
        args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"Saved profile: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
