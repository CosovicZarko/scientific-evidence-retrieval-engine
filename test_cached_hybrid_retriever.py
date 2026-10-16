"""Test adapter against simulated original retrieval methods without model downloads."""
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from cached_hybrid_retriever import CachedHybridRetriever
from profile_retrieval import TimedCrossEncoder, TimedRetriever
from test_retrieval_embedding_cache import DOCS, FakeEncoder


class FakeBM25:
    def __init__(self, tokens):
        self.tokens = tokens

    def get_scores(self, query_tokens):
        return np.array([sum(row.count(token) for token in query_tokens)
                         for row in self.tokens])


class ExistingHybridMethods:
    """Simulates the interface of the unmodified HybridRetriever."""
    def retrieve_all(self, query, k=20):
        bm25_scores = self.bm25.get_scores(query.lower().split())
        ranked_bm25 = sorted(range(len(self.doc_ids)),
                             key=lambda i: -bm25_scores[i])[:k]
        vector = self.model.encode(query, convert_to_numpy=True,
                                   normalize_embeddings=True)
        semantic_scores = self.embeddings @ vector
        ranked_semantic = sorted(range(len(self.doc_ids)),
                                 key=lambda i: -semantic_scores[i])[:k]
        # Ranking method reads all original attributes without modification.
        return {"bm25": [self.doc_ids[i] for i in ranked_bm25],
                "semantic": [self.doc_ids[i] for i in ranked_semantic],
                "hybrid": [self.doc_ids[i] for i in ranked_bm25]}

    def retrieve(self, query, k=5):
        return self.retrieve_all(query, k=k)["hybrid"]


class AdapterTests(unittest.TestCase):
    def setUp(self):
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        self.cache = Path(td.name)
        self.model = FakeEncoder()
        self.patched_modules = patch.dict(sys.modules, {
            "hybrid_retriever": types.SimpleNamespace(HybridRetriever=ExistingHybridMethods),
            "bm25_retriever": types.SimpleNamespace(tokenize=lambda s: s.lower().split()),
            "rank_bm25": types.SimpleNamespace(BM25Okapi=FakeBM25),
        })
        self.patched_modules.start()
        self.addCleanup(self.patched_modules.stop)

    def create(self, **kwargs):
        return CachedHybridRetriever(DOCS, self.model, cache_dir=self.cache,
                                     encoder_key="fake", **kwargs)

    def test_methods_from_original_implementation_called(self):
        first = self.create()
        self.assertEqual([10, 11], first.retrieve("paper", k=2))
        self.assertEqual({"bm25", "semantic", "hybrid"},
                         set(first.retrieve_all("paper", k=2)))

    def test_cold_warm_rankings_identical(self):
        cold = self.create()
        cold_ids = cold.retrieve_all("Alpha", k=2)
        warm = self.create()
        warm_ids = warm.retrieve_all("Alpha", k=2)
        self.assertEqual(cold_ids, warm_ids)
        self.assertFalse(cold.cache_result.hit)
        self.assertTrue(warm.cache_result.hit)
        # Two query encodings plus only one DOCUMENT embedding computation.
        self.assertEqual(1, len([call for call in self.model.calls
                                  if isinstance(call[0], list)]))

    def test_candidate_parameters_unchanged(self):
        adapter = self.create(candidate_k=40, rrf_constant=90)
        self.assertEqual((40, 90), (adapter.candidate_k, adapter.rrf_constant))

    def test_invalid_candidate_k(self):
        with self.assertRaises(ValueError):
            self.create(candidate_k=0)

    def test_invalid_rrf_constant(self):
        with self.assertRaises(ValueError):
            self.create(rrf_constant=True)

    def test_timed_retriever_preserves_result(self):
        base = self.create()
        timings = {}
        result = TimedRetriever(base, timings).retrieve("Paper", k=2)
        self.assertEqual(base.retrieve("Paper", k=2), result)
        self.assertIn("hybrid_query_seconds", timings)

    def test_timed_crossencoder_preserves_result(self):
        class FakeCrossEncoder:
            def predict(self, x, **kwargs):
                return [1.0] * len(x)
        timings = {}
        output = TimedCrossEncoder(FakeCrossEncoder(), timings, "document").predict([1, 2])
        self.assertEqual([1.0, 1.0], output)
        self.assertIn("document", timings)


if __name__ == "__main__":
    unittest.main()
