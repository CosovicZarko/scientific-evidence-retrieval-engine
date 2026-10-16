"""Offline tests: no Hugging Face downloads, no SciFact files or Ollama."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from retrieval_embedding_cache import (
    InvalidEmbeddingCache, cache_identity, load_or_build_embeddings,
    source_fingerprint,
)


DOCS = [
    {"doc_id": 10, "title": "Paper one", "abstract": ["Alpha beta.", "Gamma."]},
    {"doc_id": 11, "title": "Paper two", "abstract": ["Delta."]},
]


class FakeEncoder:
    def __init__(self):
        self.calls = []

    def encode(self, texts, *, batch_size=None, convert_to_numpy=None,
               normalize_embeddings=None, show_progress_bar=None):
        self.calls.append((texts, batch_size, convert_to_numpy, normalize_embeddings,
                           show_progress_bar))
        result = []
        for text in ([texts] if isinstance(texts, str) else texts):
            raw = np.array([len(text) + 1, text.lower().count("a") + 1], dtype=np.float32)
            result.append(raw / np.linalg.norm(raw))
        result = np.stack(result)
        return result[0] if isinstance(texts, str) else result


class EmbeddingCacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "cache"
        self.model = FakeEncoder()

    def build(self, docs=None, **kwargs):
        return load_or_build_embeddings(DOCS if docs is None else docs,
             self.model, cache_dir=self.path, encoder_key="test-encoder", **kwargs)

    def test_first_call_builds_once_then_hits_without_encoding(self):
        cold = self.build()
        warm = self.build()
        self.assertFalse(cold.hit)
        self.assertTrue(warm.hit)
        self.assertEqual(1, len(self.model.calls))
        np.testing.assert_array_equal(cold.embeddings, warm.embeddings)

    def test_same_encode_contract_as_original_pipeline(self):
        self.build()
        args = self.model.calls[0]
        self.assertEqual(["Paper one Alpha beta. Gamma.", "Paper two Delta."], args[0])
        self.assertEqual((32, True, True, True), args[1:])

    def test_array_is_normalized_finite_and_floating(self):
        got = self.build().embeddings
        self.assertEqual((2, 2), got.shape)
        np.testing.assert_allclose(np.linalg.norm(got, axis=1), 1, atol=1e-6)
        self.assertTrue(np.isfinite(got).all())

    def test_corpus_text_change_invalidates(self):
        self.build()
        changed = [{**d, "abstract": list(d["abstract"])} for d in DOCS]
        changed[0]["abstract"][0] += " changed"
        self.assertFalse(self.build(changed).hit)
        self.assertEqual(2, len(self.model.calls))

    def test_reordering_docs_invalidates(self):
        self.build()
        self.assertFalse(self.build(list(reversed(DOCS))).hit)

    def test_changed_encoder_key_invalidates(self):
        self.build()
        result = load_or_build_embeddings(DOCS, self.model, cache_dir=self.path,
                                          encoder_key="different-encoder")
        self.assertFalse(result.hit)

    def test_changed_encoder_revision_invalidates(self):
        self.build()
        self.assertFalse(self.build(encoder_revision="new-weights").hit)

    def test_document_id_changed_invalidates(self):
        self.build()
        altered = [{**DOCS[0], "doc_id": 12}, DOCS[1]]
        self.assertFalse(self.build(altered).hit)

    def test_rebuild_forces_encode(self):
        self.build()
        result = self.build(rebuild=True)
        self.assertFalse(result.hit)
        self.assertEqual(2, len(self.model.calls))

    def test_corrupted_npy_fails_closed(self):
        first = self.build()
        first.embeddings_path.write_bytes(b"corrupt")
        with self.assertRaisesRegex(InvalidEmbeddingCache, "Rebuild explicitly"):
            self.build()
        self.assertEqual(1, len(self.model.calls))
        self.assertFalse(self.build(rebuild=True).hit)

    def test_corrupted_metadata_fails_closed(self):
        first = self.build()
        first.embeddings_path.with_suffix(".json").write_text("{}")
        with self.assertRaises(InvalidEmbeddingCache):
            self.build()

    def test_incomplete_cache_fails_closed(self):
        first = self.build()
        first.embeddings_path.with_suffix(".json").unlink()
        with self.assertRaisesRegex(InvalidEmbeddingCache, "Incomplete"):
            self.build()

    def test_no_mutation(self):
        original = json.dumps(DOCS)
        self.build()
        self.build()
        self.assertEqual(original, json.dumps(DOCS))

    def test_source_hash_ignores_irrelevant_metadata(self):
        augmented = [{**doc, "unknown_field": "ignored"} for doc in DOCS]
        self.assertEqual(source_fingerprint(DOCS), source_fingerprint(augmented))

    def test_reject_duplicate_ids(self):
        with self.assertRaisesRegex(ValueError, "unique"):
            self.build([DOCS[0], DOCS[0]])

    def test_reject_blank_abstract(self):
        with self.assertRaises(ValueError):
            self.build([{"doc_id": 1, "title": "A", "abstract": [""]}])

    def test_reject_blank_encoder_revision(self):
        with self.assertRaises(ValueError):
            self.build(encoder_revision=" ")

    def test_reject_empty_documents(self):
        with self.assertRaises(ValueError):
            self.build([])

    def test_key_is_stable_for_same_sources(self):
        key1, spec1 = cache_identity(DOCS, encoder_key="x")
        key2, spec2 = cache_identity(DOCS, encoder_key="x")
        self.assertEqual((key1, spec1), (key2, spec2))

    def test_reject_non_unit_embedding(self):
        class BadEncoder(FakeEncoder):
            def encode(self, texts, **kw):
                return np.ones((len(texts), 2), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "unit-normalized"):
            load_or_build_embeddings(DOCS, BadEncoder(), cache_dir=self.path,
                                     encoder_key="bad")

    def test_reject_nan_embedding(self):
        class BadEncoder(FakeEncoder):
            def encode(self, texts, **kw):
                return np.full((len(texts), 2), np.nan, dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            load_or_build_embeddings(DOCS, BadEncoder(), cache_dir=self.path,
                                     encoder_key="bad")

    def test_cache_file_is_not_pickle(self):
        first = self.build()
        with first.embeddings_path.open("rb") as inp:
            self.assertEqual(b"\x93NUMPY", inp.read(6))
        self.assertEqual("float32", json.loads(
            first.embeddings_path.with_suffix(".json").read_text())["dtype"])


if __name__ == "__main__":
    unittest.main()
