"""Persistent, provenance-checked embeddings for the *existing* SciFact encoder.

Does not modify the training or ranking algorithms.  Cache identity depends on
ordered source text, document IDs, encoder identity/revision and encoding spec.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

CACHE_SCHEMA = 1
ENCODING_SPEC = "title-space-abstract-joined|batch32|unit-normalized|numpy"


class InvalidEmbeddingCache(ValueError):
    """An existing cache file failed provenance or data-integrity checks."""


@dataclass(frozen=True)
class EmbeddingCacheResult:
    embeddings: np.ndarray
    hit: bool
    cache_key: str
    embeddings_path: Path
    load_seconds: float
    build_seconds: float


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def source_fingerprint(documents: list[dict]) -> str:
    """Hash EXACTLY the ordered texts used by the existing semantic index."""
    if not isinstance(documents, list) or not documents:
        raise ValueError("documents must be a nonempty list")
    records = []
    seen = set()
    for doc in documents:
        if not isinstance(doc, dict):
            raise ValueError("Every document must be a dictionary")
        doc_id = doc.get("doc_id")
        title = doc.get("title")
        abstract = doc.get("abstract")
        if type(doc_id) is not int or doc_id <= 0 or doc_id in seen:
            raise ValueError("Document IDs must be unique positive integers")
        seen.add(doc_id)
        if not isinstance(title, str) or not title.strip():
            raise ValueError("Each document needs a nonblank title")
        if not isinstance(abstract, list) or not abstract or any(
            not isinstance(s, str) or not s.strip() for s in abstract
        ):
            raise ValueError("Each abstract needs a list of nonblank sentences")
        records.append([doc_id, title + " " + " ".join(abstract)])
    return hashlib.sha256(_json_bytes(records)).hexdigest()


def cache_identity(documents: list[dict], *, encoder_key: str,
                   encoder_revision: str = "default") -> tuple[str, dict]:
    if not isinstance(encoder_key, str) or not encoder_key.strip():
        raise ValueError("encoder_key must be a nonblank string")
    if not isinstance(encoder_revision, str) or not encoder_revision.strip():
        raise ValueError("encoder_revision must be a nonblank string")
    spec = {
        "schema": CACHE_SCHEMA,
        "ordered_sources_sha256": source_fingerprint(documents),
        "document_count": len(documents),
        "encoder_key": encoder_key.strip(),
        "encoder_revision": encoder_revision.strip(),
        "encoding_spec": ENCODING_SPEC,
    }
    return hashlib.sha256(_json_bytes(spec)).hexdigest(), spec


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as src:
        for block in iter(lambda: src.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _check_array(value: np.ndarray, expected_rows: int) -> np.ndarray:
    result = np.asarray(value)
    if (result.ndim != 2 or result.shape[0] != expected_rows
            or result.shape[1] < 1 or result.dtype.kind != "f"):
        raise ValueError("Embeddings must be an N x D floating-point matrix")
    if not np.isfinite(result).all():
        raise ValueError("Embedding matrix has nonfinite values")
    norms = np.linalg.norm(result, axis=1)
    if not np.allclose(norms, 1.0, rtol=1e-3, atol=1e-3):
        raise ValueError("Embeddings must be unit-normalized")
    return result


def _atomic_json(path: Path, data: dict) -> None:
    fd, tmp = tempfile.mkstemp(prefix=".tmp-meta-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as out:
            json.dump(data, out, indent=2, sort_keys=True)
            out.write("\n")
            out.flush()
            os.fsync(out.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _atomic_array(path: Path, value: np.ndarray) -> None:
    fd, tmp = tempfile.mkstemp(prefix=".tmp-emb-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as out:
            np.save(out, value, allow_pickle=False)
            out.flush()
            os.fsync(out.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def load_or_build_embeddings(
    documents: list[dict], model: Any, *, cache_dir: Path | str,
    encoder_key: str, encoder_revision: str = "default", rebuild: bool = False,
) -> EmbeddingCacheResult:
    """Load verified cached embeddings or compute with original encode options.

    A corrupt cache fails closed with a clear error; use rebuild=True explicitly.
    The caller must bump encoder_revision when the encoder weights change.
    """
    key, spec = cache_identity(documents, encoder_key=encoder_key,
                               encoder_revision=encoder_revision)
    cache_dir = Path(cache_dir)
    meta_path = cache_dir / (key + ".json")
    array_path = cache_dir / (key + ".npy")
    started = perf_counter()

    if not rebuild and meta_path.exists() and array_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if (meta.get("identity") != spec
                    or meta.get("array_sha256") != _sha256_file(array_path)):
                raise InvalidEmbeddingCache("Embedding cache identity or checksum mismatch")
            array = _check_array(np.load(array_path, allow_pickle=False), len(documents))
            if list(array.shape) != meta.get("shape") or str(array.dtype) != meta.get("dtype"):
                raise InvalidEmbeddingCache("Embedding cache shape or dtype mismatch")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise InvalidEmbeddingCache(
                f"Invalid embedding cache at {array_path}. Rebuild explicitly "
                "with --rebuild-cache (or remove both matching cache files)."
            ) from exc
        return EmbeddingCacheResult(array, True, key, array_path,
                                    perf_counter() - started, 0.0)

    if not rebuild and (meta_path.exists() != array_path.exists()):
        raise InvalidEmbeddingCache(
            f"Incomplete embedding cache at {array_path}. Use --rebuild-cache."
        )

    texts = [doc["title"] + " " + " ".join(doc["abstract"]) for doc in documents]
    building = perf_counter()
    embeddings = _check_array(np.asarray(model.encode(
        texts, batch_size=32, convert_to_numpy=True,
        normalize_embeddings=True, show_progress_bar=True,
    )), len(documents))
    build_seconds = perf_counter() - building
    cache_dir.mkdir(parents=True, exist_ok=True)
    _atomic_array(array_path, embeddings)
    _atomic_json(meta_path, {
        "identity": spec,
        "shape": list(embeddings.shape),
        "dtype": str(embeddings.dtype),
        "array_sha256": _sha256_file(array_path),
    })
    return EmbeddingCacheResult(embeddings, False, key, array_path,
                                0.0, build_seconds)
