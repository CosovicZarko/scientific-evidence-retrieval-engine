"""Check local corpus, model artifacts, installed packages and major versions.

No model downloads, network access, or model loading. Run from anywhere:
    python scripts/check_setup.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FILES = (
    "app_retrieval.py",
    "evidence_pipeline.py",
    "cached_hybrid_retriever.py",
    "retrieval_embedding_cache.py",
    ".streamlit/config.toml",
    "data/corpus.jsonl",
    "checkpoints/scifact_reranker/config.json",
    "checkpoints/scifact_reranker/model.safetensors",
    "checkpoints/scifact_reranker/tokenizer.json",
)

# Open-ended minor/patch versions, but bound major versions to avoid
# incompatibilities introduced by future breaking major releases.
DEPENDENCIES = {
    "numpy": ("numpy", 1, 3),
    "torch": ("torch", 2, 3),
    "streamlit": ("streamlit", 1, 2),
    "sentence-transformers": ("sentence_transformers", 3, 6),
    "transformers": ("transformers", 4, 5),
    "rank-bm25": ("rank_bm25", 0, 1),
    "scikit-learn": ("sklearn", 1, 2),
    "pandas": ("pandas", 2, 3),
}


def version_in_range(value: str, min_major: int, max_major: int) -> bool:
    """Check supported major bounds; full constraints live in requirements.txt."""
    try:
        major = int(value.split(".", 1)[0])
    except (ValueError, AttributeError):
        return False
    return min_major <= major < max_major


def check(root: Path = ROOT) -> list[str]:
    issues = []
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version_info.major}.{sys.version_info.minor}")
    if sys.version_info < (3, 10):
        issues.append("Python >=3.10 required")

    for relative in REQUIRED_FILES:
        exists = (root / relative).is_file()
        print(f"{'OK' if exists else 'MISSING'}: {relative}")
        if not exists:
            issues.append(relative)

    for distribution, (module, lo, hi) in DEPENDENCIES.items():
        try:
            installed = version(distribution)
        except PackageNotFoundError:
            installed = None
        if installed is None or importlib.util.find_spec(module) is None:
            print(f"MISSING Python dependency: {distribution}")
            issues.append(f"Missing {distribution}")
        elif not version_in_range(installed, lo, hi):
            print(f"OUT OF RANGE: {distribution}=={installed}; expected major in [{lo}, {hi})")
            issues.append(f"Incompatible {distribution} {installed}")
        else:
            print(f"OK dependency: {distribution}=={installed}")

    if (root / "checkpoints").is_symlink():
        issues.append("Checkpoints directory must not link outside repository")

    corpus = root / "data" / "corpus.jsonl"
    if corpus.is_file():
        try:
            with corpus.open(encoding="utf-8") as source:
                ids = [json.loads(line)["doc_id"] for line in source if line.strip()]
            if len(ids) != 5183 or len(set(ids)) != len(ids):
                raise ValueError("Unexpected SciFact corpus size or duplicate document IDs")
            print(f"OK corpus: {len(ids)} unique SciFact abstracts")
        except (ValueError, KeyError, TypeError, OSError) as exc:
            issues.append(f"Corpus validation: {exc}")
    return issues


def main() -> int:
    issues = check()
    if issues:
        print(f"\nSETUP INCOMPLETE ({len(issues)} issue(s)): {issues}")
        print("Run: python -m pip install -r requirements.txt")
        print("Then: python -m pip check")
        return 1
    print("\nSETUP VALIDATION PASSED (full model inference tested separately)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
