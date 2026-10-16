"""Prepare SciFact corpus and release model for hosted Streamlit deployments.

The GitHub repository intentionally excludes the dataset and model weights. On
first use, download each from its authoritative source, validate and extract
it with the repository's existing safety-checked installers. Later searches
reuse the installed files and the persistent embedding cache.
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from urllib.request import urlretrieve

from scripts.download_scifact import DATA_URL, extract_data_archive
from scripts.install_checkpoint import REQUIRED, install

MODEL_URL = (
    "https://github.com/CosovicZarko/scientific-evidence-retrieval-engine/"
    "releases/download/v1.0/scifact-reranker-v1.0.tar.gz"
)


def prepare_assets(root: Path, downloader=urlretrieve) -> None:
    root = Path(root)
    corpus = root / "data" / "corpus.jsonl"
    checkpoint_dir = root / "checkpoints"
    model_dir = checkpoint_dir / "scifact_reranker"

    if not corpus.is_file():
        print("Preparing SciFact corpus...", flush=True)
        with tempfile.TemporaryDirectory(prefix="scifact-download-") as tmp:
            archive = Path(tmp) / "data.tar.gz"
            downloader(DATA_URL, str(archive))
            extract_data_archive(archive, root / "data")
        print("SciFact corpus ready.", flush=True)

    if not all((model_dir / name).is_file() for name in REQUIRED):
        if model_dir.exists() or model_dir.is_symlink():
            raise RuntimeError(
                "Incomplete reranker checkpoint; inspect checkpoints/scifact_reranker."
            )
        print("Preparing fine-tuned SciFact reranker...", flush=True)
        with tempfile.TemporaryDirectory(prefix="reranker-download-") as tmp:
            archive = Path(tmp) / "scifact-reranker-v1.0.tar.gz"
            downloader(MODEL_URL, str(archive))
            # install validates SHA-256 and archive content before placing files
            install(archive, checkpoint_dir)
        print("Fine-tuned reranker ready.", flush=True)
