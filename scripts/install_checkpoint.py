"""Safely install the published reranker artifact without modifying the original project.

Run: python scripts/install_checkpoint.py ~/Downloads/scifact-reranker-v1.0.tar.gz
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SHA256 = "fb47bab93e50fad6368e26b2cdd3ba0a046cae3e5b47abe6daa0a85c6f8ce458"
REQUIRED = (
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def install(archive: Path, checkpoints_dir: Path, expected_sha: str = EXPECTED_SHA256):
    """Extract only regular model files, reject unsafe entries, and install once."""
    if not archive.is_file():
        raise FileNotFoundError(f"Checkpoint archive not found: {archive}")
    found = sha256(archive)
    if found != expected_sha:
        raise ValueError(f"Archive checksum mismatch: expected {expected_sha}; got {found}")
    if checkpoints_dir.is_symlink():
        raise ValueError(
            f"{checkpoints_dir} is a symbolic link. Remove only that link using "
            f"`unlink checkpoints` from the release directory before installing."
        )
    target = checkpoints_dir / "scifact_reranker"
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"Checkpoint already exists; refusing overwrite: {target}")
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".checkpoint-install-", dir=checkpoints_dir) as folder:
        staging = Path(folder) / "scifact_reranker"
        staging.mkdir()
        with tarfile.open(archive, "r:gz") as tar:
            for member in tar.getmembers():
                parts = Path(member.name).parts
                if not parts or parts[0] != "scifact_reranker" or (
                    len(parts) > 2 or ".." in parts
                ):
                    raise ValueError(f"Unexpected archive member path: {member.name}")
                if member.isdir():
                    continue
                if not member.isfile() or len(parts) != 2:
                    raise ValueError(f"Unsafe archive entry: {member.name}")
                with tar.extractfile(member) as src, (staging / parts[1]).open("wb") as dest:
                    shutil.copyfileobj(src, dest)
        for name in REQUIRED:
            if not (staging / name).is_file():
                raise ValueError(f"Checkpoint archive is missing {name}")
        staging.rename(target)
    return target


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="Path to scifact-reranker-v1.0.tar.gz")
    args = parser.parse_args(argv)
    try:
        target = install(args.archive.expanduser(), ROOT / "checkpoints")
    except (FileNotFoundError, FileExistsError, ValueError, OSError, tarfile.TarError) as exc:
        parser.exit(1, f"Checkpoint setup failed: {exc}\n")
    print(f"Trained reranker installed: {target}")


if __name__ == "__main__":
    main()
