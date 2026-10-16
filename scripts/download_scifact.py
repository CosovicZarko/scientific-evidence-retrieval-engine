"""Download the official SciFact data archive and extract validated JSONL files.

Run: python scripts/download_scifact.py
"""
from __future__ import annotations

import argparse
import json
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_URL = "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"


def extract_data_archive(archive: Path, destination: Path):
    """Only extract regular files inside data/; do not overwrite an existing corpus."""
    if destination.is_symlink():
        raise ValueError(f"{destination} is a symbolic link; refuse to follow it")
    if (destination / "corpus.jsonl").exists():
        raise FileExistsError(f"Corpus already exists: {destination / 'corpus.jsonl'}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".scifact-install-", dir=destination.parent) as folder:
        staging = Path(folder) / "data"
        staging.mkdir()
        with tarfile.open(archive, "r:gz") as tar:
            for member in tar.getmembers():
                parts = Path(member.name).parts
                # Some distributions include macOS metadata entries; ignore them.
                if parts and (parts[0].startswith("._") or any(
                    part.startswith("._") or part == ".DS_Store" for part in parts
                )):
                    continue
                if not parts or parts[0] != "data" or ".." in parts:
                    raise ValueError(f"Unexpected SciFact archive member: {member.name}")
                if member.isdir():
                    continue
                if not member.isfile() or len(parts) < 2:
                    raise ValueError(f"Unsafe SciFact archive member: {member.name}")
                target = staging.joinpath(*parts[1:])
                target.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(member) as src, target.open("wb") as dest:
                    shutil.copyfileobj(src, dest)
        corpus = staging / "corpus.jsonl"
        if not corpus.is_file():
            raise ValueError("Official archive lacks data/corpus.jsonl")
        with corpus.open(encoding="utf-8") as file:
            ids = [json.loads(line)["doc_id"] for line in file if line.strip()]
        if len(ids) != 5183 or len(set(ids)) != 5183:
            raise ValueError("Unexpected SciFact corpus size or duplicate document IDs")
        if destination.exists():
            if any(destination.iterdir()):
                raise FileExistsError(
                    f"Data directory contains files; refusing overwrite: {destination}"
                )
            destination.rmdir()
        staging.rename(destination)
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="Use a locally downloaded data.tar.gz")
    args = parser.parse_args(argv)
    data_dir = ROOT / "data"
    if (data_dir / "corpus.jsonl").is_file():
        print("SciFact corpus already installed; skipping download")
        return
    try:
        if args.archive:
            print(f"Using SciFact archive: {args.archive}")
            extract_data_archive(args.archive.expanduser(), data_dir)
        else:
            with tempfile.TemporaryDirectory() as folder:
                archive = Path(folder) / "data.tar.gz"
                print(f"Downloading official SciFact corpus from {DATA_URL}...")
                urllib.request.urlretrieve(DATA_URL, archive)
                extract_data_archive(archive, data_dir)
    except (OSError, ValueError, KeyError, tarfile.TarError) as exc:
        parser.exit(1, f"SciFact installation failed: {exc}\n")
    print(f"SciFact corpus installed: {data_dir / 'corpus.jsonl'}")


if __name__ == "__main__":
    main()
