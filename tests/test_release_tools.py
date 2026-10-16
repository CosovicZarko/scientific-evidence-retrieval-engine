"""Offline release checks; no transformer, internet, or model downloads."""
import hashlib
import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.download_scifact import extract_data_archive
from scripts.install_checkpoint import install


class ReleaseToolsTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)

    def model_archive(self, members):
        path = self.root / "reranker.tar.gz"
        with tarfile.open(path, "w:gz") as tar:
            for name, content in members.items():
                data = content.encode()
                member = tarfile.TarInfo(name)
                member.size = len(data)
                tar.addfile(member, io.BytesIO(data))
        return path, hashlib.sha256(path.read_bytes()).hexdigest()

    def test_valid_model_archive_installs_once_without_mutating_source(self):
        archive, digest = self.model_archive({
            "scifact_reranker/config.json": "{}",
            "scifact_reranker/model.safetensors": "fake",
            "scifact_reranker/tokenizer.json": "{}",
            "scifact_reranker/tokenizer_config.json": "{}",
        })
        target = install(archive, self.root / "checkpoints", expected_sha=digest)
        self.assertTrue((target / "model.safetensors").is_file())
        self.assertTrue(archive.is_file())
        with self.assertRaises(FileExistsError):
            install(archive, self.root / "checkpoints", expected_sha=digest)

    def test_model_hash_mismatch_rejected_before_directory_creation(self):
        archive, digest = self.model_archive({"scifact_reranker/config.json": "{}"})
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            install(archive, self.root / "checkpoints", expected_sha="0" * 64)
        self.assertFalse((self.root / "checkpoints").exists())

    def test_model_path_traversal_rejected(self):
        archive, digest = self.model_archive({
            "scifact_reranker/../evil": "bad",
            "scifact_reranker/config.json": "{}",
        })
        with self.assertRaisesRegex(ValueError, "Unexpected archive"):
            install(archive, self.root / "checkpoints", expected_sha=digest)
        self.assertFalse((self.root / "evil").exists())

    def test_symlink_checkpoint_path_refused(self):
        archive, digest = self.model_archive({"scifact_reranker/config.json": "{}"})
        (self.root / "original").mkdir()
        (self.root / "checkpoints").symlink_to(self.root / "original")
        with self.assertRaisesRegex(ValueError, "symbolic link"):
            install(archive, self.root / "checkpoints", expected_sha=digest)
        self.assertTrue((self.root / "checkpoints").is_symlink())

    def test_malformed_data_archive_rejected(self):
        archive, _ = self.model_archive({"../../data/corpus.jsonl": "{}"})
        with self.assertRaises(ValueError):
            extract_data_archive(archive, self.root / "data")

    def test_data_archive_with_5183_docs_and_mac_metadata(self):
        archive = self.root / "scifact.tar.gz"
        payload = "".join(json.dumps({"doc_id": i, "title": "test", "abstract": ["text"]}) + "\n"
                          for i in range(1, 5184)).encode()
        with tarfile.open(archive, "w:gz") as tar:
            for name, data in [("._data", b"mac metadata"), ("data/corpus.jsonl", payload)]:
                info = tarfile.TarInfo(name)
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
        dest = extract_data_archive(archive, self.root / "data")
        self.assertTrue((dest / "corpus.jsonl").is_file())
        self.assertFalse((self.root / "._data").exists())
        with self.assertRaises(FileExistsError):
            extract_data_archive(archive, dest)


if __name__ == "__main__":
    unittest.main()
