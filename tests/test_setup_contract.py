"""Tests for dependency bounds and release Streamlit watcher configuration."""
import unittest
from pathlib import Path

from scripts.check_setup import version_in_range

ROOT = Path(__file__).resolve().parents[1]


class SetupContractTests(unittest.TestCase):
    def test_transformers_5_not_allowed_by_release(self):
        self.assertTrue(version_in_range("4.57.2", 4, 5))
        self.assertFalse(version_in_range("5.20.0", 4, 5))

    def test_sentence_transformers_6_not_allowed_by_release(self):
        self.assertTrue(version_in_range("5.1.0", 3, 6))
        self.assertFalse(version_in_range("6.1.0", 3, 6))

    def test_bad_version_fails_closed(self):
        self.assertFalse(version_in_range("", 4, 5))
        self.assertFalse(version_in_range("broken", 4, 5))

    def test_streamlit_watcher_disabled_in_repository(self):
        from configparser import ConfigParser
        parser = ConfigParser()
        files = parser.read(ROOT / ".streamlit" / "config.toml")
        self.assertTrue(files)
        self.assertEqual(parser.get("server", "fileWatcherType").strip('"'), "none")

    def test_requirements_bounds_match_setup(self):
        text = (ROOT / "requirements.txt").read_text(encoding="utf-8")
        self.assertIn("sentence-transformers>=3,<6", text)
        self.assertIn("transformers>=4.40,<5", text)


if __name__ == "__main__":
    unittest.main()
