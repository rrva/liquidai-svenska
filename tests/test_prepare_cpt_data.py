"""Tests for scripts/prepare_cpt_data.py — text processing, quality filters, file loading."""

import hashlib
from unittest.mock import patch

import pytest

import prepare_cpt_data as cpt


# ─── normalize_text ──────────────────────────────────────────────────────────


class TestNormalizeText:
    def test_nfc_normalization(self):
        decomposed = "e\u0301"  # e + combining accent
        assert cpt.normalize_text(decomposed) == "\u00e9"

    def test_preserves_internal_whitespace(self):
        # CPT normalize_text strips per-line but does NOT collapse internal spaces
        assert cpt.normalize_text("hello   world") == "hello   world"

    def test_preserve_paragraph_breaks(self):
        text = "para one\n\npara two"
        assert cpt.normalize_text(text) == "para one\n\npara two"

    def test_strip_per_line_whitespace(self):
        assert cpt.normalize_text("  hello  \n  world  ") == "hello\nworld"

    def test_collapse_multiple_blank_lines(self):
        assert cpt.normalize_text("a\n\n\n\nb") == "a\n\nb"

    def test_strip_outer_whitespace(self):
        assert cpt.normalize_text("\n\nhello\n\n") == "hello"

    def test_empty_string(self):
        assert cpt.normalize_text("") == ""

    def test_swedish_characters(self):
        text = "Hej världen! Åäö fungerar."
        assert cpt.normalize_text(text) == text


# ─── text_hash ───────────────────────────────────────────────────────────────


class TestTextHash:
    def test_known_hash(self):
        expected = hashlib.sha256(b"hello").hexdigest()
        assert cpt.text_hash("hello") == expected

    def test_different_text_different_hash(self):
        assert cpt.text_hash("hello") != cpt.text_hash("hallo")

    def test_deterministic(self):
        assert cpt.text_hash("test") == cpt.text_hash("test")

    def test_swedish_characters(self):
        h = cpt.text_hash("Hej världen")
        expected = hashlib.sha256("Hej världen".encode("utf-8")).hexdigest()
        assert h == expected


# ─── detect_swedish ──────────────────────────────────────────────────────────


try:
    import langdetect as _langdetect
    _has_langdetect = True
except ImportError:
    _has_langdetect = False


@pytest.mark.skipif(not _has_langdetect, reason="langdetect not installed")
class TestDetectSwedish:
    @pytest.fixture(autouse=True)
    def _seed_langdetect(self):
        from langdetect import DetectorFactory
        DetectorFactory.seed = 0

    def test_swedish_text(self):
        text = "Sverige är ett land i Skandinavien med en befolkning på cirka tio miljoner människor."
        assert cpt.detect_swedish(text) is True

    def test_english_text(self):
        text = (
            "The United Kingdom is a country in Western Europe. London is the capital city. "
            "The population is approximately sixty-seven million people. English is the primary "
            "language spoken throughout the country. Parliament governs from Westminster."
        )
        assert cpt.detect_swedish(text) is False

    def test_empty_string_returns_true(self):
        # langdetect raises on empty string, so graceful fallback
        assert cpt.detect_swedish("") is True


class TestDetectSwedishFallback:
    """Test exception handling without requiring langdetect."""

    def test_exception_returns_false(self):
        """When langdetect raises, detect_swedish rejects the document."""
        import sys
        import types
        fake = types.ModuleType("langdetect")
        fake.detect_langs = lambda text: (_ for _ in ()).throw(Exception("mock"))
        with patch.dict(sys.modules, {"langdetect": fake}):
            assert cpt.detect_swedish("anything") is False

    def test_no_langdetect_returns_false(self):
        """When langdetect import fails entirely, detect_swedish rejects the document."""
        import sys
        with patch.dict(sys.modules, {"langdetect": None}):
            assert cpt.detect_swedish("anything") is False


# ─── passes_quality ──────────────────────────────────────────────────────────


class TestPassesQuality:
    def _make_text(self, n_chars, alpha_ratio=0.8):
        """Generate text with approximately n_chars characters and given alpha ratio."""
        n_alpha = int(n_chars * alpha_ratio)
        n_other = n_chars - n_alpha
        # Make word-like text
        words = []
        remaining_alpha = n_alpha
        remaining_other = n_other
        while remaining_alpha > 0 or remaining_other > 0:
            if remaining_alpha > 0:
                word_len = min(5, remaining_alpha)
                words.append("a" * word_len)
                remaining_alpha -= word_len
            if remaining_other > 0:
                words.append("1")
                remaining_other -= 1
        return " ".join(words)

    def test_passing_text(self):
        text = self._make_text(500)
        assert cpt.passes_quality(text, 200, 100000, 30) is True

    def test_too_short(self):
        text = self._make_text(100)
        assert cpt.passes_quality(text, 200, 100000, 30) is False

    def test_too_long(self):
        text = self._make_text(2000)
        assert cpt.passes_quality(text, 200, 1000, 30) is False

    def test_too_few_words(self):
        text = "abcde " * 10  # 10 words
        assert cpt.passes_quality(text, 10, 100000, 30) is False

    def test_low_alpha_ratio(self):
        text = "12345 " * 100  # mostly digits
        assert cpt.passes_quality(text, 10, 100000, 5) is False

    def test_boundary_exactly_min_chars(self):
        text = "abcde " * 40  # 240 chars, 40 words
        assert cpt.passes_quality(text, len(text), 100000, 30) is True

    def test_boundary_exactly_max_chars(self):
        text = "abcde " * 40
        assert cpt.passes_quality(text, 10, len(text), 30) is True


# ─── load_config ─────────────────────────────────────────────────────────────


class TestLoadConfig:
    def test_load_real_datasets_config(self, datasets_yaml_path):
        cfg = cpt.load_config(str(datasets_yaml_path))
        assert "cpt_sources" in cfg
        assert "sft_sources" in cfg
        assert "quality" in cfg
        assert "splits" in cfg

    def test_load_synthetic_yaml(self, tmp_yaml):
        path = tmp_yaml({"key": "value", "nested": {"a": 1}})
        cfg = cpt.load_config(path)
        assert cfg == {"key": "value", "nested": {"a": 1}}


# ─── load_local_source ──────────────────────────────────────────────────────


class TestLoadLocalSource:
    def test_load_real_sample(self, sample_cpt_jsonl_path):
        cfg = {"path": str(sample_cpt_jsonl_path), "name": "test", "text_field": "text"}
        docs = cpt.load_local_source(cfg)
        assert len(docs) >= 5
        for doc in docs:
            assert "text" in doc
            assert "source" in doc
            assert doc["source"] == "test"
            assert len(doc["text"]) > 0

    def test_custom_text_field(self, tmp_jsonl):
        path = tmp_jsonl([{"content": "abc"}, {"content": "def"}])
        cfg = {"path": path, "name": "test", "text_field": "content"}
        docs = cpt.load_local_source(cfg)
        assert len(docs) == 2
        assert docs[0]["text"] == "abc"

    def test_skips_empty_text(self, tmp_jsonl):
        path = tmp_jsonl([{"text": "keep"}, {"text": ""}, {"text": "also keep"}])
        cfg = {"path": path, "name": "test", "text_field": "text"}
        docs = cpt.load_local_source(cfg)
        assert len(docs) == 2
