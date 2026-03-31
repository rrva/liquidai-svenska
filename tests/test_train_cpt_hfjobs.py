"""Tests for scripts/train_cpt_hfjobs.py — manifest loading."""

import pytest

torch = pytest.importorskip("torch")
import train_cpt_hfjobs as cpt_train  # noqa: E402


# ─── load_manifest ──────────────────────────────────────────────────────────


class TestLoadManifest:
    def test_valid_jsonl(self, tmp_jsonl):
        path = tmp_jsonl([{"text": "a"}, {"text": "b"}])
        texts = cpt_train.load_manifest(path)
        assert texts == ["a", "b"]

    def test_skips_empty_text(self, tmp_jsonl):
        path = tmp_jsonl([{"text": ""}, {"text": "b"}])
        texts = cpt_train.load_manifest(path)
        assert texts == ["b"]

    def test_real_sample_data(self, sample_cpt_jsonl_path):
        texts = cpt_train.load_manifest(str(sample_cpt_jsonl_path))
        assert len(texts) >= 5
        for t in texts:
            assert isinstance(t, str)
            assert len(t) > 0
