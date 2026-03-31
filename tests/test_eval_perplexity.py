"""Tests for scripts/eval_perplexity.py — eval data loading."""

import pytest

torch = pytest.importorskip("torch")
import eval_perplexity  # noqa: E402


# ─── load_eval_texts ────────────────────────────────────────────────────────


class TestLoadEvalTexts:
    def test_valid_jsonl(self, tmp_jsonl):
        path = tmp_jsonl([{"text": "a"}, {"text": "b"}])
        texts = eval_perplexity.load_eval_texts(path)
        assert texts == ["a", "b"]

    def test_skips_empty_text(self, tmp_jsonl):
        path = tmp_jsonl([{"text": "a"}, {"text": ""}, {"text": "c"}])
        texts = eval_perplexity.load_eval_texts(path)
        assert texts == ["a", "c"]

    def test_missing_text_field(self, tmp_jsonl):
        path = tmp_jsonl([{"other": "x"}])
        texts = eval_perplexity.load_eval_texts(path)
        assert texts == []

    def test_real_sample_data(self, sample_cpt_jsonl_path):
        texts = eval_perplexity.load_eval_texts(str(sample_cpt_jsonl_path))
        assert len(texts) >= 5
        for t in texts:
            assert len(t) > 0
