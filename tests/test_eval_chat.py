"""Tests for scripts/eval_chat.py — prompt loading, markdown formatting."""

import pytest

torch = pytest.importorskip("torch")
import eval_chat  # noqa: E402


# ─── load_prompts ───────────────────────────────────────────────────────────


class TestLoadPrompts:
    def test_real_prompts_file(self, eval_prompts_path):
        prompts = eval_chat.load_prompts(str(eval_prompts_path))
        assert len(prompts) > 20
        for p in prompts:
            assert not p.startswith("#")
            assert p.strip() == p
            assert len(p) > 0

    def test_skips_comments(self, tmp_textfile):
        path = tmp_textfile(["# comment", "prompt1", "# another comment", "prompt2"])
        prompts = eval_chat.load_prompts(path)
        assert prompts == ["prompt1", "prompt2"]

    def test_skips_blank_lines(self, tmp_textfile):
        path = tmp_textfile(["a", "", "", "b"])
        prompts = eval_chat.load_prompts(path)
        assert prompts == ["a", "b"]

    def test_all_comments(self, tmp_textfile):
        path = tmp_textfile(["# only", "# comments"])
        prompts = eval_chat.load_prompts(path)
        assert prompts == []


# ─── format_comparison ──────────────────────────────────────────────────────


class TestFormatComparison:
    def test_two_way(self):
        md = eval_chat.format_comparison(["P1"], ["B1"], ["C1"])
        assert "Base vs CPT" in md
        assert "**Prompt:** P1" in md
        assert "**Base:**" in md
        assert "**CPT:**" in md
        assert "**SFT:**" not in md

    def test_three_way(self):
        md = eval_chat.format_comparison(["P1"], ["B1"], ["C1"], ["S1"])
        assert "Base vs CPT vs SFT" in md
        assert "**SFT:**" in md
        assert "S1" in md

    def test_multiple_prompts(self):
        md = eval_chat.format_comparison(
            ["P1", "P2", "P3"],
            ["B1", "B2", "B3"],
            ["C1", "C2", "C3"],
        )
        assert "## Prompt 1" in md
        assert "## Prompt 2" in md
        assert "## Prompt 3" in md
