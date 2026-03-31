"""Tests for scripts/prepare_sft_data.py — adapters, validation, hashing."""

import hashlib
import json

import prepare_sft_data as sft


# ─── normalize_text (SFT version — collapses newlines unlike CPT) ────────────


class TestNormalizeText:
    def test_collapses_newlines(self):
        assert sft.normalize_text("hello\n\nworld") == "hello world"

    def test_collapses_tabs_and_spaces(self):
        assert sft.normalize_text("hello\t\t  world") == "hello world"

    def test_nfc_normalization(self):
        assert sft.normalize_text("e\u0301") == "\u00e9"

    def test_empty_string(self):
        assert sft.normalize_text("") == ""

    def test_none_returns_empty(self):
        assert sft.normalize_text(None) == ""

    def test_already_clean(self):
        assert sft.normalize_text("hello world") == "hello world"


# ─── messages_hash ───────────────────────────────────────────────────────────


class TestMessagesHash:
    def test_deterministic(self):
        msgs = [{"role": "user", "content": "hello"}]
        assert sft.messages_hash(msgs) == sft.messages_hash(msgs)

    def test_different_content_different_hash(self):
        a = [{"role": "user", "content": "a"}]
        b = [{"role": "user", "content": "b"}]
        assert sft.messages_hash(a) != sft.messages_hash(b)

    def test_sort_keys_makes_key_order_irrelevant(self):
        a = [{"role": "user", "content": "hi"}]
        b = [{"content": "hi", "role": "user"}]
        assert sft.messages_hash(a) == sft.messages_hash(b)

    def test_swedish_content(self):
        msgs = [{"role": "user", "content": "Hej världen"}]
        h = sft.messages_hash(msgs)
        blob = json.dumps(msgs, ensure_ascii=False, sort_keys=True)
        expected = hashlib.sha256(blob.encode("utf-8")).hexdigest()
        assert h == expected


# ─── adapt_alpaca_swedish ────────────────────────────────────────────────────


class TestAdaptAlpacaSwedish:
    def test_with_input(self):
        ds = [{"instruction": "Översätt", "input": "Hello", "output": "Hej"}]
        result = sft.adapt_alpaca_swedish(ds, "test")
        assert len(result) == 1
        assert result[0]["source"] == "test"
        msgs = result[0]["messages"]
        assert msgs[0]["role"] == "user"
        assert "Översätt" in msgs[0]["content"]
        assert "Hello" in msgs[0]["content"]
        assert msgs[1]["role"] == "assistant"
        assert msgs[1]["content"] == "Hej"

    def test_without_input(self):
        ds = [{"instruction": "Säg hej", "input": "", "output": "Hej!"}]
        result = sft.adapt_alpaca_swedish(ds, "test")
        assert len(result) == 1
        assert result[0]["messages"][0]["content"] == "Säg hej"

    def test_skip_empty_instruction(self):
        ds = [{"instruction": "", "input": "", "output": "Hej"}]
        assert sft.adapt_alpaca_swedish(ds, "test") == []

    def test_skip_empty_output(self):
        ds = [{"instruction": "Säg hej", "input": "", "output": ""}]
        assert sft.adapt_alpaca_swedish(ds, "test") == []

    def test_multiple_rows(self):
        ds = [
            {"instruction": f"Fråga {i}", "input": "", "output": f"Svar {i}"}
            for i in range(3)
        ]
        assert len(sft.adapt_alpaca_swedish(ds, "test")) == 3


# ─── adapt_wikipedia_qa_sv ───────────────────────────────────────────────────


class TestAdaptWikipediaQaSv:
    def test_valid_pair(self):
        ds = [{"user": "Vad är Sverige?", "assistant": "Sverige är ett land i Skandinavien."}]
        result = sft.adapt_wikipedia_qa_sv(ds, "test")
        assert len(result) == 1
        assert result[0]["messages"][0]["content"] == "Vad är Sverige?"

    def test_short_assistant_filtered(self):
        ds = [{"user": "Fråga", "assistant": "Kort"}]  # < 10 chars
        assert sft.adapt_wikipedia_qa_sv(ds, "test") == []

    def test_empty_user_filtered(self):
        ds = [{"user": "", "assistant": "Tillräckligt långt svar"}]
        assert sft.adapt_wikipedia_qa_sv(ds, "test") == []

    def test_empty_assistant_filtered(self):
        ds = [{"user": "Fråga", "assistant": ""}]
        assert sft.adapt_wikipedia_qa_sv(ds, "test") == []


# ─── adapt_oasst2_sv ────────────────────────────────────────────────────────


def _oasst2_row(msg_id, parent_id, text, role, lang="sv", tree_id="tree1", deleted=False):
    return {
        "message_id": msg_id,
        "parent_id": parent_id,
        "text": text,
        "role": role,
        "lang": lang,
        "message_tree_id": tree_id,
        "deleted": deleted,
        "rank": None,
    }


class TestAdaptOasst2Sv:
    def test_simple_two_turn(self):
        ds = [
            _oasst2_row("m1", None, "Hej, hur mår du?", "prompter"),
            _oasst2_row("m2", "m1", "Jag mår bra, tack!", "assistant"),
        ]
        result = sft.adapt_oasst2_sv(ds, "test")
        assert len(result) == 1
        msgs = result[0]["messages"]
        assert len(msgs) == 2
        assert msgs[0]["role"] == "user"
        assert msgs[1]["role"] == "assistant"

    def test_four_turn_chain(self):
        ds = [
            _oasst2_row("m1", None, "Hej!", "prompter"),
            _oasst2_row("m2", "m1", "Hej! Hur kan jag hjälpa dig?", "assistant"),
            _oasst2_row("m3", "m2", "Berätta om Sverige", "prompter"),
            _oasst2_row("m4", "m3", "Sverige är ett nordiskt land.", "assistant"),
        ]
        result = sft.adapt_oasst2_sv(ds, "test")
        assert len(result) == 1
        assert len(result[0]["messages"]) == 4

    def test_filters_non_swedish(self):
        ds = [
            _oasst2_row("m1", None, "Hello", "prompter", lang="en"),
            _oasst2_row("m2", "m1", "Hi there", "assistant", lang="en"),
        ]
        assert sft.adapt_oasst2_sv(ds, "test") == []

    def test_filters_deleted(self):
        ds = [
            _oasst2_row("m1", None, "Hej", "prompter"),
            _oasst2_row("m2", "m1", "Hej!", "assistant", deleted=True),
        ]
        # m2 is deleted, so m1 becomes a leaf with only 1 message -> filtered
        assert sft.adapt_oasst2_sv(ds, "test") == []

    def test_branching_tree(self):
        ds = [
            _oasst2_row("m1", None, "Fråga", "prompter"),
            _oasst2_row("m2", "m1", "Svar ett", "assistant"),
            _oasst2_row("m3", "m1", "Svar två", "assistant"),
        ]
        result = sft.adapt_oasst2_sv(ds, "test")
        # Two leaves (m2, m3), each forming a 2-turn conversation
        assert len(result) == 2

    def test_single_message_filtered(self):
        ds = [_oasst2_row("m1", None, "Hej", "prompter")]
        assert sft.adapt_oasst2_sv(ds, "test") == []

    def test_root_is_assistant_filtered(self):
        ds = [
            _oasst2_row("m1", None, "Jag börjar", "assistant"),
            _oasst2_row("m2", "m1", "Okej", "prompter"),
        ]
        result = sft.adapt_oasst2_sv(ds, "test")
        # Chain starts with assistant -> filtered
        assert len(result) == 0


# ─── adapt_scandi_qa_sv ──────────────────────────────────────────────────────


class TestAdaptScandiQaSv:
    def test_with_context(self):
        ds = [{"question": "Vad?", "context": "Kontext här", "answers": {"text": ["Svaret"]}}]
        result = sft.adapt_scandi_qa_sv(ds, "test")
        assert len(result) == 1
        user_content = result[0]["messages"][0]["content"]
        assert "Givet följande text:" in user_content
        assert "Kontext här" in user_content
        assert "Vad?" in user_content

    def test_without_context(self):
        ds = [{"question": "Vad?", "context": "", "answers": {"text": ["Svaret"]}}]
        result = sft.adapt_scandi_qa_sv(ds, "test")
        assert result[0]["messages"][0]["content"] == "Vad?"

    def test_empty_answers(self):
        ds = [{"question": "Q", "context": "C", "answers": {"text": []}}]
        assert sft.adapt_scandi_qa_sv(ds, "test") == []

    def test_answers_not_dict(self):
        ds = [{"question": "Q", "context": "C", "answers": "bad"}]
        assert sft.adapt_scandi_qa_sv(ds, "test") == []


# ─── adapt_swedish_instruct_gpt4 ────────────────────────────────────────────


class TestAdaptSwedishInstructGpt4:
    def test_valid_pair(self):
        ds = [{"human": "Fråga", "gpt": "Ett tillräckligt långt svar"}]
        result = sft.adapt_swedish_instruct_gpt4(ds, "test")
        assert len(result) == 1
        assert result[0]["messages"][0]["role"] == "user"
        assert result[0]["messages"][1]["role"] == "assistant"

    def test_short_gpt_filtered(self):
        ds = [{"human": "Fråga", "gpt": "Kort"}]
        assert sft.adapt_swedish_instruct_gpt4(ds, "test") == []

    def test_empty_human_filtered(self):
        ds = [{"human": "", "gpt": "Svar som är tillräckligt"}]
        assert sft.adapt_swedish_instruct_gpt4(ds, "test") == []


# ─── validate_conversation ───────────────────────────────────────────────────


class TestValidateConversation:
    def test_valid_two_turn(self):
        conv = {"messages": [
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "A"},
        ]}
        assert sft.validate_conversation(conv) is True

    def test_valid_four_turn(self):
        conv = {"messages": [
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1"},
            {"role": "user", "content": "Q2"},
            {"role": "assistant", "content": "A2"},
        ]}
        assert sft.validate_conversation(conv) is True

    def test_too_few_messages(self):
        conv = {"messages": [{"role": "user", "content": "Q"}]}
        assert sft.validate_conversation(conv) is False

    def test_first_not_user(self):
        conv = {"messages": [
            {"role": "assistant", "content": "A"},
            {"role": "user", "content": "Q"},
        ]}
        assert sft.validate_conversation(conv) is False

    def test_last_not_assistant(self):
        conv = {"messages": [
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1"},
            {"role": "user", "content": "Q2"},
        ]}
        assert sft.validate_conversation(conv) is False

    def test_empty_content(self):
        conv = {"messages": [
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "   "},
        ]}
        assert sft.validate_conversation(conv) is False

    def test_missing_messages_key(self):
        assert sft.validate_conversation({}) is False

    def test_empty_messages_list(self):
        assert sft.validate_conversation({"messages": []}) is False
