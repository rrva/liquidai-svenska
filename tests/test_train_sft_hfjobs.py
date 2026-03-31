"""Tests for scripts/train_sft_hfjobs.py — manifest loading, format conversion."""

import train_sft_hfjobs as sft_train


# ─── load_sft_manifest ──────────────────────────────────────────────────────


class TestLoadSftManifest:
    def test_valid_manifest(self, tmp_jsonl):
        path = tmp_jsonl([
            {"messages": [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}], "source": "s"},
            {"messages": [{"role": "user", "content": "Q2"}, {"role": "assistant", "content": "A2"}], "source": "s"},
        ])
        rows = sft_train.load_sft_manifest(path)
        assert len(rows) == 2

    def test_too_few_messages_filtered(self, tmp_jsonl):
        path = tmp_jsonl([
            {"messages": [{"role": "user", "content": "Q"}], "source": "s"},
        ])
        rows = sft_train.load_sft_manifest(path)
        assert len(rows) == 0

    def test_missing_messages_key_filtered(self, tmp_jsonl):
        path = tmp_jsonl([{"source": "s"}])
        rows = sft_train.load_sft_manifest(path)
        assert len(rows) == 0

    def test_mixed_valid_and_invalid(self, tmp_jsonl):
        path = tmp_jsonl([
            {"messages": [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}]},
            {"messages": [{"role": "user", "content": "Q"}]},  # too few
            {"source": "s"},  # no messages
        ])
        rows = sft_train.load_sft_manifest(path)
        assert len(rows) == 1


# ─── to_conversations_format ────────────────────────────────────────────────


class TestToConversationsFormat:
    def test_single_conversation(self):
        rows = [{"messages": [
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "A"},
        ]}]
        result = sft_train.to_conversations_format(rows)
        assert len(result) == 1
        convs = result[0]["conversations"]
        assert convs[0] == {"from": "user", "value": "Q"}
        assert convs[1] == {"from": "assistant", "value": "A"}

    def test_multi_turn(self):
        rows = [{"messages": [
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1"},
            {"role": "user", "content": "Q2"},
            {"role": "assistant", "content": "A2"},
        ]}]
        result = sft_train.to_conversations_format(rows)
        assert len(result[0]["conversations"]) == 4

    def test_multiple_rows(self):
        rows = [
            {"messages": [{"role": "user", "content": f"Q{i}"}, {"role": "assistant", "content": f"A{i}"}]}
            for i in range(3)
        ]
        result = sft_train.to_conversations_format(rows)
        assert len(result) == 3

    def test_preserves_swedish_content(self):
        rows = [{"messages": [
            {"role": "user", "content": "Vad är Sveriges huvudstad?"},
            {"role": "assistant", "content": "Stockholm är Sveriges huvudstad."},
        ]}]
        result = sft_train.to_conversations_format(rows)
        assert result[0]["conversations"][1]["value"] == "Stockholm är Sveriges huvudstad."
