"""Shared fixtures for liquidai-svenska tests."""

import json
import pathlib
import sys

import pytest
import yaml

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))


@pytest.fixture
def repo_root():
    return REPO_ROOT


@pytest.fixture
def sample_cpt_jsonl_path():
    return REPO_ROOT / "data" / "samples" / "cpt_sample.jsonl"


@pytest.fixture
def datasets_yaml_path():
    return REPO_ROOT / "configs" / "datasets.yaml"


@pytest.fixture
def eval_prompts_path():
    return REPO_ROOT / "prompts" / "eval_prompts_sv.txt"


@pytest.fixture
def tmp_jsonl(tmp_path):
    """Write a list of dicts as JSONL and return the path."""
    def _write(rows, filename="data.jsonl"):
        path = tmp_path / filename
        with open(path, "w") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return str(path)
    return _write


@pytest.fixture
def tmp_yaml(tmp_path):
    """Write a dict as YAML and return the path."""
    def _write(data, filename="config.yaml"):
        path = tmp_path / filename
        with open(path, "w") as f:
            yaml.dump(data, f)
        return str(path)
    return _write


@pytest.fixture
def tmp_textfile(tmp_path):
    """Write lines to a text file and return the path."""
    def _write(lines, filename="prompts.txt"):
        path = tmp_path / filename
        path.write_text("\n".join(lines) + "\n")
        return str(path)
    return _write
