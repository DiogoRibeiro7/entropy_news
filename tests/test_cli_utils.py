"""Tests for CLI helper utilities."""

import json
from pathlib import Path

import pytest


def _write_vocab(path: Path) -> None:
    data = {"<PAD>": 0, "hello": 1}
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"vocab_size": len(data), "vocab": data}, f)


def test_load_model_and_vocab_missing_model(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    from entropy_news.utils.cli import load_model_and_vocab

    vocab_file = tmp_path / "vocab.json"
    _write_vocab(vocab_file)
    with pytest.raises(FileNotFoundError):
        load_model_and_vocab(
            str(vocab_file),
            str(tmp_path / "missing.pth"),
            embed_dim=10,
            hidden_dim=16,
            num_layers=2,
            dropout=0.1,
        )


def test_load_encoded_dataset(tmp_path) -> None:
    pytest.importorskip("numpy")
    from entropy_news.data import TextPreprocessor
    from entropy_news.utils.cli import load_encoded_dataset

    text_file = tmp_path / "data.txt"
    text_file.write_text("hello\n")
    preprocessor = TextPreprocessor()
    preprocessor.build_vocab(["hello"])

    dataset = load_encoded_dataset(
        preprocessor, str(text_file), seq_len=5, lazy=False
    )
    assert len(dataset) == 1
