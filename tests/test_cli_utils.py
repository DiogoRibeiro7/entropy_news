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


def test_load_model_and_vocab_uses_map_location(tmp_path, monkeypatch) -> None:
    """``torch.load`` should be called with a ``map_location`` argument."""

    torch = pytest.importorskip("torch")
    from entropy_news.utils.cli import load_model_and_vocab
    from entropy_news.model import EntropyLSTM

    vocab_file = tmp_path / "vocab.json"
    _write_vocab(vocab_file)

    # Create and save a dummy model
    model = EntropyLSTM(vocab_size=2, embed_dim=4, hidden_dim=4)
    model_path = tmp_path / "model.pth"
    torch.save(model.state_dict(), model_path)

    called = {}
    orig_load = torch.load

    def fake_load(path, map_location=None):  # type: ignore[override]
        called["map_location"] = map_location
        return orig_load(path, map_location=map_location)

    monkeypatch.setattr(torch, "load", fake_load)

    device = torch.device("cpu")
    _preprocessor, _model, used_device = load_model_and_vocab(
        str(vocab_file),
        str(model_path),
        embed_dim=4,
        hidden_dim=4,
        num_layers=1,
        dropout=0.0,
        device=device,
    )

    assert called["map_location"] == device
    assert used_device == device


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
