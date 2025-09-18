import json
from pathlib import Path

import pytest

from entropy_news.main import main as train_main
from entropy_news.main_evaluate import main as eval_main
from entropy_news.main_forecast import main as forecast_main


def _write_vocab(path: Path) -> None:
    data = {"<PAD>": 0, "hello": 1}
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"vocab_size": len(data), "vocab": data}, f)




def test_train_main_missing_glove(tmp_path, caplog) -> None:
    """Training CLI should gracefully handle missing GloVe files."""

    torch = pytest.importorskip("torch")
    data_file = tmp_path / "data.txt"
    data_file.write_text("hello\n")
    with caplog.at_level("ERROR"):
        with pytest.raises(SystemExit):
            train_main([
                "--train-data",
                str(data_file),
                "--glove-path",
                str(tmp_path / "missing_glove.txt"),
                "--epochs",
                "1",
            ])
    assert any("GloVe embeddings missing" in rec.message for rec in caplog.records)


def test_train_main_missing_data(tmp_path, caplog) -> None:
    """Training CLI should fail with a helpful message when data is missing."""

    torch = pytest.importorskip("torch")
    with caplog.at_level("ERROR"):
        with pytest.raises(SystemExit):
            train_main(["--train-data", str(tmp_path / "missing.txt")])
    assert any("not found" in rec.message for rec in caplog.records)


def test_eval_main_missing_model(tmp_path, caplog) -> None:
    """Evaluation CLI should error if the model file is absent."""

    torch = pytest.importorskip("torch")
    data_file = tmp_path / "data.txt"
    data_file.write_text("hello\n")
    vocab_file = tmp_path / "vocab.json"
    _write_vocab(vocab_file)

    with caplog.at_level("ERROR"):
        with pytest.raises(SystemExit):
            eval_main([
                "--data",
                str(data_file),
                "--vocab-path",
                str(vocab_file),
                "--model-path",
                str(tmp_path / "missing.pth"),
            ])
    assert any("Model file not found" in r.message for r in caplog.records)


def test_eval_main_missing_data(tmp_path, caplog) -> None:
    """Evaluation CLI should report missing data files."""

    torch = pytest.importorskip("torch")
    vocab_file = tmp_path / "vocab.json"
    _write_vocab(vocab_file)
    model_file = tmp_path / "model.pth"
    model_file.write_bytes(b"0")

    with caplog.at_level("ERROR"):
        with pytest.raises(SystemExit):
            eval_main([
                "--data",
                str(tmp_path / "missing.txt"),
                "--vocab-path",
                str(vocab_file),
                "--model-path",
                str(model_file),
            ])
    assert any("not found" in r.message for r in caplog.records)


def test_forecast_main_missing_model(tmp_path, caplog) -> None:
    """Forecast CLI should error if the model file is absent."""

    torch = pytest.importorskip("torch")
    new_file = tmp_path / "new.txt"
    new_file.write_text("hello\n")
    vocab_file = tmp_path / "vocab.json"
    _write_vocab(vocab_file)

    with caplog.at_level("ERROR"):
        with pytest.raises(SystemExit):
            forecast_main([
                "--new-data",
                str(new_file),
                "--output-csv",
                str(tmp_path / "out.csv"),
                "--vocab-path",
                str(vocab_file),
                "--model-path",
                str(tmp_path / "missing.pth"),
            ])
    assert any("Model file not found" in r.message for r in caplog.records)


def test_forecast_main_missing_data(tmp_path, caplog) -> None:
    """Forecast CLI should report missing data files."""

    torch = pytest.importorskip("torch")
    vocab_file = tmp_path / "vocab.json"
    _write_vocab(vocab_file)
    model_file = tmp_path / "model.pth"
    model_file.write_bytes(b"0")

    with caplog.at_level("ERROR"):
        with pytest.raises(SystemExit):
            forecast_main([
                "--new-data",
                str(tmp_path / "missing.txt"),
                "--output-csv",
                str(tmp_path / "out.csv"),
                "--vocab-path",
                str(vocab_file),
                "--model-path",
                str(model_file),
            ])
    assert any("not found" in r.message for r in caplog.records)

