import pickle
import pytest

from entropy_news.main_evaluate import main as eval_main
from entropy_news.main_forecast import main as forecast_main


def _write_vocab(path):
    data = {"<PAD>": 0, "hello": 1}
    with open(path, "wb") as f:
        pickle.dump(data, f)


def test_eval_main_missing_model(tmp_path, caplog):
    torch = pytest.importorskip("torch")
    data_file = tmp_path / "data.txt"
    data_file.write_text("hello\n")
    vocab_file = tmp_path / "vocab.pkl"
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


def test_forecast_main_missing_model(tmp_path, caplog):
    torch = pytest.importorskip("torch")
    new_file = tmp_path / "new.txt"
    new_file.write_text("hello\n")
    vocab_file = tmp_path / "vocab.pkl"
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

