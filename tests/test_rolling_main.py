import pytest

pd = pytest.importorskip("pandas")

from entropy_news.rolling_train_forecast import main as rolling_main


def test_rolling_main_runs(tmp_path, monkeypatch) -> None:
    pytest.importorskip("torch")
    (tmp_path / "news_2023-01.txt").write_text("a b\n")
    (tmp_path / "news_2023-02.txt").write_text("b c\n")

    class DummyPreprocessor:
        vocab = {"<PAD>": 0}

    def fake_prepare(months, base_dir, seq_len, vocab_size):
        return object(), DummyPreprocessor()

    class DummyModel:
        device = "cpu"
        def state_dict(self):
            return {}

    def fake_train(*args, **kwargs):
        return DummyModel()

    def fake_update(*args, **kwargs):
        return {
            "baseline_entropy": 1.0,
            "updated_entropy": 0.9,
            "model_update_delta": -0.1,
        }

    monkeypatch.setattr(
        "entropy_news.rolling_train_forecast.prepare_training_set", fake_prepare
    )
    monkeypatch.setattr(
        "entropy_news.rolling_train_forecast.train_model", fake_train
    )
    monkeypatch.setattr(
        "entropy_news.rolling_train_forecast.update_with_new_month", fake_update
    )

    rolling_main([
        "2023-01",
        "2023-02",
        "--base-data-dir",
        str(tmp_path),
        "--output-dir",
        str(tmp_path),
        "--train-window-size",
        "1",
    ])

    df = pd.read_csv(tmp_path / "rolling_forecast_results.csv")
    assert {
        "baseline_entropy",
        "updated_entropy",
        "model_update_delta",
        "month",
    }.issubset(df.columns)
