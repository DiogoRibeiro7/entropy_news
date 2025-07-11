import pytest
pd = pytest.importorskip("pandas")

from entropy_news.rolling_train_forecast import main as rolling_main


def test_rolling_main_runs(tmp_path, monkeypatch) -> None:
    """Verify the rolling CLI executes with patched helpers."""

    torch = pytest.importorskip("torch")

    # Create dummy monthly files
    (tmp_path / "news_2023-01.txt").write_text("a b\n")
    (tmp_path / "news_2023-02.txt").write_text("b c\n")

    # Dummy implementations to skip heavy work
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
        return {"ENT": 0.0, "ENT_news": 0.0, "ENT_model": 0.0}

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

    out_csv = tmp_path / "rolling_forecast_results.csv"
    assert out_csv.exists()
    df = pd.read_csv(out_csv)
    assert {"ENT", "ENT_news", "ENT_model", "month"}.issubset(df.columns)
