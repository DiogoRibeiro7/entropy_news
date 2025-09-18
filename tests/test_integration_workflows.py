from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


@pytest.mark.integration
def test_training_and_forecast_pipeline(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("numpy")

    from entropy_news import main as train_cli
    from entropy_news import main_forecast as forecast_cli

    train_data = tmp_path / "train.txt"
    train_data.write_text("Market rallies on positive outlook\nStocks slump despite earnings\n")

    new_data = tmp_path / "new.txt"
    new_data.write_text("Investors celebrate growth\n")

    glove_file = tmp_path / "glove.txt"
    glove_file.write_text(
        "market 0.1 0.2 0.3 0.4\n"
        "rallies 0.0 0.1 0.0 0.1\n"
        "on 0.0 0.0 0.0 0.0\n"
        "positive 0.3 0.3 0.3 0.3\n"
        "outlook 0.2 0.1 0.2 0.1\n"
        "stocks 0.1 0.0 0.1 0.0\n"
        "slump 0.0 0.2 0.1 0.2\n"
        "despite 0.2 0.2 0.1 0.1\n"
        "earnings 0.3 0.3 0.4 0.4\n"
        "investors 0.3 0.2 0.1 0.0\n"
        "celebrate 0.2 0.1 0.3 0.2\n"
        "growth 0.3 0.2 0.3 0.4\n"
    )

    output_dir = tmp_path / "output"
    output_dir.mkdir()
    model_path = output_dir / "model.pth"
    vocab_path = output_dir / "vocab.json"
    config_path = output_dir / "config.json"

    train_cli.main(
        [
            "--train-data",
            str(train_data),
            "--glove-path",
            str(glove_file),
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--embed-dim",
            "4",
            "--hidden-dim",
            "8",
            "--num-layers",
            "1",
            "--learning-rate",
            "0.01",
            "--model-out",
            str(model_path),
            "--vocab-out",
            str(vocab_path),
            "--config-out",
            str(config_path),
            "--vocab-size",
            "20",
            "--no-progress",
        ]
    )

    assert model_path.exists()
    assert vocab_path.exists()
    assert config_path.exists()

    output_csv = output_dir / "forecast.csv"
    forecast_cli.main(
        [
            "--vocab-path",
            str(vocab_path),
            "--model-path",
            str(model_path),
            "--config-path",
            str(config_path),
            "--new-data",
            str(new_data),
            "--output-csv",
            str(output_csv),
            "--embed-dim",
            "4",
            "--hidden-dim",
            "8",
            "--num-layers",
            "1",
            "--fine-tune-epochs",
            "1",
            "--batch-size",
            "1",
            "--no-progress",
        ]
    )

    df = pd.read_csv(output_csv)
    assert {"ENT", "ENT_news", "ENT_model"}.issubset(df.columns)
