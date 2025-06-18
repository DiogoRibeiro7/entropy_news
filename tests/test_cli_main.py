import pytest

pytest.importorskip("torch")

from entropy_news.main import build_parser as build_train_parser
from entropy_news.main_forecast import build_parser as build_forecast_parser


def test_build_train_parser_parses_args() -> None:
    """Check that the training CLI parser handles custom arguments."""

    parser = build_train_parser()
    args = parser.parse_args([
        "--train-data",
        "foo.txt",
        "--glove-path",
        "bar.txt",
        "--epochs",
        "10",
        "--batch-size",
        "16",
    ])
    assert args.train_data == "foo.txt"
    assert args.glove_path == "bar.txt"
    assert args.epochs == 10
    assert args.batch_size == 16


def test_build_forecast_parser_defaults() -> None:
    """Verify that forecast parser returns expected defaults."""

    parser = build_forecast_parser()
    args = parser.parse_args([])
    assert args.seq_len == 100
    assert args.batch_size == 1
