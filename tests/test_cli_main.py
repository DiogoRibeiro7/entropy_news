"""Tests for the training and forecasting CLI parsers."""

import pytest

from entropy_news.main import build_parser as build_train_parser
from entropy_news.main_evaluate import build_parser as build_eval_parser
from entropy_news.main_forecast import build_parser as build_forecast_parser


def test_build_train_parser_parses_args() -> None:
    """Check that the training CLI parser handles custom arguments."""

    parser = build_train_parser()
    args = parser.parse_args([
        "--train-data",
        "foo.txt",
        "--glove-path",
        "bar.txt",
        "--log-file",
        "my.log",
        "--epochs",
        "10",
        "--batch-size",
        "16",
    ])
    assert args.train_data == "foo.txt"
    assert args.glove_path == "bar.txt"
    assert args.epochs == 10
    assert args.batch_size == 16
    assert args.log_file == "my.log"


def test_build_train_parser_defaults() -> None:
    """Ensure training parser provides sensible defaults."""

    parser = build_train_parser()
    args = parser.parse_args([])

    assert args.epochs == 50
    assert args.seq_len == 100
    assert args.dropout is None
    assert args.embed_dim is None
    assert args.architecture is None
    assert args.log_file is None
    assert args.progress is True


def test_build_train_parser_no_progress() -> None:
    """`--no-progress` disables progress bars."""

    parser = build_train_parser()
    args = parser.parse_args(["--no-progress"])
    assert args.progress is False


def test_build_forecast_parser_defaults() -> None:
    """Verify that forecast parser returns expected defaults."""

    parser = build_forecast_parser()
    args = parser.parse_args([])
    assert args.seq_len == 100
    assert args.batch_size == 1
    assert args.log_file is None
    assert args.progress is True
    assert args.allow_unsafe_load is False


def test_build_forecast_parser_no_progress() -> None:
    """Forecast parser should handle `--no-progress`."""

    parser = build_forecast_parser()
    args = parser.parse_args(["--no-progress"])
    assert args.progress is False
    assert args.allow_unsafe_load is False


def test_build_eval_parser_allow_unsafe_flag() -> None:
    """Evaluation parser should expose the unsafe checkpoint toggle."""

    parser = build_eval_parser()
    args = parser.parse_args(["--allow-unsafe-load"])
    assert args.allow_unsafe_load is True
