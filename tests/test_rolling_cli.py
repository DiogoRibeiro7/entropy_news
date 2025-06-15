import argparse
pd = __import__("pytest").importorskip("pandas")
from entropy_news.rolling_train_forecast import build_parser


def test_build_parser_parses_args() -> None:
    parser = build_parser()
    args = parser.parse_args([
        "2023-01",
        "2023-02",
        "--base-data-dir",
        "foo",
        "--output-dir",
        "bar",
        "--seq-len",
        "80",
        "--train-window-size",
        "3",
    ])
    assert args.months == ["2023-01", "2023-02"]
    assert args.base_data_dir == "foo"
    assert args.output_dir == "bar"
    assert args.seq_len == 80
    assert args.train_window_size == 3


def test_build_parser_defaults() -> None:
    parser = build_parser()
    args = parser.parse_args(["2023-03"])
    assert args.months == ["2023-03"]
    assert args.base_data_dir == "data/"
    assert args.output_dir == "output/"
    assert args.seq_len == 100
    assert args.train_window_size == 6
