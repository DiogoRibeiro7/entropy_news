"""Tests for the evaluation CLI parser."""

import pytest

from entropy_news.main_evaluate import build_parser


def test_build_parser_parses_args() -> None:
    parser = build_parser()
    args = parser.parse_args([
        "--data",
        "foo.txt",
        "--batch-size",
        "4",
    ])
    assert args.data == "foo.txt"
    assert args.batch_size == 4


def test_build_parser_defaults() -> None:
    parser = build_parser()
    args = parser.parse_args([])
    assert args.data == "data/news_new.txt"
    assert args.batch_size == 1
    assert args.output_csv is None
