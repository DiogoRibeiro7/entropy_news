import json

import pytest

pytest.importorskip("torch")

from entropy_news.paper_reproduction import PaperProtocol
from entropy_news.paper_rolling import _resolve_corpus_contract, _sha256


def _months(count: int = 19) -> list[str]:
    months: list[str] = []
    year, month = 2022, 1
    for _ in range(count):
        months.append(f"{year:04d}-{month:02d}")
        month += 1
        if month == 13:
            month = 1
            year += 1
    return months


def test_runner_defaults_to_methodological_reproduction() -> None:
    contract, path = _resolve_corpus_contract(_months(), PaperProtocol(), None)
    assert path is None
    assert contract.mode == "methodological_reproduction"
    assert contract.source_name == "user_supplied"
    assert contract.notes.startswith("No external corpus manifest supplied")


def test_runner_rejects_bad_empirical_manifest_before_data_loading(tmp_path) -> None:
    path = tmp_path / "bad-corpus.json"
    path.write_text(
        json.dumps(
            {
                "mode": "empirical_reuters_replication",
                "source_name": "Not Reuters",
                "source_period_start": "2022-01",
                "source_period_end": "2023-07",
                "first_rewrite_only": True,
                "language": "English",
                "company_universe": "S&P 500",
                "headline_exclusions_applied": True,
                "min_article_words": 30,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="source_name=Reuters"):
        _resolve_corpus_contract(_months(), PaperProtocol(), str(path))


def test_runner_binds_valid_external_manifest_bytes(tmp_path) -> None:
    path = tmp_path / "corpus.json"
    path.write_text(
        json.dumps(
            {
                "mode": "methodological_reproduction",
                "source_name": "licensed-alternative-feed",
                "source_period_start": "2022-01",
                "source_period_end": "2023-07",
                "first_rewrite_only": False,
                "language": "English",
                "company_universe": "custom",
                "headline_exclusions_applied": False,
                "min_article_words": 30,
                "notes": "Alternative-data methods reproduction.",
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    contract, resolved = _resolve_corpus_contract(
        _months(), PaperProtocol(), str(path)
    )
    assert resolved == path
    assert contract.mode == "methodological_reproduction"
    assert len(_sha256(resolved)) == 64
