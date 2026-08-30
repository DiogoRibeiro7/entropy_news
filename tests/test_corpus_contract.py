import json

import pytest

from entropy_news.corpus_contract import (
    CorpusContract,
    default_methodological_contract,
    validate_corpus_contract,
)


def _reuters_contract(**overrides) -> CorpusContract:
    values = {
        "mode": "empirical_reuters_replication",
        "source_name": "Reuters",
        "source_period_start": "1996-01",
        "source_period_end": "2022-12",
        "first_rewrite_only": True,
        "language": "English",
        "company_universe": "S&P 500",
        "headline_exclusions_applied": True,
        "min_article_words": 30,
        "notes": "Source eligibility declared by the corpus preparer.",
    }
    values.update(overrides)
    return CorpusContract(**values)


def test_empirical_reuters_contract_accepts_paper_compatible_declaration() -> None:
    contract = _reuters_contract()
    validate_corpus_contract(
        contract,
        ["1997-01", "1997-02", "1997-03"],
        protocol_min_article_words=30,
    )


def test_empirical_reuters_contract_rejects_non_reuters_source() -> None:
    with pytest.raises(ValueError, match="source_name=Reuters"):
        validate_corpus_contract(
            _reuters_contract(source_name="Alternative News"),
            ["1997-01"],
            protocol_min_article_words=30,
        )


def test_empirical_reuters_contract_rejects_missing_selection_rules() -> None:
    with pytest.raises(ValueError, match="first rewrite"):
        validate_corpus_contract(
            _reuters_contract(first_rewrite_only=False),
            ["1997-01"],
            protocol_min_article_words=30,
        )
    with pytest.raises(ValueError, match="headline exclusions"):
        validate_corpus_contract(
            _reuters_contract(headline_exclusions_applied=False),
            ["1997-01"],
            protocol_min_article_words=30,
        )


def test_contract_rejects_period_or_filter_mismatch() -> None:
    with pytest.raises(ValueError, match="outside the declared corpus period"):
        validate_corpus_contract(
            _reuters_contract(source_period_end="1996-12"),
            ["1997-01"],
            protocol_min_article_words=30,
        )
    with pytest.raises(ValueError, match="min_article_words"):
        validate_corpus_contract(
            _reuters_contract(min_article_words=25),
            ["1997-01"],
            protocol_min_article_words=30,
        )


def test_no_manifest_defaults_to_methodological_reproduction() -> None:
    contract = default_methodological_contract(["2022-01", "2022-02"], 30)
    assert contract.mode == "methodological_reproduction"
    assert contract.source_name == "user_supplied"
    validate_corpus_contract(
        contract,
        ["2022-01", "2022-02"],
        protocol_min_article_words=30,
    )


def test_contract_round_trips_from_json(tmp_path) -> None:
    path = tmp_path / "corpus.json"
    expected = _reuters_contract()
    path.write_text(json.dumps(expected.as_dict()), encoding="utf-8")
    assert CorpusContract.from_json(path) == expected
