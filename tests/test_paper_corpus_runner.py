import json

import pytest

pytest.importorskip("torch")

from entropy_news.paper_reproduction import PaperProtocol
from entropy_news.paper_rolling import (
    _resolve_corpus_contract,
    _sha256,
    _write_provenance_manifest,
)


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


def _write_contract(path, *, mode: str = "methodological_reproduction") -> None:
    path.write_text(
        json.dumps(
            {
                "mode": mode,
                "source_name": (
                    "Reuters" if mode == "empirical_reuters_replication" else "licensed-alternative-feed"
                ),
                "source_period_start": "2022-01",
                "source_period_end": "2023-07",
                "first_rewrite_only": mode == "empirical_reuters_replication",
                "language": "English",
                "company_universe": (
                    "S&P 500" if mode == "empirical_reuters_replication" else "custom"
                ),
                "headline_exclusions_applied": mode == "empirical_reuters_replication",
                "min_article_words": 30,
                "notes": "External corpus declaration.",
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def test_runner_defaults_to_methodological_reproduction() -> None:
    contract, path = _resolve_corpus_contract(_months(), PaperProtocol(), None)
    assert path is None
    assert contract.mode == "methodological_reproduction"
    assert contract.source_name == "user_supplied"
    assert contract.notes.startswith("No external corpus manifest supplied")


def test_runner_rejects_bad_empirical_manifest_before_data_loading(tmp_path) -> None:
    path = tmp_path / "bad-corpus.json"
    _write_contract(path, mode="empirical_reuters_replication")
    data = json.loads(path.read_text(encoding="utf-8"))
    data["source_name"] = "Not Reuters"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="source_name=Reuters"):
        _resolve_corpus_contract(_months(), PaperProtocol(), str(path))


def test_runner_binds_valid_external_manifest_bytes(tmp_path) -> None:
    path = tmp_path / "corpus.json"
    _write_contract(path)
    contract, resolved = _resolve_corpus_contract(
        _months(), PaperProtocol(), str(path)
    )
    assert resolved == path
    assert contract.mode == "methodological_reproduction"
    assert len(_sha256(resolved)) == 64


def test_provenance_records_external_corpus_contract_and_hash(tmp_path) -> None:
    months = _months()
    for month in months:
        (tmp_path / f"news_{month}.txt").write_text("article\n", encoding="utf-8")
    glove = tmp_path / "glove.txt"
    glove.write_text("word 0.1 0.2\n", encoding="utf-8")
    out = tmp_path / "out"
    out.mkdir()
    (out / "paper_entropy_results.csv").write_text("month,ENT\n", encoding="utf-8")
    (out / "paper_protocol.json").write_text("{}\n", encoding="utf-8")
    (out / "paper_vocabulary.json").write_text("{}\n", encoding="utf-8")

    corpus_path = tmp_path / "corpus.json"
    _write_contract(corpus_path, mode="empirical_reuters_replication")
    contract, resolved = _resolve_corpus_contract(
        months, PaperProtocol(), str(corpus_path)
    )

    _write_provenance_manifest(
        out,
        months,
        str(tmp_path),
        str(glove),
        PaperProtocol(),
        0.001,
        {month: 1 for month in months},
        {month: 1 for month in months},
        10_000,
        10_000,
        177_488,
        contract,
        resolved,
    )

    manifest = json.loads((out / "paper_run_manifest.json").read_text())
    assert manifest["manifest_version"] == 4
    assert manifest["corpus"]["classification"] == "empirical_reuters_replication"
    assert manifest["corpus"]["contract"]["source_name"] == "Reuters"
    assert manifest["corpus"]["external_manifest_supplied"] is True
    assert manifest["corpus"]["provenance_certified_by_software"] is False
    assert manifest["corpus"]["manifest"]["sha256"] == _sha256(corpus_path)
