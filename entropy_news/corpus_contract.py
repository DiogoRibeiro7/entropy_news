"""Machine-readable corpus contract for paper reproduction and replication runs."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Sequence

import pandas as pd

ReplicationMode = Literal["methodological_reproduction", "empirical_reuters_replication"]


@dataclass(frozen=True)
class CorpusContract:
    """Declared provenance and selection contract for an input news corpus.

    The contract is intentionally declarative. Validation checks whether the
    declaration is structurally compatible with the requested run; it cannot
    independently prove that proprietary source files truly satisfy the claim.
    """

    mode: ReplicationMode
    source_name: str
    source_period_start: str
    source_period_end: str
    first_rewrite_only: bool
    language: str
    company_universe: str
    headline_exclusions_applied: bool
    min_article_words: int
    notes: str = ""

    @classmethod
    def from_json(cls, path: str | Path) -> "CorpusContract":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**data)

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _parse_month(value: str, field: str) -> pd.Period:
    try:
        period = pd.Period(value, freq="M")
    except ValueError as exc:
        raise ValueError(f"{field} must use YYYY-MM") from exc
    if str(period) != value:
        raise ValueError(f"{field} must use zero-padded YYYY-MM")
    return period


def validate_corpus_contract(
    contract: CorpusContract,
    months: Sequence[str],
    *,
    protocol_min_article_words: int,
) -> None:
    """Validate a declared corpus contract against one requested paper run."""

    if contract.mode not in {
        "methodological_reproduction",
        "empirical_reuters_replication",
    }:
        raise ValueError(f"unsupported corpus mode: {contract.mode}")
    start = _parse_month(contract.source_period_start, "source_period_start")
    end = _parse_month(contract.source_period_end, "source_period_end")
    if end < start:
        raise ValueError("source_period_end must not precede source_period_start")
    requested = [_parse_month(month, "requested month") for month in months]
    if requested and (requested[0] < start or requested[-1] > end):
        raise ValueError("requested months fall outside the declared corpus period")
    if contract.min_article_words != protocol_min_article_words:
        raise ValueError(
            "corpus min_article_words must match the paper protocol filter"
        )

    if contract.mode == "empirical_reuters_replication":
        if contract.source_name.strip().lower() != "reuters":
            raise ValueError("empirical Reuters replication requires source_name=Reuters")
        if not contract.first_rewrite_only:
            raise ValueError("empirical Reuters replication requires first rewrite only")
        if contract.language.strip().lower() != "english":
            raise ValueError("empirical Reuters replication requires English articles")
        if contract.company_universe.strip().lower() not in {
            "s&p 500",
            "sp500",
            "s&p500",
        }:
            raise ValueError("empirical Reuters replication requires the S&P 500 universe")
        if not contract.headline_exclusions_applied:
            raise ValueError(
                "empirical Reuters replication requires the paper headline exclusions"
            )


def default_methodological_contract(months: Sequence[str], min_article_words: int) -> CorpusContract:
    """Return the explicit classification used when no source manifest is supplied."""

    if not months:
        raise ValueError("at least one month is required for a corpus contract")
    return CorpusContract(
        mode="methodological_reproduction",
        source_name="user_supplied",
        source_period_start=months[0],
        source_period_end=months[-1],
        first_rewrite_only=False,
        language="unspecified",
        company_universe="unspecified",
        headline_exclusions_applied=False,
        min_article_words=min_article_words,
        notes="No external corpus manifest supplied; source-specific claims are not made.",
    )
