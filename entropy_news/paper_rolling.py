"""Causal rolling reproduction of the entropy measures in *New News is Bad News*."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import pandas as pd
import torch

from entropy_news.data import TextPreprocessor
from entropy_news.model import ModelConfig, ModelFactory, Trainer
from entropy_news.paper_reproduction import (
    EntropyComponents,
    PaperProtocol,
    decompose_entropy,
    exponentially_sample_history,
    filter_articles,
    make_training_dataset,
    monthly_article_entropy,
)
from entropy_news.rolling_train_forecast import load_texts_for_month
from entropy_news.utils import get_device


@dataclass(frozen=True)
class PaperMonthResult:
    month: str
    current_entropy: float
    year_ago_entropy: float
    year_ago_model_on_current: float
    ENT: float
    ENT_NEWS: float
    ENT_MODEL: float

    @classmethod
    def from_components(
        cls, month: str, values: EntropyComponents
    ) -> "PaperMonthResult":
        return cls(month=month, **values.as_dict())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_revision() -> str:
    github_sha = os.environ.get("GITHUB_SHA")
    if github_sha:
        return github_sha
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _validate_consecutive_months(months: Sequence[str]) -> None:
    if len(months) != len(set(months)):
        raise ValueError("months must be unique and chronologically ordered")
    periods: list[pd.Period] = []
    for month in months:
        try:
            period = pd.Period(month, freq="M")
        except ValueError as exc:
            raise ValueError(f"invalid month {month!r}; expected YYYY-MM") from exc
        if str(period) != month:
            raise ValueError(f"invalid month {month!r}; expected zero-padded YYYY-MM")
        periods.append(period)
    for previous, current in zip(periods, periods[1:]):
        if current != previous + 1:
            raise ValueError(
                "months must be strictly consecutive; "
                f"expected {previous + 1} after {previous}, got {current}"
            )


def _paper_config(protocol: PaperProtocol, vocab_size: int) -> ModelConfig:
    return ModelConfig(
        architecture="lstm",
        vocab_size=vocab_size,
        embed_dim=protocol.embedding_dim,
        hidden_dim=protocol.hidden_dim,
        num_heads=1,
        ff_dim=128,
        num_layers=1,
        dropout=0.0,
    )


def _fit_initial_model(
    texts: Sequence[str],
    preprocessor: TextPreprocessor,
    protocol: PaperProtocol,
    *,
    learning_rate: float,
    device: torch.device,
    show_progress: bool,
):
    dataset = make_training_dataset(
        texts,
        preprocessor,
        protocol.sequence_length,
        min_article_words=protocol.min_article_words,
    )
    if len(dataset) == 0:
        raise ValueError("initial six-month window has no trainable article chunks")
    config = _paper_config(protocol, len(preprocessor.vocab))
    model = ModelFactory.create(
        config,
        embedding_matrix=preprocessor.embedding_matrix,
    ).to(device)
    Trainer(model, learning_rate=learning_rate, device=device).train(
        dataset,
        epochs=protocol.epochs,
        batch_size=protocol.batch_size,
        show_progress=show_progress,
    )
    return model, config


def _updated_model(
    previous_model,
    update_texts: Sequence[str],
    preprocessor: TextPreprocessor,
    config: ModelConfig,
    protocol: PaperProtocol,
    *,
    learning_rate: float,
    device: torch.device,
    show_progress: bool,
):
    model = ModelFactory.create(
        replace(config, vocab_size=len(preprocessor.vocab)),
        embedding_matrix=preprocessor.embedding_matrix,
    ).to(device)
    model.load_state_dict(previous_model.state_dict())
    dataset = make_training_dataset(
        update_texts,
        preprocessor,
        protocol.sequence_length,
        min_article_words=protocol.min_article_words,
    )
    if len(dataset) == 0:
        raise ValueError("monthly update sample has no trainable article chunks")
    Trainer(model, learning_rate=learning_rate, device=device).fine_tune(
        dataset,
        epochs=protocol.epochs,
        batch_size=protocol.batch_size,
        show_progress=show_progress,
    )
    return model


def _write_provenance_manifest(
    out: Path,
    months: Sequence[str],
    base_data_dir: str,
    glove_path: str,
    protocol: PaperProtocol,
    learning_rate: float,
    raw_counts: dict[str, int],
    filtered_counts: dict[str, int],
    vocabulary_entries: int,
) -> None:
    base = Path(base_data_dir)
    glove = Path(glove_path)
    monthly_inputs = []
    for month in months:
        path = base / f"news_{month}.txt"
        monthly_inputs.append(
            {
                "month": month,
                "file": path.name,
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
                "raw_articles": raw_counts[month],
                "qualifying_articles": filtered_counts[month],
            }
        )

    result_path = out / "paper_entropy_results.csv"
    protocol_path = out / "paper_protocol.json"
    vocabulary_path = out / "paper_vocabulary.json"
    manifest = {
        "manifest_version": 2,
        "git_revision": _git_revision(),
        "execution": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": torch.__version__,
            "pandas": pd.__version__,
            "learning_rate": learning_rate,
        },
        "protocol": asdict(protocol),
        "months": list(months),
        "vocabulary": {
            "scope": "whole_requested_corpus",
            "configured_top_words": protocol.vocabulary_size,
            "actual_entries_including_reserved_tokens": vocabulary_entries,
        },
        "inputs": {
            "monthly_news": monthly_inputs,
            "glove": {
                "file": glove.name,
                "sha256": _sha256(glove),
                "bytes": glove.stat().st_size,
            },
        },
        "outputs": {
            "paper_entropy_results.csv": {
                "sha256": _sha256(result_path),
                "bytes": result_path.stat().st_size,
            },
            "paper_protocol.json": {
                "sha256": _sha256(protocol_path),
                "bytes": protocol_path.stat().st_size,
            },
            "paper_vocabulary.json": {
                "sha256": _sha256(vocabulary_path),
                "bytes": vocabulary_path.stat().st_size,
            },
        },
    }
    (out / "paper_run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def run_paper_reproduction(
    months: Sequence[str],
    base_data_dir: str,
    *,
    protocol: PaperProtocol | None = None,
    learning_rate: float = 0.001,
    glove_path: str | None = None,
    output_dir: str | None = None,
    show_progress: bool = True,
) -> list[PaperMonthResult]:
    """Run the rolling design used to construct the paper's ENT series.

    The model used to score month ``t`` is fit only with information through
    ``t-1``. Following the paper literally, the vocabulary is selected from the
    whole requested corpus before rolling model training. Thus vocabulary
    membership can use later corpus information even though model weights do
    not. The paper path also requires GloVe embeddings, strictly consecutive
    monthly inputs, and emits byte-level provenance when outputs are written.
    """

    protocol = protocol or PaperProtocol()
    required_months = protocol.history_months + protocol.year_lag + 1
    if len(months) < required_months:
        raise ValueError(
            f"at least {required_months} ordered months are required for paper ENT"
        )
    _validate_consecutive_months(months)
    if glove_path is None:
        raise ValueError("paper reproduction requires --glove-path")
    glove = Path(glove_path)
    if not glove.is_file():
        raise FileNotFoundError(f"GloVe file not found: {glove_path}")

    preprocessor = TextPreprocessor(vocab_size=protocol.vocabulary_size)
    raw_month_articles = {
        month: load_texts_for_month(month, base_data_dir) for month in months
    }
    month_articles = {
        month: filter_articles(
            articles,
            preprocessor,
            protocol.min_article_words,
        )
        for month, articles in raw_month_articles.items()
    }
    empty_months = [month for month, articles in month_articles.items() if not articles]
    if empty_months:
        listed = ", ".join(empty_months)
        raise ValueError(
            "paper reproduction requires at least one qualifying article in every "
            f"requested month; empty after filtering: {listed}"
        )

    whole_corpus_texts = [
        text for month in months for text in month_articles[month]
    ]
    preprocessor.build_vocab(whole_corpus_texts)

    initial_months = list(months[: protocol.history_months])
    initial_texts = [
        text for month in initial_months for text in month_articles[month]
    ]
    preprocessor.load_glove_embeddings(
        glove_path,
        embedding_dim=protocol.embedding_dim,
        seed=protocol.sampling_seed,
        show_progress=show_progress,
    )

    device = get_device()
    first_eval_idx = protocol.history_months
    model, config = _fit_initial_model(
        initial_texts,
        preprocessor,
        protocol,
        learning_rate=learning_rate,
        device=device,
        show_progress=show_progress,
    )

    model_states: dict[str, dict[str, torch.Tensor]] = {}
    entropy_at_own_month: dict[str, float] = {}
    results: list[PaperMonthResult] = []

    for eval_idx in range(first_eval_idx, len(months)):
        month = months[eval_idx]
        if eval_idx > first_eval_idx:
            prior = list(
                reversed(months[eval_idx - protocol.history_months : eval_idx])
            )
            update_texts = exponentially_sample_history(
                month_articles,
                prior,
                base_seed=protocol.sampling_seed,
                evaluation_month=month,
            )
            model = _updated_model(
                model,
                update_texts,
                preprocessor,
                config,
                protocol,
                learning_rate=learning_rate,
                device=device,
                show_progress=show_progress,
            )

        model_states[month] = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }

        current_entropy = monthly_article_entropy(
            model,
            month_articles[month],
            preprocessor,
            sequence_length=protocol.sequence_length,
            min_article_words=protocol.min_article_words,
            device=device,
        )
        if not torch.isfinite(torch.tensor(current_entropy)):
            raise ValueError(f"non-finite monthly entropy for {month}")
        entropy_at_own_month[month] = current_entropy

        lag_idx = eval_idx - protocol.year_lag
        if lag_idx < first_eval_idx:
            continue
        lag_month = months[lag_idx]
        lag_state = model_states.get(lag_month)
        if lag_state is None:
            raise RuntimeError(f"missing retained model state for lag month {lag_month}")

        lag_model = ModelFactory.create(
            replace(config, vocab_size=len(preprocessor.vocab)),
            embedding_matrix=preprocessor.embedding_matrix,
        ).to(device)
        lag_model.load_state_dict(lag_state)
        lag_on_current = monthly_article_entropy(
            lag_model,
            month_articles[month],
            preprocessor,
            sequence_length=protocol.sequence_length,
            min_article_words=protocol.min_article_words,
            device=device,
        )
        components = decompose_entropy(
            current_entropy=current_entropy,
            year_ago_entropy=entropy_at_own_month[lag_month],
            year_ago_model_on_current=lag_on_current,
        )
        results.append(PaperMonthResult.from_components(month, components))

    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([asdict(row) for row in results]).to_csv(
            out / "paper_entropy_results.csv", index=False
        )
        (out / "paper_protocol.json").write_text(
            json.dumps(asdict(protocol), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        preprocessor.save_vocab(str(out / "paper_vocabulary.json"))
        _write_provenance_manifest(
            out,
            months,
            base_data_dir,
            glove_path,
            protocol,
            learning_rate,
            {month: len(raw_month_articles[month]) for month in months},
            {month: len(month_articles[month]) for month in months},
            len(preprocessor.vocab),
        )
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the New News is Bad News entropy protocol"
    )
    parser.add_argument("months", nargs="+", help="Consecutive YYYY-MM months")
    parser.add_argument("--base-data-dir", default="data/")
    parser.add_argument("--output-dir", default="output/paper_reproduction")
    parser.add_argument("--glove-path", required=True)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--no-progress", action="store_false", dest="progress")
    parser.set_defaults(progress=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    run_paper_reproduction(
        args.months,
        args.base_data_dir,
        learning_rate=args.learning_rate,
        glove_path=args.glove_path,
        output_dir=args.output_dir,
        show_progress=args.progress,
    )


if __name__ == "__main__":
    main()
