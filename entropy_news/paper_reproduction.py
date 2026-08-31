"""Paper-faithful helpers for Glasserman, Mamaysky and Qin (2023)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import hashlib
import math
import random

import torch

from entropy_news.data import NewsDataset, TextPreprocessor
from entropy_news.model import EntropyLSTM
from entropy_news.model.paper_lstm import PaperEntropyLSTM
from entropy_news.utils import get_device


@dataclass(frozen=True)
class PaperProtocol:
    """Defaults reported in the paper's language-model specification."""

    sequence_length: int = 100
    history_months: int = 6
    year_lag: int = 12
    batch_size: int = 128
    epochs: int = 50
    embedding_dim: int = 100
    hidden_dim: int = 16
    vocabulary_size: int = 10_000
    min_article_words: int = 30
    sampling_seed: int = 1729


@dataclass(frozen=True)
class EntropyComponents:
    current_entropy: float
    year_ago_entropy: float
    year_ago_model_on_current: float
    ENT: float
    ENT_NEWS: float
    ENT_MODEL: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def decompose_entropy(
    current_entropy: float,
    year_ago_entropy: float,
    year_ago_model_on_current: float,
) -> EntropyComponents:
    ent = current_entropy - year_ago_entropy
    ent_news = year_ago_model_on_current - year_ago_entropy
    ent_model = current_entropy - year_ago_model_on_current
    if not math.isclose(ent, ent_news + ent_model, rel_tol=1e-12, abs_tol=1e-12):
        raise ArithmeticError("entropy decomposition identity failed")
    return EntropyComponents(
        current_entropy=current_entropy,
        year_ago_entropy=year_ago_entropy,
        year_ago_model_on_current=year_ago_model_on_current,
        ENT=ent,
        ENT_NEWS=ent_news,
        ENT_MODEL=ent_model,
    )


def filter_articles(
    texts: Sequence[str],
    preprocessor: TextPreprocessor,
    min_article_words: int,
) -> list[str]:
    if min_article_words < 2:
        raise ValueError("min_article_words must be at least 2")
    retained: list[str] = []
    for text in texts:
        cleaned = preprocessor.clean_text(text)
        if len(preprocessor.tokenize(cleaned)) >= min_article_words:
            retained.append(text)
    return retained


def chunk_articles_for_training(
    encoded_articles: Sequence[Sequence[int]], sequence_length: int = 100
) -> list[list[int]]:
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    chunks: list[list[int]] = []
    for article in encoded_articles:
        tokens = list(article)
        if len(tokens) < 2:
            continue
        for start in range(0, len(tokens) - 1, sequence_length):
            chunk = tokens[start : start + sequence_length + 1]
            if len(chunk) >= 2:
                chunks.append(chunk)
    return chunks


def make_training_dataset(
    texts: Sequence[str],
    preprocessor: TextPreprocessor,
    sequence_length: int = 100,
    *,
    min_article_words: int = 2,
    in_memory: bool = True,
) -> NewsDataset:
    retained = filter_articles(texts, preprocessor, min_article_words)
    encoded = [preprocessor.encode(text) for text in retained]
    chunks = chunk_articles_for_training(encoded, sequence_length)
    return NewsDataset(chunks, seq_len=sequence_length, in_memory=in_memory)


def _score_article(
    model: EntropyLSTM | PaperEntropyLSTM,
    token_ids: Sequence[int],
    *,
    sequence_length: int,
    device: torch.device,
) -> float:
    """Return mean negative log probability over all article words."""
    tokens = list(token_ids)
    if not tokens:
        return float("nan")

    total_log_prob = 0.0
    total_tokens = 0
    model.eval()
    with torch.no_grad():
        # Equation (1) in the paper treats the first factor as the unconditional
        # marginal P(w_1).  With the recurrent state initialised at h_0 = 0,
        # this is softmax(U_s h_0 + b_s) = softmax(b_s): no synthetic input is
        # consumed before the first observed word is scored.
        initial_hidden = torch.zeros(
            (1, 1, model.fc.in_features),
            dtype=model.fc.weight.dtype,
            device=device,
        )
        first_logits = model.fc(initial_hidden)
        first_log_probs = torch.log_softmax(first_logits, dim=-1)
        total_log_prob += first_log_probs[0, 0, tokens[0]].item()
        total_tokens = 1
        state = None

        inputs = tokens[:-1]
        targets = tokens[1:]
        for start in range(0, len(inputs), sequence_length):
            x = torch.tensor(
                [inputs[start : start + sequence_length]],
                dtype=torch.long,
                device=device,
            )
            y = torch.tensor(
                [targets[start : start + sequence_length]],
                dtype=torch.long,
                device=device,
            )
            logits, state = model.forward_with_state(x, state)
            log_probs = torch.log_softmax(logits, dim=-1)
            chosen = log_probs.gather(2, y.unsqueeze(-1)).squeeze(-1)
            total_log_prob += chosen.sum().item()
            total_tokens += int(y.numel())

    return -total_log_prob / total_tokens


def monthly_article_entropy(
    model: EntropyLSTM | PaperEntropyLSTM,
    texts: Sequence[str],
    preprocessor: TextPreprocessor,
    *,
    sequence_length: int = 100,
    min_article_words: int = 30,
    device: torch.device | None = None,
) -> float:
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    retained = filter_articles(texts, preprocessor, min_article_words)
    resolved_device = device or get_device()
    model = model.to(resolved_device)
    scores: list[float] = []
    for text in retained:
        score = _score_article(
            model,
            preprocessor.encode(text),
            sequence_length=sequence_length,
            device=resolved_device,
        )
        if math.isfinite(score):
            scores.append(score)
    if not scores:
        return float("inf")
    return sum(scores) / len(scores)


def _stable_month_seed(base_seed: int, month: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{month}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def exponentially_sample_history(
    month_to_articles: Mapping[str, Sequence[str]],
    prior_months_newest_first: Sequence[str],
    *,
    base_seed: int = 1729,
    evaluation_month: str,
) -> list[str]:
    if len(prior_months_newest_first) != 6:
        raise ValueError("paper update requires exactly six prior months")
    rng = random.Random(_stable_month_seed(base_seed, evaluation_month))
    selected: list[str] = []
    for offset, month in enumerate(prior_months_newest_first):
        articles = list(month_to_articles.get(month, ()))
        if not articles:
            continue
        if offset == 0:
            selected.extend(articles)
            continue
        sample_size = int(len(articles) * (0.5**offset))
        if sample_size > 0:
            selected.extend(rng.sample(articles, sample_size))
    return selected
