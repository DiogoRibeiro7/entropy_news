"""Paper-faithful helpers for Glasserman, Mamaysky and Qin (2023).

This module isolates the scientific contract of *New News is Bad News* from the
broader Entropy News platform.  The paper's core quantities are:

A_t = m_[t-6,t-1](t)
B_t = m_[t-18,t-13](t-12)
C_t = m_[t-18,t-13](t)

ENT_t       = A_t - B_t
ENT_NEWS_t  = C_t - B_t
ENT_MODEL_t = A_t - C_t

so ENT_t = ENT_NEWS_t + ENT_MODEL_t by construction.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
import random
from collections.abc import Mapping, Sequence

import torch

from entropy_news.data import NewsDataset, TextPreprocessor
from entropy_news.model import EntropyLSTM
from entropy_news.utils import get_device


@dataclass(frozen=True)
class PaperProtocol:
    """Defaults reported in Section 3.3 of the paper."""

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
    """Equation (5) and Equation (15) quantities for one month."""

    current_entropy: float
    year_ago_entropy: float
    year_ago_model_on_current: float
    ENT: float
    ENT_NEWS: float
    ENT_MODEL: float

    def as_dict(self) -> dict[str, float]:
        """Return a machine-readable representation."""

        return asdict(self)


def decompose_entropy(
    current_entropy: float,
    year_ago_entropy: float,
    year_ago_model_on_current: float,
) -> EntropyComponents:
    """Implement Equations (5) and (15) exactly.

    Args:
        current_entropy: ``m_[t-6,t-1](t)``.
        year_ago_entropy: ``m_[t-18,t-13](t-12)``.
        year_ago_model_on_current: ``m_[t-18,t-13](t)``.
    """

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


def chunk_articles_for_training(
    encoded_articles: Sequence[Sequence[int]], sequence_length: int = 100
) -> list[list[int]]:
    """Use every article token while resetting state between training chunks.

    ``NewsDataset`` consumes ``sequence_length + 1`` tokens to create a shifted
    input/target pair of length ``sequence_length``.  Consecutive chunks overlap
    by one token so no next-token target is lost at a chunk boundary.
    """

    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")

    chunks: list[list[int]] = []
    for article in encoded_articles:
        tokens = list(article)
        if len(tokens) < 2:
            continue
        step = sequence_length
        for start in range(0, len(tokens) - 1, step):
            chunk = tokens[start : start + sequence_length + 1]
            if len(chunk) >= 2:
                chunks.append(chunk)
    return chunks


def make_training_dataset(
    texts: Sequence[str],
    preprocessor: TextPreprocessor,
    sequence_length: int = 100,
    *,
    in_memory: bool = True,
) -> NewsDataset:
    """Encode complete articles and create the paper-style chunk dataset."""

    encoded = [preprocessor.encode(text) for text in texts]
    chunks = chunk_articles_for_training(encoded, sequence_length)
    return NewsDataset(chunks, seq_len=sequence_length, in_memory=in_memory)


def _score_article(
    model: EntropyLSTM,
    token_ids: Sequence[int],
    *,
    sequence_length: int,
    device: torch.device,
) -> float:
    """Return average next-token negative log probability for one article.

    The LSTM state is carried across chunks of the same article and reset for the
    next article, matching the evaluation procedure described after Equation (4).
    """

    tokens = list(token_ids)
    if len(tokens) < 2:
        return float("nan")

    inputs = tokens[:-1]
    targets = tokens[1:]
    state: tuple[torch.Tensor, torch.Tensor] | None = None
    total_log_prob = 0.0
    total_tokens = 0

    model.eval()
    with torch.no_grad():
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
    model: EntropyLSTM,
    texts: Sequence[str],
    preprocessor: TextPreprocessor,
    *,
    sequence_length: int = 100,
    min_article_words: int = 30,
    device: torch.device | None = None,
) -> float:
    """Compute the equal-weighted monthly average of article entropies.

    Each qualifying article contributes exactly one value to the monthly mean;
    longer articles therefore do not receive larger weights.
    """

    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    if min_article_words < 2:
        raise ValueError("min_article_words must be at least 2")

    resolved_device = device or get_device()
    model = model.to(resolved_device)
    scores: list[float] = []
    for text in texts:
        encoded = preprocessor.encode(text)
        if len(encoded) < min_article_words:
            continue
        score = _score_article(
            model,
            encoded,
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
    """Build the paper's monthly retraining sample.

    All articles from ``t-1`` are retained, then fractions ``1/2, 1/4, ...``
    are sampled from ``t-2`` through ``t-6``.  Sampling is deterministic for a
    given base seed and evaluation month so the reproduction is auditable.
    """

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
        fraction = 0.5**offset
        sample_size = int(len(articles) * fraction)
        if sample_size > 0:
            selected.extend(rng.sample(articles, sample_size))
    return selected
