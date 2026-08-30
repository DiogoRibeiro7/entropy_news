import math

import pytest

torch = pytest.importorskip("torch")

from entropy_news.data import TextPreprocessor
from entropy_news.model import EntropyLSTM
from entropy_news.paper_reproduction import (
    chunk_articles_for_training,
    decompose_entropy,
    exponentially_sample_history,
    monthly_article_entropy,
)


def test_equation_15_decomposition_identity() -> None:
    result = decompose_entropy(
        current_entropy=4.0,
        year_ago_entropy=3.5,
        year_ago_model_on_current=3.8,
    )
    assert math.isclose(result.ENT, 0.5)
    assert math.isclose(result.ENT_NEWS, 0.3)
    assert math.isclose(result.ENT_MODEL, 0.2)
    assert math.isclose(result.ENT, result.ENT_NEWS + result.ENT_MODEL)


def test_training_chunks_cover_complete_article_without_losing_boundaries() -> None:
    chunks = chunk_articles_for_training([list(range(1, 10))], sequence_length=3)
    assert chunks == [[1, 2, 3, 4], [4, 5, 6, 7], [7, 8, 9]]
    targets = [token for chunk in chunks for token in chunk[1:]]
    assert targets == list(range(2, 10))


def test_exponential_history_sampling_is_causal_and_deterministic() -> None:
    months = [f"2023-{m:02d}" for m in range(6, 0, -1)]
    data = {month: [f"{month}-{i}" for i in range(16)] for month in months}
    first = exponentially_sample_history(
        data, months, base_seed=7, evaluation_month="2023-07"
    )
    second = exponentially_sample_history(
        data, months, base_seed=7, evaluation_month="2023-07"
    )
    assert first == second
    expected = 16 + 8 + 4 + 2 + 1 + 0
    assert len(first) == expected
    assert all(item.startswith(tuple(months)) for item in first)


def test_monthly_entropy_is_equal_weighted_across_articles() -> None:
    pre = TextPreprocessor(vocab_size=20)
    texts = ["a b c", "a b c d e f g"]
    pre.build_vocab(texts)
    model = EntropyLSTM(vocab_size=len(pre.vocab), embed_dim=4, hidden_dim=2)
    for parameter in model.parameters():
        torch.nn.init.constant_(parameter, 0.0)

    value = monthly_article_entropy(
        model,
        texts,
        pre,
        sequence_length=2,
        min_article_words=2,
        device=torch.device("cpu"),
    )
    assert math.isclose(value, math.log(len(pre.vocab)), rel_tol=1e-5)
