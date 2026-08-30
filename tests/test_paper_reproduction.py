import math

import pytest

torch = pytest.importorskip("torch")

from entropy_news.data import TextPreprocessor
from entropy_news.model import EntropyLSTM
from entropy_news.paper_reproduction import (
    PaperProtocol,
    chunk_articles_for_training,
    decompose_entropy,
    exponentially_sample_history,
    filter_articles,
    monthly_article_entropy,
)
from entropy_news.paper_rolling import run_paper_reproduction


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


def test_article_filter_uses_cleaned_token_count() -> None:
    pre = TextPreprocessor(vocab_size=20)
    texts = ["One, two!", "One, two, three.", "four five six seven"]
    assert filter_articles(texts, pre, min_article_words=3) == texts[1:]


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
    assert len(first) == 16 + 8 + 4 + 2 + 1
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


def test_paper_runner_produces_year_over_year_identity(tmp_path) -> None:
    months = []
    year, month = 2022, 1
    for _ in range(19):
        label = f"{year:04d}-{month:02d}"
        months.append(label)
        (tmp_path / f"news_{label}.txt").write_text(
            f"market news changes in {label} today\n"
            f"investors read another story in {label}\n"
        )
        month += 1
        if month == 13:
            month = 1
            year += 1

    protocol = PaperProtocol(
        sequence_length=3,
        batch_size=2,
        epochs=1,
        embedding_dim=4,
        hidden_dim=2,
        vocabulary_size=30,
        min_article_words=2,
        sampling_seed=11,
    )
    output_dir = tmp_path / "out"
    results = run_paper_reproduction(
        months,
        str(tmp_path),
        protocol=protocol,
        learning_rate=0.01,
        output_dir=str(output_dir),
        show_progress=False,
    )

    assert len(results) == 1
    result = results[0]
    assert result.month == months[-1]
    assert math.isclose(result.ENT, result.ENT_NEWS + result.ENT_MODEL, rel_tol=1e-7)
    assert (output_dir / "paper_entropy_results.csv").exists()
    assert (output_dir / "paper_protocol.json").exists()
