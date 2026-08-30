import hashlib
import json
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from entropy_news.data import TextPreprocessor
from entropy_news.model import EntropyLSTM
from entropy_news.model.paper_lstm import PaperEntropyLSTM
from entropy_news.paper_architecture import PAPER_TARGET_IGNORE_INDEX, PaperNewsDataset
from entropy_news.paper_reproduction import (
    PaperProtocol,
    _score_article,
    chunk_articles_for_training,
    decompose_entropy,
    exponentially_sample_history,
    filter_articles,
    monthly_article_entropy,
)
from entropy_news.paper_rolling import run_paper_reproduction


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _months(count: int = 19) -> list[str]:
    months = []
    year, month = 2022, 1
    for _ in range(count):
        months.append(f"{year:04d}-{month:02d}")
        month += 1
        if month == 13:
            month = 1
            year += 1
    return months


class _EmbeddingStub:
    embedding_dim = 1
    weight = torch.zeros(1)


class _FirstTokenTestModel:
    embedding = _EmbeddingStub()

    def eval(self):
        return self

    def lstm(self, x, state):
        output = torch.zeros((1, 1, 1), dtype=torch.float32)
        next_state = (torch.zeros((1, 1, 1)), torch.zeros((1, 1, 1)))
        return output, next_state

    def fc(self, output):
        return torch.tensor([[[0.0, 2.0, 0.0]]], dtype=torch.float32)

    def forward_with_state(self, x, state):
        logits = torch.zeros((1, x.size(1), 3), dtype=torch.float32)
        return logits, state


def test_equation_15_decomposition_identity() -> None:
    result = decompose_entropy(4.0, 3.5, 3.8)
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


def test_exact_paper_model_has_reported_parameter_count() -> None:
    matrix = np.zeros((10_001, 100), dtype="float32")
    model = PaperEntropyLSTM(
        10_000,
        embed_dim=100,
        hidden_dim=16,
        embedding_matrix=matrix,
        padding_idx=10_000,
    )
    assert model.fc.out_features == 10_000
    assert model.embedding.num_embeddings == 10_001
    assert not model.embedding.weight.requires_grad
    assert model.trainable_parameter_count() == 177_488


def test_paper_dataset_keeps_unk_predictive_and_padding_nonpredictive() -> None:
    dataset = PaperNewsDataset([[0, 2, 3]], seq_len=4, padding_id=10)
    x, y = dataset[0]
    assert x.tolist() == [0, 2, 3, 10]
    assert y.tolist() == [2, 3, PAPER_TARGET_IGNORE_INDEX, PAPER_TARGET_IGNORE_INDEX]
    assert 0 not in y.tolist() or 0 != PAPER_TARGET_IGNORE_INDEX


def test_article_entropy_includes_first_word_probability() -> None:
    model = _FirstTokenTestModel()
    score = _score_article(
        model,
        [1, 2, 2],
        sequence_length=1,
        device=torch.device("cpu"),
    )
    first_nll = -torch.log_softmax(torch.tensor([0.0, 2.0, 0.0]), dim=0)[1].item()
    later_nll = math.log(3.0)
    expected = (first_nll + 2.0 * later_nll) / 3.0
    assert math.isclose(score, expected, rel_tol=1e-7)
    assert not math.isclose(score, later_nll, rel_tol=1e-7)


def test_exponential_history_sampling_is_causal_and_deterministic() -> None:
    months = [f"2023-{m:02d}" for m in range(6, 0, -1)]
    data = {month: [f"{month}-{i}" for i in range(16)] for month in months}
    first = exponentially_sample_history(data, months, base_seed=7, evaluation_month="2023-07")
    second = exponentially_sample_history(data, months, base_seed=7, evaluation_month="2023-07")
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


def test_paper_runner_requires_glove(tmp_path) -> None:
    with pytest.raises(ValueError, match="requires --glove-path"):
        run_paper_reproduction(_months(), str(tmp_path), show_progress=False)


def test_paper_runner_rejects_duplicate_months(tmp_path) -> None:
    months = _months()
    months[5] = months[4]
    glove = tmp_path / "glove.txt"
    glove.write_text("market 0.1 0.2 0.3 0.4\n")
    with pytest.raises(ValueError, match="unique"):
        run_paper_reproduction(months, str(tmp_path), glove_path=str(glove), show_progress=False)


def test_paper_runner_rejects_nonconsecutive_months(tmp_path) -> None:
    months = _months()
    months[-1] = "2023-08"
    glove = tmp_path / "glove.txt"
    glove.write_text("market 0.1 0.2 0.3 0.4\n")
    with pytest.raises(ValueError, match="strictly consecutive"):
        run_paper_reproduction(months, str(tmp_path), glove_path=str(glove), show_progress=False)


def test_paper_runner_rejects_missing_glove_file(tmp_path) -> None:
    missing = tmp_path / "missing-glove.txt"
    with pytest.raises(FileNotFoundError, match="GloVe file not found"):
        run_paper_reproduction(_months(), str(tmp_path), glove_path=str(missing), show_progress=False)


def test_paper_runner_produces_identity_and_provenance(tmp_path, monkeypatch) -> None:
    months = _months()
    for label in months:
        late_word = " futureonly" if label == months[-1] else ""
        (tmp_path / f"news_{label}.txt").write_text(
            f"market news changes in {label} today{late_word}\n"
            f"investors read another story in {label}\n"
        )
    glove = tmp_path / "glove.txt"
    glove.write_text(
        "market 0.1 0.2 0.3 0.4\nnews 0.2 0.1 0.0 0.3\nchanges 0.0 0.1 0.2 0.3\n"
        "in 0.1 0.1 0.1 0.1\ntoday 0.2 0.2 0.2 0.2\ninvestors 0.3 0.2 0.1 0.0\n"
        "read 0.1 0.3 0.2 0.0\nanother 0.2 0.0 0.1 0.3\nstory 0.0 0.2 0.3 0.1\n"
        "futureonly 0.4 0.3 0.2 0.1\n"
    )
    monkeypatch.setenv("GITHUB_SHA", "abc123provenance")
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
        glove_path=str(glove),
        output_dir=str(output_dir),
        show_progress=False,
    )
    assert len(results) == 1
    assert math.isclose(results[0].ENT, results[0].ENT_NEWS + results[0].ENT_MODEL, rel_tol=1e-7)

    result_path = output_dir / "paper_entropy_results.csv"
    protocol_path = output_dir / "paper_protocol.json"
    vocabulary_path = output_dir / "paper_vocabulary.json"
    manifest_path = output_dir / "paper_run_manifest.json"
    vocabulary = json.loads(vocabulary_path.read_text())["vocab"]
    assert "futureonly" in vocabulary

    manifest = json.loads(manifest_path.read_text())
    assert manifest["manifest_version"] == 3
    assert manifest["git_revision"] == "abc123provenance"
    assert manifest["vocabulary"]["scope"] == "whole_requested_corpus"
    assert manifest["vocabulary"]["predictive_entries"] == len(vocabulary) == 30
    assert manifest["architecture"]["predictive_classes"] == 30
    assert manifest["architecture"]["lexical_classes"] == 29
    assert manifest["architecture"]["unk_class"] == 0
    assert manifest["architecture"]["padding_id"] == 30
    assert manifest["architecture"]["padding_is_predictive_class"] is False
    assert manifest["architecture"]["lstm_bias_vectors"] == 1
    assert manifest["inputs"]["glove"]["sha256"] == _sha256(glove)
    assert len(manifest["inputs"]["monthly_news"]) == 19
    assert manifest["outputs"]["paper_entropy_results.csv"]["sha256"] == _sha256(result_path)
    assert manifest["outputs"]["paper_protocol.json"]["sha256"] == _sha256(protocol_path)
    assert manifest["outputs"]["paper_vocabulary.json"]["sha256"] == _sha256(vocabulary_path)
