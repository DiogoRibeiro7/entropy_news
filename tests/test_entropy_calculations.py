import math
import pytest

torch = pytest.importorskip("torch")

from entropy_news.data.dataset import NewsDataset
from entropy_news.evaluation.entropy_calculator import EntropyCalculator
from entropy_news.evaluation.news_model_update import NewsModelUpdateCalculator


class ConstantModel(torch.nn.Module):
    def __init__(self, vocab_size: int, logit: float = 0.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.logit = logit
        self.device = torch.device("cpu")

    def forward(self, x):
        batch, seq_len = x.shape
        return torch.full((batch, seq_len, self.vocab_size), self.logit)


class EchoModel(torch.nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.device = torch.device("cpu")

    def forward(self, x):
        batch, seq_len = x.shape
        logits = torch.zeros(batch, seq_len, self.vocab_size)
        for i in range(batch):
            for j in range(seq_len):
                logits[i, j, x[i, j]] = 1e3
        return logits


def test_compute_entropy_uniform():
    dataset = NewsDataset([[1, 2, 3]], seq_len=2)
    ent = EntropyCalculator(ConstantModel(vocab_size=10)).compute_entropy(dataset)
    assert math.isclose(ent, math.log(10), rel_tol=1e-5)


def test_compute_entropy_handles_singleton_batch() -> None:
    dataset = NewsDataset([[1, 2, 3]] * 3, seq_len=2)
    ent = EntropyCalculator(ConstantModel(vocab_size=5)).compute_entropy(
        dataset, batch_size=2
    )
    assert math.isclose(ent, math.log(5), rel_tol=1e-5)


def test_news_model_update_uses_nonpaper_names():
    dataset = NewsDataset([[1, 1, 1]], seq_len=2)
    result = NewsModelUpdateCalculator(
        ConstantModel(vocab_size=5), EchoModel(vocab_size=5)
    ).compute_entropies(dataset)
    assert set(result) == {
        "baseline_entropy",
        "updated_entropy",
        "model_update_delta",
    }
    assert math.isclose(result["baseline_entropy"], math.log(5), rel_tol=1e-5)
    assert result["updated_entropy"] < 1e-6
    assert math.isclose(
        result["model_update_delta"],
        result["updated_entropy"] - result["baseline_entropy"],
        rel_tol=1e-6,
    )


def test_compute_perplexity_uniform() -> None:
    dataset = NewsDataset([[1, 2, 3]], seq_len=2)
    ppl = EntropyCalculator(ConstantModel(vocab_size=4)).compute_perplexity(dataset)
    assert math.isclose(ppl, 4.0, rel_tol=1e-5)


def test_entropy_empty_dataset_returns_inf() -> None:
    dataset = NewsDataset([], seq_len=2)
    calc = EntropyCalculator(ConstantModel(vocab_size=5))
    assert math.isinf(calc.compute_entropy(dataset))
    assert math.isinf(calc.compute_perplexity(dataset))
