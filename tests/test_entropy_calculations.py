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
        logits = torch.full((batch, seq_len, self.vocab_size), self.logit)
        return logits

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
    model = ConstantModel(vocab_size=10)
    calc = EntropyCalculator(model)
    ent = calc.compute_entropy(dataset, batch_size=1)
    assert math.isclose(ent, math.log(10), rel_tol=1e-5)


def test_compute_entropy_handles_singleton_batch() -> None:
    """Ensure squeezing doesn't drop the batch dimension when size is 1."""

    dataset = NewsDataset([[1, 2, 3], [1, 2, 3], [1, 2, 3]], seq_len=2)
    model = ConstantModel(vocab_size=5)
    calc = EntropyCalculator(model)
    ent = calc.compute_entropy(dataset, batch_size=2)
    assert math.isclose(ent, math.log(5), rel_tol=1e-5)

def test_news_model_update_keys_and_values():
    dataset = NewsDataset([[1, 1, 1]], seq_len=2)
    old_model = ConstantModel(vocab_size=5)
    new_model = EchoModel(vocab_size=5)
    calc = NewsModelUpdateCalculator(old_model, new_model)
    result = calc.compute_entropies(dataset, batch_size=1)
    assert set(result.keys()) == {"ENT", "ENT_news", "ENT_model"}
    assert math.isclose(result["ENT_news"], math.log(5), rel_tol=1e-5)
    assert result["ENT"] < 1e-6
    assert math.isclose(result["ENT_model"], result["ENT"] - result["ENT_news"], rel_tol=1e-6)


def test_compute_perplexity_uniform() -> None:
    """Perplexity of a uniform model equals its vocab size."""

    dataset = NewsDataset([[1, 2, 3]], seq_len=2)
    model = ConstantModel(vocab_size=4)
    calc = EntropyCalculator(model)

    ppl = calc.compute_perplexity(dataset, batch_size=1)
    assert math.isclose(ppl, 4.0, rel_tol=1e-5)


def test_entropy_empty_dataset_returns_inf() -> None:
    """Entropy and perplexity are infinite on an empty dataset."""

    dataset = NewsDataset([], seq_len=2)
    model = ConstantModel(vocab_size=5)
    calc = EntropyCalculator(model)

    ent = calc.compute_entropy(dataset)
    ppl = calc.compute_perplexity(dataset)

    assert math.isinf(ent)
    assert math.isinf(ppl)
