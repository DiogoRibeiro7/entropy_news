import math
import pytest

torch = pytest.importorskip("torch")
from entropy_news.data.dataset import NewsDataset
from entropy_news.evaluation.entropy_calculator import EntropyCalculator
from entropy_news.evaluation.news_model_update import NewsModelUpdateCalculator
from entropy_news.model.lstm_entropy import EntropyLSTM


def _zero_lstm(vocab_size: int) -> EntropyLSTM:
    model = EntropyLSTM(vocab_size=vocab_size, embed_dim=4, hidden_dim=2)
    for param in model.parameters():
        param.data.fill_(0.0)
    return model


def test_compute_entropy_dummy_lstm() -> None:
    dataset = NewsDataset([[1, 2, 3, 4]], seq_len=3)
    model = _zero_lstm(vocab_size=5)
    calc = EntropyCalculator(model)
    ent = calc.compute_entropy(dataset, batch_size=1)
    assert math.isclose(ent, math.log(5), rel_tol=1e-5)


def test_news_model_update_dummy_lstm() -> None:
    dataset = NewsDataset([[1, 1, 1, 1]], seq_len=3)
    old_model = _zero_lstm(vocab_size=5)
    new_model = _zero_lstm(vocab_size=5)
    new_model.fc.bias.data[1] = 1e3
    calc = NewsModelUpdateCalculator(old_model, new_model)
    result = calc.compute_entropies(dataset, batch_size=1)
    assert set(result.keys()) == {"ENT", "ENT_news", "ENT_model"}
    assert math.isclose(result["ENT_news"], math.log(5), rel_tol=1e-5)
    assert result["ENT"] < 1e-3
    assert math.isclose(result["ENT_model"], result["ENT"] - result["ENT_news"], rel_tol=1e-6)
