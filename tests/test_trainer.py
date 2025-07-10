import math
import pytest

torch = pytest.importorskip("torch")
from entropy_news.data.dataset import NewsDataset
from entropy_news.model.lstm_entropy import EntropyLSTM
from entropy_news.model.trainer import Trainer


def _zero_lstm(vocab_size: int) -> EntropyLSTM:
    model = EntropyLSTM(vocab_size=vocab_size, embed_dim=4, hidden_dim=2)
    for param in model.parameters():
        param.data.fill_(0.0)
    return model


def test_evaluate_returns_log_vocab_size() -> None:
    dataset = NewsDataset([[1, 2, 3]], seq_len=2)
    model = _zero_lstm(vocab_size=5)
    trainer = Trainer(model)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1)

    loss = trainer.evaluate(loader)
    assert math.isclose(loss, math.log(5), rel_tol=1e-5)


def test_train_updates_weights() -> None:
    """Training should modify model parameters."""

    dataset = NewsDataset([[1, 2, 3, 4]], seq_len=3)
    model = _zero_lstm(vocab_size=6)
    trainer = Trainer(model)

    initial_weight = model.embed.weight.clone()
    trainer.train(dataset, epochs=1, batch_size=1)

    assert not torch.allclose(model.embed.weight, initial_weight)
