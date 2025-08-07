import math
import logging
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


def test_early_stopping_triggers(caplog, monkeypatch) -> None:
    """Trainer should log when early stopping criteria are met."""

    dataset = NewsDataset([[1, 2, 3, 4]] * 3, seq_len=3)
    val_dataset = NewsDataset([[1, 2, 3, 4]] * 3, seq_len=3)
    model = _zero_lstm(vocab_size=5)
    trainer = Trainer(model)

    def const_eval(self, loader):
        return 1.0

    monkeypatch.setattr(Trainer, "evaluate", const_eval)

    with caplog.at_level(logging.INFO):
        trainer.train(
            dataset,
            epochs=5,
            batch_size=1,
            val_dataset=val_dataset,
            early_stopping=True,
            patience=1,
        )

    assert any("Early stopping" in rec.message for rec in caplog.records)


def test_save_checkpoint_creates_dirs(tmp_path) -> None:
    """Trainer.save_checkpoint should make parent directories."""

    model = _zero_lstm(vocab_size=5)
    trainer = Trainer(model)
    chk_path = tmp_path / "nested" / "ckpt.pth"
    trainer.save_checkpoint(chk_path, epoch=1)
    assert chk_path.exists()


def test_load_checkpoint_restores_state(tmp_path) -> None:
    """Loading a checkpoint restores weights and reports the epoch."""

    torch = pytest.importorskip("torch")
    model = _zero_lstm(vocab_size=4)
    trainer = Trainer(model)
    chk = tmp_path / "ckpt.pth"
    trainer.save_checkpoint(chk, epoch=3)

    new_model = _zero_lstm(vocab_size=4)
    new_trainer = Trainer(new_model)
    epoch = new_trainer.load_checkpoint(chk)

    assert epoch == 3
    for p_old, p_new in zip(model.parameters(), new_model.parameters()):
        assert torch.allclose(p_old, p_new)
