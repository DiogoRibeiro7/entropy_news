"""Tests for CLI helper utilities."""

import json
from pathlib import Path

import pytest



def _write_vocab(path: Path) -> None:
    data = {"<PAD>": 0, "hello": 1}
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"vocab_size": len(data), "vocab": data}, f)


def test_load_model_and_vocab_missing_model(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    from entropy_news.utils.cli import load_model_and_vocab

    vocab_file = tmp_path / "vocab.json"
    _write_vocab(vocab_file)
    with pytest.raises(FileNotFoundError):
        load_model_and_vocab(
            str(vocab_file),
            str(tmp_path / "missing.pth"),
            embed_dim=10,
            hidden_dim=16,
            num_layers=2,
            dropout=0.1,
        )




def test_load_model_and_vocab_from_config(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    from entropy_news.utils.cli import load_model_and_vocab
    from entropy_news.model import EntropyLSTMAttention, ModelConfig

    vocab_file = tmp_path / "vocab.json"
    _write_vocab(vocab_file)
    config_path = tmp_path / "model_config.json"
    ModelConfig(
        architecture="lstm_attention",
        vocab_size=2,
        embed_dim=4,
        hidden_dim=4,
        num_layers=1,
        num_heads=2,
        ff_dim=8,
        dropout=0.1,
    ).save(config_path)

    model = EntropyLSTMAttention(vocab_size=2, embed_dim=4, hidden_dim=4, num_layers=1, num_heads=2)
    model_path = tmp_path / "model.pth"
    torch.save(model.state_dict(), model_path)

    _preprocessor, _model, _device, cfg = load_model_and_vocab(
        str(vocab_file),
        str(model_path),
        config_path=str(config_path),
    )
    assert cfg.architecture == "lstm_attention"


def test_load_model_and_vocab_uses_map_location(tmp_path, monkeypatch) -> None:
    """``torch.load`` should be called with a ``map_location`` argument."""

    torch = pytest.importorskip("torch")
    from entropy_news.utils.cli import load_model_and_vocab
    from entropy_news.model import EntropyLSTM

    vocab_file = tmp_path / "vocab.json"
    _write_vocab(vocab_file)

    # Create and save a dummy model
    model = EntropyLSTM(vocab_size=2, embed_dim=4, hidden_dim=4)
    model_path = tmp_path / "model.pth"
    torch.save(model.state_dict(), model_path)

    called: dict[str, object | None] = {}
    state_dict = model.state_dict()

    def fake_load(path, *args, **kwargs):  # type: ignore[override]
        called["map_location"] = kwargs.get("map_location")
        called["weights_only"] = kwargs.get("weights_only")
        return state_dict

    monkeypatch.setattr(torch, "load", fake_load)

    device = torch.device("cpu")
    _preprocessor, _model, used_device, _config = load_model_and_vocab(
        str(vocab_file),
        str(model_path),
        embed_dim=4,
        hidden_dim=4,
        num_layers=1,
        dropout=0.0,
        device=device,
    )

    assert called["map_location"] == device
    assert called["weights_only"] is True
    assert used_device == device


def _write_config(path: Path) -> None:
    """Persist a minimal :class:`ModelConfig` for tests."""

    from entropy_news.model import ModelConfig

    ModelConfig(
        architecture="lstm",
        vocab_size=2,
        embed_dim=4,
        hidden_dim=4,
        num_layers=1,
        num_heads=2,
        ff_dim=8,
        dropout=0.1,
    ).save(path)


def _fake_state_dict() -> dict[str, object]:
    """Return a consistent state dict for LSTM-based tests."""

    from entropy_news.model import EntropyLSTM

    model = EntropyLSTM(vocab_size=2, embed_dim=4, hidden_dim=4)
    return model.state_dict()


def test_load_model_and_vocab_requires_flag_for_unsafe(
    tmp_path, monkeypatch
) -> None:
    torch = pytest.importorskip("torch")
    from entropy_news.utils.cli import load_model_and_vocab

    vocab_file = tmp_path / "vocab.json"
    _write_vocab(vocab_file)
    config_path = tmp_path / "model_config.json"
    _write_config(config_path)
    model_path = tmp_path / "model.pth"
    model_path.write_bytes(b"legacy")

    state_dict = _fake_state_dict()

    def fake_load(path, *args, **kwargs):  # type: ignore[override]
        weights_only = kwargs.get("weights_only")
        if weights_only in (None, True):
            raise RuntimeError("requires weights_only_false")
        return state_dict

    monkeypatch.setattr(torch, "load", fake_load)

    with pytest.raises(OSError) as excinfo:
        load_model_and_vocab(
            str(vocab_file),
            str(model_path),
            config_path=str(config_path),
        )
    assert "--allow-unsafe-load" in str(excinfo.value)


def test_load_model_and_vocab_allows_flag(tmp_path, monkeypatch, caplog) -> None:
    torch = pytest.importorskip("torch")
    from entropy_news.model import EntropyLSTM
    from entropy_news.utils.cli import load_model_and_vocab

    vocab_file = tmp_path / "vocab.json"
    _write_vocab(vocab_file)
    config_path = tmp_path / "model_config.json"
    _write_config(config_path)
    model_path = tmp_path / "model.pth"
    model_path.write_bytes(b"legacy")

    state_dict = _fake_state_dict()

    def fake_load(path, *args, **kwargs):  # type: ignore[override]
        weights_only = kwargs.get("weights_only")
        if weights_only in (None, True):
            raise RuntimeError("requires weights_only_false")
        return state_dict

    monkeypatch.setattr(torch, "load", fake_load)

    with caplog.at_level("WARNING"):
        _preprocessor, model, _device, _config = load_model_and_vocab(
            str(vocab_file),
            str(model_path),
            config_path=str(config_path),
            allow_unsafe_load=True,
        )
    assert any(
        "Unsafe checkpoint loading enabled" in record.message for record in caplog.records
    )
    assert isinstance(model, EntropyLSTM)


def test_load_model_and_vocab_env_flag(tmp_path, monkeypatch, caplog) -> None:
    torch = pytest.importorskip("torch")
    from entropy_news.utils.cli import load_model_and_vocab

    monkeypatch.setenv("ENTROPY_NEWS_ALLOW_UNSAFE_LOAD", "true")

    vocab_file = tmp_path / "vocab.json"
    _write_vocab(vocab_file)
    config_path = tmp_path / "model_config.json"
    _write_config(config_path)
    model_path = tmp_path / "model.pth"
    model_path.write_bytes(b"legacy")

    state_dict = _fake_state_dict()

    def fake_load(path, *args, **kwargs):  # type: ignore[override]
        weights_only = kwargs.get("weights_only")
        if weights_only in (None, True):
            raise RuntimeError("requires weights_only_false")
        return state_dict

    monkeypatch.setattr(torch, "load", fake_load)

    with caplog.at_level("WARNING"):
        load_model_and_vocab(
            str(vocab_file),
            str(model_path),
            config_path=str(config_path),
        )
    assert any(
        "Unsafe checkpoint loading enabled" in record.message for record in caplog.records
    )


def test_load_encoded_dataset(tmp_path) -> None:
    pytest.importorskip("numpy")
    from entropy_news.data import TextPreprocessor
    from entropy_news.utils.cli import load_encoded_dataset

    text_file = tmp_path / "data.txt"
    text_file.write_text("hello\n")
    preprocessor = TextPreprocessor()
    preprocessor.build_vocab(["hello"])

    dataset = load_encoded_dataset(
        preprocessor, str(text_file), seq_len=5, lazy=False
    )
    assert len(dataset) == 1


def test_resolve_model_config_prefers_overrides() -> None:
    from entropy_news.model import ModelConfig
    from entropy_news.utils.cli import ConfigOverrides, resolve_model_config

    base = ModelConfig(
        architecture="lstm",
        vocab_size=100,
        embed_dim=32,
        hidden_dim=64,
        num_heads=2,
        ff_dim=128,
        num_layers=2,
        dropout=0.2,
    )
    result = resolve_model_config(
        base_config=base,
        overrides=ConfigOverrides(hidden_dim=128, dropout=0.3),
    )
    assert result.hidden_dim == 128
    assert result.dropout == pytest.approx(0.3)
    assert result.embed_dim == 32


def test_resolve_model_config_uses_defaults() -> None:
    from entropy_news.utils.cli import ConfigDefaults, ConfigOverrides, resolve_model_config

    defaults = ConfigDefaults(vocab_size=50, embed_dim=8)
    result = resolve_model_config(
        base_config=None,
        overrides=ConfigOverrides(),
        defaults=defaults,
    )
    assert result.architecture == "lstm"
    assert result.vocab_size == 50
    assert result.embed_dim == 8
