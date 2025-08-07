import pytest

torch = pytest.importorskip('torch')

from entropy_news.model import (
    EntropyLSTM,
    EntropyTransformer,
    ModelConfig,
    ModelFactory,
)


def test_model_factory_builds_lstm() -> None:
    cfg = ModelConfig(architecture='lstm', vocab_size=8, embed_dim=32)
    model = ModelFactory.create(cfg)
    assert isinstance(model, EntropyLSTM)
    assert model.embedding.embedding_dim == 32


def test_model_factory_builds_transformer() -> None:
    cfg = ModelConfig(
        architecture='transformer',
        vocab_size=10,
        embed_dim=32,
        num_heads=2,
        ff_dim=64,
    )
    model = ModelFactory.create(cfg)
    assert isinstance(model, EntropyTransformer)
    assert model.fc.in_features == 32


def test_model_config_serialization_roundtrip() -> None:
    cfg = ModelConfig(architecture='lstm', vocab_size=4, hidden_dim=20)
    data = cfg.to_dict()
    new_cfg = ModelConfig.from_dict(data)
    assert new_cfg == cfg


def test_model_config_validation() -> None:
    cfg = ModelConfig(architecture='lstm', vocab_size=0)
    with pytest.raises(ValueError):
        cfg.validate()


def test_transformer_config_validation_heads() -> None:
    cfg = ModelConfig(architecture='transformer', vocab_size=8, embed_dim=30, num_heads=7)
    with pytest.raises(ValueError):
        cfg.validate()
