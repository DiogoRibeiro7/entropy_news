"""Tests for ModelConfig serialization helpers."""

from entropy_news.model import ModelConfig


def test_model_config_json_roundtrip(tmp_path) -> None:
    cfg = ModelConfig(architecture="lstm", vocab_size=100, embed_dim=32, hidden_dim=16)
    json_str = cfg.to_json()
    loaded = ModelConfig.from_json(json_str)
    assert loaded == cfg

    path = tmp_path / "cfg.json"
    cfg.save(path)
    loaded_from_file = ModelConfig.load(path)
    assert loaded_from_file == cfg
