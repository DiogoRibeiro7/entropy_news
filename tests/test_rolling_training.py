import pytest

torch = pytest.importorskip("torch")

from entropy_news.rolling_train_forecast import (
    prepare_training_set,
    train_model,
    update_with_new_month,
    load_texts_for_month,
)


def test_train_and_update(tmp_path):
    (tmp_path / "news_2023-01.txt").write_text("a b c\nb c d\n")
    (tmp_path / "news_2023-02.txt").write_text("c d e\n")

    train_ds, pre = prepare_training_set(["2023-01"], str(tmp_path), seq_len=3, vocab_size=10)
    model = train_model(
        train_ds,
        vocab_size=len(pre.vocab),
        embed_dim=4,
        hidden_dim=2,
        learning_rate=0.01,
        epochs=1,
        batch_size=1,
    )

    new_texts = load_texts_for_month("2023-02", str(tmp_path))
    result = update_with_new_month(
        model,
        pre,
        new_texts,
        seq_len=3,
        embed_dim=4,
        hidden_dim=2,
        fine_tune_epochs=1,
        learning_rate=0.01,
    )

    assert set(result.keys()) == {"ENT", "ENT_news", "ENT_model"}

