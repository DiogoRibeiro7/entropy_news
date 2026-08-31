import hashlib
import json

import numpy as np
import pytest

from entropy_news.data import TextPreprocessor
from entropy_news.paper_architecture import (
    PAPER_UNK_EMBEDDING_CONVENTION,
    PAPER_UNK_EMBEDDING_PAPER_SPECIFIED,
    build_paper_vocabulary,
    load_paper_glove_embeddings,
)


def test_paper_vocabulary_requires_full_predictive_cardinality() -> None:
    pre = TextPreprocessor(vocab_size=4)
    with pytest.raises(
        ValueError,
        match="paper vocabulary requires 3 distinct lexical tokens; found 2",
    ):
        build_paper_vocabulary(pre, ["market news market"], 4)


def test_paper_unk_embedding_is_seeded_and_not_taken_from_glove(tmp_path) -> None:
    pre = TextPreprocessor(vocab_size=4)
    padding_id = build_paper_vocabulary(pre, ["market news signal market"], 4)
    glove = tmp_path / "glove.txt"
    glove.write_text(
        "<UNK> 9.0 9.0 9.0\n"
        "market 1.0 2.0 3.0\n"
        "news 4.0 5.0 6.0\n"
        "signal 7.0 8.0 9.0\n",
        encoding="utf-8",
    )

    seed = 17
    load_paper_glove_embeddings(
        pre,
        str(glove),
        embedding_dim=3,
        seed=seed,
        padding_id=padding_id,
        show_progress=False,
    )

    expected = np.random.default_rng(seed).normal(0, 1, 3).astype("float32")
    actual = pre.embedding_matrix[pre.vocab["<UNK>"]]
    assert np.array_equal(actual, expected)
    assert not np.array_equal(actual, np.array([9.0, 9.0, 9.0], dtype="float32"))
    assert np.array_equal(pre.embedding_matrix[padding_id], np.zeros(3, dtype="float32"))

    metadata = pre.vocab_metadata["paper_unk_embedding"]
    assert metadata["convention"] == PAPER_UNK_EMBEDDING_CONVENTION
    assert metadata["paper_specified"] is PAPER_UNK_EMBEDDING_PAPER_SPECIFIED is False
    assert metadata["seed"] == seed
    assert metadata["sha256"] == hashlib.sha256(expected.tobytes()).hexdigest()


def test_paper_unk_provenance_survives_vocabulary_round_trip(tmp_path) -> None:
    pre = TextPreprocessor(vocab_size=4)
    padding_id = build_paper_vocabulary(pre, ["market news signal market"], 4)
    glove = tmp_path / "glove.txt"
    glove.write_text(
        "market 1.0 2.0 3.0\nnews 4.0 5.0 6.0\nsignal 7.0 8.0 9.0\n",
        encoding="utf-8",
    )
    load_paper_glove_embeddings(
        pre,
        str(glove),
        embedding_dim=3,
        seed=5,
        padding_id=padding_id,
        show_progress=False,
    )

    vocab_path = tmp_path / "paper_vocabulary.json"
    pre.save_vocab(str(vocab_path))
    raw = json.loads(vocab_path.read_text(encoding="utf-8"))
    assert raw["metadata"]["paper_unk_embedding"]["paper_specified"] is False

    restored = TextPreprocessor(vocab_size=1)
    restored.load_vocab(str(vocab_path))
    assert restored.vocab_metadata == pre.vocab_metadata
