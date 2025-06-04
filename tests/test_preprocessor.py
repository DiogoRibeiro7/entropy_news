import json
from pathlib import Path

import pytest
np = pytest.importorskip("numpy")
from entropy_news.data.preprocessor import TextPreprocessor


def test_save_and_load_vocab(tmp_path: Path):
    texts = ["hello world", "hello there"]
    pre = TextPreprocessor(vocab_size=10)
    pre.build_vocab(texts)

    file_path = tmp_path / "vocab.json"
    pre.save_vocab(file_path)
    assert file_path.exists()

    new_pre = TextPreprocessor()
    new_pre.load_vocab(file_path)

    assert new_pre.vocab == pre.vocab
    assert new_pre.vocab_size == pre.vocab_size


def test_load_glove_embeddings_infer_dim(tmp_path: Path):
    glove_file = tmp_path / "glove.txt"
    glove_file.write_text("word1 0.1 0.2 0.3\nword2 0.4 0.5 0.6\n")

    pre = TextPreprocessor(vocab_size=10)
    pre.vocab = {"<PAD>": 0, "<UNK>": 1, "word1": 2, "word2": 3}
    pre.reverse_vocab = {idx: word for word, idx in pre.vocab.items()}
    pre.load_glove_embeddings(glove_file)

    assert pre.embedding_matrix.shape == (len(pre.vocab), 3)
    np.testing.assert_array_almost_equal(pre.embedding_matrix[2], [0.1, 0.2, 0.3])


def test_load_glove_embeddings_override_dim(tmp_path: Path):
    glove_file = tmp_path / "glove.txt"
    glove_file.write_text("word1 0.1 0.2 0.3\n")

    pre = TextPreprocessor(vocab_size=10)
    pre.vocab = {"<PAD>": 0, "<UNK>": 1, "word1": 2}
    pre.reverse_vocab = {idx: word for word, idx in pre.vocab.items()}
    pre.load_glove_embeddings(glove_file, embedding_dim=2)

    assert pre.embedding_matrix.shape == (len(pre.vocab), 2)
    np.testing.assert_array_almost_equal(pre.embedding_matrix[2], [0.1, 0.2])
