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


def test_load_glove_embeddings_missing_file(tmp_path: Path) -> None:
    """A clear ``FileNotFoundError`` should be raised for absent files."""

    pre = TextPreprocessor()
    missing = tmp_path / "no.txt"

    with pytest.raises(FileNotFoundError):
        pre.load_glove_embeddings(missing)


def test_load_glove_embeddings_skips_bad_lines(tmp_path: Path) -> None:
    """Malformed lines are skipped without raising errors."""

    glove_file = tmp_path / "glove.txt"
    glove_file.write_text("word1 0.1 0.2\nbad_line\nword2 0.3 0.4\n")

    pre = TextPreprocessor(vocab_size=10)
    pre.vocab = {"<PAD>": 0, "<UNK>": 1, "word1": 2, "word2": 3}
    pre.reverse_vocab = {idx: word for word, idx in pre.vocab.items()}
    pre.load_glove_embeddings(glove_file, embedding_dim=2)

    assert pre.embedding_matrix.shape == (len(pre.vocab), 2)


def test_load_glove_embeddings_invalid_encoding(tmp_path: Path) -> None:
    """Invalid encoding triggers a clear ``ValueError``."""

    glove_file = tmp_path / "glove.txt"
    glove_file.write_bytes(b"\xff\xff")
    pre = TextPreprocessor()

    with pytest.raises(ValueError, match="Failed to decode GloVe file"):
        pre.load_glove_embeddings(glove_file)


def test_load_glove_embeddings_seed_reproducible(tmp_path: Path) -> None:
    """Providing a seed yields deterministic random vectors."""

    glove_file = tmp_path / "glove.txt"
    glove_file.write_text("known 0.1 0.2\n")

    vocab = {"<PAD>": 0, "<UNK>": 1, "known": 2, "other": 3}
    reverse = {idx: word for word, idx in vocab.items()}

    pre1 = TextPreprocessor(vocab_size=10)
    pre1.vocab = vocab
    pre1.reverse_vocab = reverse
    pre1.load_glove_embeddings(glove_file, embedding_dim=2, seed=123)

    pre2 = TextPreprocessor(vocab_size=10)
    pre2.vocab = vocab
    pre2.reverse_vocab = reverse
    pre2.load_glove_embeddings(glove_file, embedding_dim=2, seed=123)

    np.testing.assert_array_equal(pre1.embedding_matrix[3], pre2.embedding_matrix[3])

def test_clean_text_and_tokenize() -> None:
    """Cleaning removes punctuation and lowers case."""

    pre = TextPreprocessor()
    text = "Hello, WORLD!!"
    cleaned = pre.clean_text(text)
    assert cleaned == "hello world"
    assert pre.tokenize(cleaned) == ["hello", "world"]


def test_encode_decode_roundtrip() -> None:
    """Encoding then decoding should yield the original text."""

    pre = TextPreprocessor(vocab_size=5)
    texts = ["foo bar", "bar baz"]
    pre.build_vocab(texts)

    encoded = pre.encode("foo bar")
    decoded = pre.decode(encoded)
    assert decoded == "foo bar"
