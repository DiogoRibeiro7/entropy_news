import json
from pathlib import Path

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
