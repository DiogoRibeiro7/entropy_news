"""Tests for pluggable tokenization strategies."""

from entropy_news.data.preprocessor import TextPreprocessor


class CharTokenizer:
    """Tokenize text into individual characters."""

    def tokenize(self, text: str) -> list[str]:
        return list(text)


def test_preprocessor_uses_custom_tokenizer() -> None:
    tokenizer = CharTokenizer()
    pre = TextPreprocessor(tokenizer=tokenizer, vocab_size=10)
    pre.build_vocab(["ab"])
    assert pre.encode("ab") == [pre.vocab["a"], pre.vocab["b"]]

