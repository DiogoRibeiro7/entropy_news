"""Property-based tests for preprocessing and streaming dataset utilities."""

from __future__ import annotations

import string

import pytest
import torch

pytest.importorskip("hypothesis")
from hypothesis import HealthCheck, assume, given, settings, strategies as st

from entropy_news.data.preprocessor import TextPreprocessor
from entropy_news.data.streaming_dataset import StreamingNewsDataset


ALPHABET = string.ascii_letters + string.digits + " "


@settings(max_examples=50, deadline=None)
@given(
    st.lists(
        st.text(alphabet=ALPHABET, min_size=0, max_size=30),
        min_size=1,
        max_size=10,
    )
)
def test_preprocessor_encoding_within_vocab(texts: list[str]) -> None:
    """Encoding should only emit token ids present in the vocabulary."""

    preprocessor = TextPreprocessor(vocab_size=256)
    preprocessor.build_vocab(texts)
    valid_ids = set(preprocessor.vocab.values())
    valid_tokens = set(preprocessor.reverse_vocab.values())

    for text in texts:
        encoded = preprocessor.encode(text)
        assert all(idx in valid_ids for idx in encoded)
        decoded_tokens = preprocessor.decode(encoded).split()
        assert set(decoded_tokens).issubset(valid_tokens)


@settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    st.lists(
        st.text(alphabet=ALPHABET, min_size=0, max_size=40),
        min_size=1,
        max_size=20,
    )
)
def test_streaming_dataset_shifted_targets(tmp_path, texts: list[str]) -> None:
    """Streaming dataset should emit sequential input/target pairs."""

    preprocessor = TextPreprocessor(vocab_size=512)
    preprocessor.build_vocab(texts)

    corpus_path = tmp_path / "corpus.txt"
    payload = "\n".join(texts)
    if not payload.endswith("\n"):
        payload += "\n"
    corpus_path.write_text(payload)
    expected_lines = len(payload.splitlines())
    assume(expected_lines > 0)

    dataset = StreamingNewsDataset(
        str(corpus_path), preprocessor, seq_len=12, chunk_size=5, cache_size=3
    )

    assert len(dataset) == expected_lines

    for idx in range(expected_lines):
        inputs, targets = dataset[idx]
        assert inputs.shape == (12,)
        assert targets.shape == (12,)
        assert inputs.dtype == torch.long
        assert targets.dtype == torch.long
        assert torch.equal(inputs[1:], targets[:-1])
