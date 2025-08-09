import gzip
import pytest

torch = pytest.importorskip("torch")

from entropy_news.data import TextPreprocessor
from entropy_news.data.streaming_dataset import StreamingNewsDataset


def test_streaming_dataset_yields_padded_sequences(tmp_path) -> None:
    texts = ["hello world", "foo bar baz", "lorem ipsum"]
    file = tmp_path / "texts.txt"
    file.write_text("\n".join(texts))

    pre = TextPreprocessor()
    pre.build_vocab(texts)

    ds = StreamingNewsDataset(str(file), pre, seq_len=4, chunk_size=2, cache_size=1)
    assert len(ds) == 3

    x, y = ds[1]
    encoded = pre.encode(texts[1])[:5]
    while len(encoded) < 5:
        encoded.append(0)
    expected_x = encoded[:-1]
    expected_y = encoded[1:]
    assert x.tolist() == expected_x
    assert y.tolist() == expected_y


def test_streaming_dataset_reads_gzip(tmp_path) -> None:
    texts = ["alpha", "beta", "gamma"]
    gz = tmp_path / "texts.txt.gz"
    with gzip.open(gz, "wt", encoding="utf-8") as f:
        f.write("\n".join(texts))

    pre = TextPreprocessor()
    pre.build_vocab(texts)

    ds = StreamingNewsDataset(str(gz), pre, seq_len=4, chunk_size=2, cache_size=1)
    assert len(ds) == 3
    x, _ = ds[0]
    assert x.numel() == 4
