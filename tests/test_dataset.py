import pytest
torch = pytest.importorskip("torch")
from entropy_news.data.dataset import NewsDataset


def test_padding_and_truncation():
    sequences = [[1, 2, 3], [4, 5, 6, 7, 8, 9]]
    dataset = NewsDataset(sequences, seq_len=4)

    x0, y0 = dataset[0]
    assert x0.tolist() == [1, 2, 3, 0]
    assert y0.tolist() == [2, 3, 0, 0]

    x1, y1 = dataset[1]
    assert x1.tolist() == [4, 5, 6, 7]
    assert y1.tolist() == [5, 6, 7, 8]


def test_dataset_len_matches_input() -> None:
    """``len(dataset)`` should equal number of provided sequences."""

    sequences = [[1], [2], [3]]
    dataset = NewsDataset(sequences, seq_len=1)

    assert len(dataset) == 3


def test_dataset_getitem_returns_shifted_pairs() -> None:
    """``__getitem__`` should return input and target shifted by one."""

    torch = pytest.importorskip("torch")
    dataset = NewsDataset([[1, 2, 3, 4]], seq_len=3)

    x, y = dataset[0]
    assert x.tolist() == [1, 2, 3]
    assert y.tolist() == [2, 3, 4]
