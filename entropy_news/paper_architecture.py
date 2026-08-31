"""Paper-specific vocabulary and batching contracts."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from entropy_news.data import TextPreprocessor
from entropy_news.paper_reproduction import chunk_articles_for_training, filter_articles

PAPER_TARGET_IGNORE_INDEX = -100
PAPER_UNK_EMBEDDING_CONVENTION = "seeded_random_normal_0_1"
PAPER_UNK_EMBEDDING_PAPER_SPECIFIED = False


def build_paper_vocabulary(
    preprocessor: TextPreprocessor,
    texts: Sequence[str],
    predictive_vocab_size: int,
) -> int:
    """Build ``UNK + (V-1) lexical words`` and return a separate padding ID."""
    if predictive_vocab_size < 2:
        raise ValueError("predictive_vocab_size must be at least 2")

    counter: Counter[str] = Counter()
    for text in texts:
        counter.update(preprocessor.tokenize(preprocessor.clean_text(text)))

    lexical = counter.most_common(predictive_vocab_size - 1)
    preprocessor.vocab = {"<UNK>": 0}
    preprocessor.vocab.update(
        {word: index for index, (word, _) in enumerate(lexical, start=1)}
    )
    preprocessor.reverse_vocab = {
        index: word for word, index in preprocessor.vocab.items()
    }
    preprocessor.vocab_size = predictive_vocab_size
    return predictive_vocab_size


def load_paper_glove_embeddings(
    preprocessor: TextPreprocessor,
    glove_path: str,
    *,
    embedding_dim: int,
    seed: int,
    padding_id: int,
    show_progress: bool,
) -> None:
    """Load frozen paper embeddings with explicit UNK and padding conventions.

    The paper specifies 100-dimensional GloVe vectors and an ``UNK`` token but
    does not specify the vector assigned to ``UNK``.  The strict reproduction
    path therefore makes its implementation inference explicit: ``UNK`` uses
    a deterministic ``N(0, 1)`` vector generated from the protocol seed.  This
    matches the historical generic-loader behaviour for standard GloVe files
    while preventing an incidental ``<UNK>`` row in a custom file from silently
    changing the convention. Padding remains a separate all-zero input row.
    """
    if padding_id != len(preprocessor.vocab):
        raise ValueError("paper padding ID must follow the predictive vocabulary")
    preprocessor.load_glove_embeddings(
        glove_path,
        embedding_dim=embedding_dim,
        seed=seed,
        show_progress=show_progress,
    )
    matrix = preprocessor.embedding_matrix
    if matrix is None:
        raise RuntimeError("GloVe loader did not create an embedding matrix")

    unk_id = preprocessor.vocab["<UNK>"]
    rng = np.random.default_rng(seed)
    matrix[unk_id] = rng.normal(0, 1, embedding_dim).astype(matrix.dtype)

    zero_row = np.zeros((1, embedding_dim), dtype=matrix.dtype)
    preprocessor.embedding_matrix = np.concatenate([matrix, zero_row], axis=0)


class PaperNewsDataset(Dataset):
    """Fixed-width training chunks with padding outside the predictive classes."""

    def __init__(
        self,
        encoded_chunks: Sequence[Sequence[int]],
        *,
        seq_len: int,
        padding_id: int,
    ) -> None:
        self.seq_len = seq_len
        self.padding_id = padding_id
        self.data = [list(chunk) for chunk in encoded_chunks]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        sequence = self.data[idx][: self.seq_len + 1]
        if len(sequence) < self.seq_len + 1:
            sequence = sequence + [self.padding_id] * (
                self.seq_len + 1 - len(sequence)
            )
        full = torch.tensor(sequence, dtype=torch.long)
        x = full[:-1]
        y = full[1:].clone()
        y[y == self.padding_id] = PAPER_TARGET_IGNORE_INDEX
        return x, y


def make_paper_training_dataset(
    texts: Sequence[str],
    preprocessor: TextPreprocessor,
    sequence_length: int,
    *,
    min_article_words: int,
    padding_id: int,
) -> PaperNewsDataset:
    retained = filter_articles(texts, preprocessor, min_article_words)
    encoded = [preprocessor.encode(text) for text in retained]
    chunks = chunk_articles_for_training(encoded, sequence_length)
    return PaperNewsDataset(chunks, seq_len=sequence_length, padding_id=padding_id)
