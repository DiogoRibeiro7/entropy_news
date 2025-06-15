# entropy_news/data/preprocessor.py

import re
import os
import logging
import json
from collections import Counter
from typing import List, Dict, Tuple, Optional

import numpy as np

logger = logging.getLogger(__name__)

class TextPreprocessor:
    """Text cleaning and vocabulary management helper."""

    def __init__(self, vocab_size: int = 10000):
        """Initialise the preprocessor with ``vocab_size`` tokens.

        Parameters
        ----------
        vocab_size : int, optional
            Maximum number of words to keep, by default ``10000``.
        """
        self.vocab_size = vocab_size
        self.vocab: Dict[str, int] = {}
        self.reverse_vocab: Dict[int, str] = {}
        self.embedding_matrix: Optional[np.ndarray] = None

    def clean_text(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def tokenize(self, text: str) -> List[str]:
        return text.split()

    def build_vocab(self, texts: List[str]) -> None:
        logger.info("Building vocabulary...")
        counter: Counter[str] = Counter()
        for text in texts:
            tokens = self.tokenize(self.clean_text(text))
            counter.update(tokens)

        most_common = counter.most_common(self.vocab_size)
        self.vocab = {word: idx + 2 for idx, (word, _) in enumerate(most_common)}
        self.vocab['<PAD>'] = 0
        self.vocab['<UNK>'] = 1
        self.reverse_vocab = {idx: word for word, idx in self.vocab.items()}
        logger.info(f"Vocabulary size: {len(self.vocab)}")

    def encode(self, text: str) -> List[int]:
        tokens = self.tokenize(self.clean_text(text))
        return [self.vocab.get(token, self.vocab['<UNK>']) for token in tokens]

    def decode(self, ids: List[int]) -> str:
        return ' '.join([self.reverse_vocab.get(idx, '<UNK>') for idx in ids])

    def load_glove_embeddings(self, glove_path: str, embedding_dim: int | None = None) -> None:
        """Load pre-trained GloVe vectors and build ``embedding_matrix``.

        The dimension is automatically inferred from the first line of the file
        unless ``embedding_dim`` is provided.  In either case the resulting
        ``embedding_matrix`` has shape ``(len(vocab), embedding_dim)`` and is
        initialised with a normal distribution when no vector is available for a
        given token.
        """
        logger.info("Loading GloVe embeddings...")
        embeddings_index = {}

        with open(glove_path, "r", encoding="utf-8") as f:
            first_line = f.readline()
            parts = first_line.strip().split()
            detected_dim = len(parts) - 1
            if embedding_dim is None:
                embedding_dim = detected_dim

            word = parts[0]
            vector = np.asarray(parts[1:], dtype="float32")[:embedding_dim]
            embeddings_index[word] = vector

            for line in f:
                values = line.strip().split()
                word = values[0]
                vector = np.asarray(values[1:], dtype="float32")[:embedding_dim]
                embeddings_index[word] = vector

        self.embedding_matrix = np.random.normal(0, 1, (len(self.vocab), embedding_dim))
        for word, idx in self.vocab.items():
            vector = embeddings_index.get(word)
            if vector is not None and isinstance(vector, np.ndarray) and len(vector) == embedding_dim:
                self.embedding_matrix[idx] = vector

        logger.info(
            f"Loaded GloVe embeddings with shape: {self.embedding_matrix.shape}"
        )

    def save_vocab(self, filepath: str) -> None:
        """Persist the current vocabulary to ``filepath`` as JSON.

        Filenames without a directory component are allowed.
        """
        directory = os.path.dirname(filepath)
        if directory:
            os.makedirs(directory, exist_ok=True)
        data = {
            "vocab_size": self.vocab_size,
            "vocab": self.vocab,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        logger.info("Saved vocabulary to %s", filepath)

    def load_vocab(self, filepath: str) -> None:
        """Load a vocabulary from ``filepath`` and rebuild lookup tables."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.vocab = {str(k): int(v) for k, v in data.get("vocab", {}).items()}
        self.vocab_size = int(data.get("vocab_size", len(self.vocab)))
        self.reverse_vocab = {idx: word for word, idx in self.vocab.items()}
        logger.info("Loaded vocabulary from %s", filepath)
