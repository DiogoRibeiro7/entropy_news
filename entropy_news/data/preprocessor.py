# entropy_news/data/preprocessor.py

import re
import os
import logging
import json
from collections import Counter
from typing import Dict, List

from entropy_news.types import EmbeddingMatrix

import numpy as np

logger = logging.getLogger(__name__)

class TextPreprocessor:
    """Utility for text cleaning and vocabulary management."""

    def __init__(self, vocab_size: int = 10000):
        """Instantiate the preprocessor.

        Args:
            vocab_size: Maximum number of words to keep. Defaults to ``10000``.
        """
        self.vocab_size = vocab_size
        self.vocab: Dict[str, int] = {}
        self.reverse_vocab: Dict[int, str] = {}
        self.embedding_matrix: EmbeddingMatrix | None = None

    def clean_text(self, text: str) -> str:
        """Normalize text by lowercasing and removing punctuation.

        Args:
            text: Raw text to be cleaned.

        Returns:
            The processed string with collapsed whitespace.
        """
        # Basic normalization
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def tokenize(self, text: str) -> List[str]:
        """Split cleaned ``text`` into tokens.

        Args:
            text: String returned by :meth:`clean_text`.

        Returns:
            List of whitespace-separated tokens.
        """
        return text.split()

    def build_vocab(self, texts: List[str]) -> None:
        """Populate ``vocab`` with the most frequent words.

        Args:
            texts: Collection of raw texts used to build the vocabulary.
        """
        logger.info("Building vocabulary...")
        # Tally token frequencies
        counter: Counter[str] = Counter()
        for text in texts:
            tokens = self.tokenize(self.clean_text(text))
            counter.update(tokens)

        most_common = counter.most_common(self.vocab_size)
        # Reserve 0 and 1 for PAD and UNK respectively
        self.vocab = {word: idx + 2 for idx, (word, _) in enumerate(most_common)}
        self.vocab["<PAD>"] = 0
        self.vocab["<UNK>"] = 1
        self.reverse_vocab = {idx: word for word, idx in self.vocab.items()}
        logger.info(f"Vocabulary size: {len(self.vocab)}")

    def encode(self, text: str) -> List[int]:
        """Convert text to token IDs using the current vocabulary.

        Args:
            text: Raw text to encode.

        Returns:
            Sequence of token identifiers.
        """
        tokens = self.tokenize(self.clean_text(text))
        # Map tokens to ids, defaulting to <UNK>
        return [self.vocab.get(token, self.vocab["<UNK>"]) for token in tokens]

    def decode(self, ids: List[int]) -> str:
        """Reconstruct text from token IDs.

        Args:
            ids: Sequence of token identifiers.

        Returns:
            Decoded string with tokens joined by spaces.
        """
        # Convert ids back to tokens
        return " ".join([self.reverse_vocab.get(idx, "<UNK>") for idx in ids])

    def load_glove_embeddings(
        self,
        glove_path: str,
        embedding_dim: int | None = None,
        seed: int | None = None,
        show_progress: bool = False,
    ) -> None:
        """Load pre-trained GloVe vectors and create ``embedding_matrix``.

        Args:
            glove_path: Path to the GloVe text file.
            embedding_dim: Desired vector dimension. If ``None``, it is
                inferred from the first line of the file.
            seed: Optional random seed for deterministic initialisation.
            show_progress: Whether to display a loading progress bar.

        Raises:
            FileNotFoundError: If ``glove_path`` does not exist.
            ValueError: If the file is empty or contains malformed lines.
        """

        logger.info("Loading GloVe embeddings...")

        if not os.path.exists(glove_path):
            msg = f"GloVe file not found: {glove_path}"
            logger.error(msg)
            raise FileNotFoundError(msg)

        rng = np.random.default_rng(seed)

        with open(glove_path, "r", encoding="utf-8") as f:
            first_line = f.readline().strip()
            if not first_line:
                raise ValueError("GloVe file is empty")

            parts = first_line.split()
            if len(parts) < 2:
                raise ValueError(f"Malformed line in GloVe file: {first_line}")

            detected_dim = len(parts) - 1
            if embedding_dim is None:
                embedding_dim = detected_dim

            self.embedding_matrix = rng.normal(0, 1, (len(self.vocab), embedding_dim))

            word = parts[0]
            vector = np.asarray(parts[1:], dtype="float32")[:embedding_dim]
            idx = self.vocab.get(word)
            if idx is not None and len(vector) == embedding_dim:
                self.embedding_matrix[idx] = vector

            iterator = f
            if show_progress:
                from tqdm import tqdm

                iterator = tqdm(f, desc="Loading GloVe", unit="vec")

            for line in iterator:
                values = line.strip().split()
                if len(values) < embedding_dim + 1:
                    logger.warning("Skipping malformed line: %s", line.strip())
                    continue
                word = values[0]
                vector = np.asarray(values[1:], dtype="float32")[:embedding_dim]
                idx = self.vocab.get(word)
                if idx is not None and len(vector) == embedding_dim:
                    self.embedding_matrix[idx] = vector

        logger.info(
            f"Loaded GloVe embeddings with shape: {self.embedding_matrix.shape}"
        )

    def save_vocab(self, filepath: str) -> None:
        """Persist the current vocabulary to a JSON file.

        Args:
            filepath: Destination path for the JSON vocabulary.
        """
        directory = os.path.dirname(filepath)
        if directory:
            # Make sure the target directory is present
            os.makedirs(directory, exist_ok=True)
        data = {
            "vocab_size": self.vocab_size,
            "vocab": self.vocab,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        logger.info("Saved vocabulary to %s", filepath)

    def load_vocab(self, filepath: str) -> None:
        """Load a saved vocabulary.

        Args:
            filepath: JSON file produced by :meth:`save_vocab`.
        """
        with open(filepath, "r", encoding="utf-8") as f:
            # Load the saved vocabulary dictionary
            data = json.load(f)
        self.vocab = {str(k): int(v) for k, v in data.get("vocab", {}).items()}
        self.vocab_size = int(data.get("vocab_size", len(self.vocab)))
        self.reverse_vocab = {idx: word for word, idx in self.vocab.items()}
        logger.info("Loaded vocabulary from %s", filepath)
