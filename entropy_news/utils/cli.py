"""Helper functions for CLI scripts."""

from __future__ import annotations

import os

try:  # pragma: no cover - torch may not be installed during some checks
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore

from .device import get_device
from .io import load_texts


def load_model_and_vocab(
    vocab_path: str,
    model_path: str,
    embed_dim: int,
    hidden_dim: int,
    num_layers: int,
    dropout: float,
    device: torch.device | None = None,
) -> tuple[TextPreprocessor, EntropyLSTM, torch.device]:
    """Load a saved vocabulary and model.

    Args:
        vocab_path: Location of the saved vocabulary JSON file.
        model_path: Location of the model ``state_dict``.
        embed_dim: Embedding dimension of the model.
        hidden_dim: Hidden dimension of the model.
        num_layers: Number of LSTM layers.
        dropout: Dropout rate between LSTM layers.
        device: Optional device to place the model on.

    Returns:
        Tuple containing the text preprocessor, loaded model and device used.

    Raises:
        FileNotFoundError: If either ``vocab_path`` or ``model_path`` is missing.
        OSError: If loading of the vocabulary or model fails.
    """
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"Vocabulary file not found: {vocab_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    from ..data import TextPreprocessor
    from ..model import EntropyLSTM

    preprocessor = TextPreprocessor()
    try:
        preprocessor.load_vocab(vocab_path)
    except Exception as exc:  # noqa: BLE001 - propagate friendly message
        raise OSError(f"Failed to load vocabulary from {vocab_path}: {exc}") from exc

    device = device or get_device()
    model = EntropyLSTM(
        vocab_size=len(preprocessor.vocab),
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)
    try:
        state_dict = torch.load(model_path, map_location=device)
    except Exception as exc:  # noqa: BLE001
        raise OSError(f"Failed to load model from {model_path}: {exc}") from exc
    model.load_state_dict(state_dict)
    return preprocessor, model, device


def load_encoded_dataset(
    preprocessor: TextPreprocessor,
    data_path: str,
    seq_len: int,
    lazy: bool,
) -> NewsDataset:
    """Read ``data_path`` and return an encoded ``NewsDataset``.

    Args:
        preprocessor: Preprocessor with an existing vocabulary.
        data_path: Text file containing one document per line.
        seq_len: Sequence length for dataset examples.
        lazy: Whether to defer padding to retrieval time.

    Returns:
        ``NewsDataset`` built from the provided data.

    Raises:
        OSError: If ``data_path`` is missing or unreadable.
    """
    from ..data import NewsDataset

    try:
        texts = load_texts(data_path)
    except (OSError, ValueError) as exc:
        raise OSError(str(exc)) from exc
    encoded = [preprocessor.encode(t) for t in texts]
    return NewsDataset(encoded, seq_len=seq_len, in_memory=not lazy)
