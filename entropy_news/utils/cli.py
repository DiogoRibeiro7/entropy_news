"""Helper functions for CLI scripts."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

try:  # pragma: no cover - torch may not be installed during some checks
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore

from .device import get_device
from .io import load_texts

if TYPE_CHECKING:  # pragma: no cover - imported for type checking only
    from ..model.factory import SupportsForward

from ..model.config import ModelConfig
from ..model.factory import ModelFactory

UNSAFE_ENV_VAR = "ENTROPY_NEWS_ALLOW_UNSAFE_LOAD"


def _env_flag_enabled(value: str | None) -> bool:
    """Return ``True`` when an environment toggle string represents truth."""

    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class ConfigDefaults:
    """Baseline hyperparameters used when a CLI omits overrides."""

    architecture: str = "lstm"
    vocab_size: int = 10000
    embed_dim: int = 100
    hidden_dim: int = 16
    num_heads: int = 2
    ff_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.1


@dataclass(frozen=True)
class ConfigOverrides:
    """Container for CLI supplied model overrides."""

    architecture: str | None = None
    vocab_size: int | None = None
    embed_dim: int | None = None
    hidden_dim: int | None = None
    num_heads: int | None = None
    ff_dim: int | None = None
    num_layers: int | None = None
    dropout: float | None = None


def load_base_config(path: str | None) -> ModelConfig | None:
    """Load a :class:`ModelConfig` from ``path`` when provided."""

    if not path:
        return None
    return ModelConfig.load(path)


def resolve_model_config(
    *,
    base_config: ModelConfig | None,
    overrides: ConfigOverrides,
    defaults: ConfigDefaults | None = None,
) -> ModelConfig:
    """Merge CLI overrides with ``base_config`` and ``defaults``.

    Args:
        base_config: Optional configuration loaded from disk.
        overrides: Values supplied directly by the CLI.
        defaults: Baseline hyperparameters when neither overrides nor
            ``base_config`` provide a value. Falls back to :class:`ConfigDefaults`.

    Returns:
        Fully resolved :class:`ModelConfig`.
    """

    defaults = defaults or ConfigDefaults()

    def choose(attr: str) -> object:
        value = getattr(overrides, attr)
        if value is not None:
            return value
        if base_config is not None:
            return getattr(base_config, attr)
        return getattr(defaults, attr)

    config = ModelConfig(
        architecture=str(choose("architecture")),
        vocab_size=int(choose("vocab_size")),
        embed_dim=int(choose("embed_dim")),
        hidden_dim=int(choose("hidden_dim")),
        num_heads=int(choose("num_heads")),
        ff_dim=int(choose("ff_dim")),
        num_layers=int(choose("num_layers")),
        dropout=float(choose("dropout")),
    )
    return config


def load_model_and_vocab(
    vocab_path: str,
    model_path: str,
    embed_dim: int | None = None,
    hidden_dim: int | None = None,
    num_layers: int | None = None,
    dropout: float | None = None,
    *,
    config_path: str | None = None,
    config: ModelConfig | None = None,
    device: torch.device | None = None,
    allow_unsafe_load: bool = False,
) -> tuple[TextPreprocessor, SupportsForward, torch.device, ModelConfig]:
    """Load a saved vocabulary and model.

    Args:
        vocab_path: Location of the saved vocabulary JSON file.
        model_path: Location of the model ``state_dict``.
        embed_dim: Embedding dimension of the legacy model configuration.
        hidden_dim: Hidden dimension of the legacy model configuration.
        num_layers: Number of layers for LSTM based models.
        dropout: Dropout rate between recurrent layers.
        config_path: Optional path to a serialized :class:`ModelConfig`.
        config: Directly provided configuration object to reuse.
        device: Optional device to place the model on.
        allow_unsafe_load: When ``True`` the loader may fall back to
            ``torch.load(..., weights_only=False)`` for legacy checkpoints.

    Returns:
        Tuple containing the text preprocessor, loaded model, resolved device
        and the configuration used to instantiate the model.

    Raises:
        FileNotFoundError: If either ``vocab_path`` or ``model_path`` is missing.
        OSError: If loading of the vocabulary or model fails.
        ValueError: When insufficient information is provided to construct a
            configuration.
    """

    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"Vocabulary file not found: {vocab_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    from ..data import TextPreprocessor
    from ..model import ModelConfig, ModelFactory

    preprocessor = TextPreprocessor()
    try:
        preprocessor.load_vocab(vocab_path)
    except Exception as exc:  # noqa: BLE001 - propagate friendly message
        raise OSError(f"Failed to load vocabulary from {vocab_path}: {exc}") from exc

    base_config = config
    if base_config is None and config_path is not None:
        base_config = ModelConfig.load(config_path)

    if base_config is None:
        missing = [
            name
            for name, value in (
                ("embed_dim", embed_dim),
                ("hidden_dim", hidden_dim),
                ("num_layers", num_layers),
                ("dropout", dropout),
            )
            if value is None
        ]
        if missing:
            raise ValueError(
                "Configuration details missing: " + ", ".join(missing)
            )

    overrides = ConfigOverrides(
        vocab_size=len(preprocessor.vocab),
        embed_dim=embed_dim if base_config is None else None,
        hidden_dim=hidden_dim if base_config is None else None,
        num_layers=num_layers if base_config is None else None,
        dropout=dropout if base_config is None else None,
    )
    resolved_config = resolve_model_config(
        base_config=base_config,
        overrides=overrides,
        defaults=ConfigDefaults(vocab_size=len(preprocessor.vocab)),
    )

    device = device or get_device()
    model = ModelFactory.create(resolved_config, embedding_matrix=None).to(device)
    load_kwargs = {"map_location": device}
    logger = logging.getLogger(__name__)
    allow_env = _env_flag_enabled(os.getenv(UNSAFE_ENV_VAR))
    allow_unsafe = allow_unsafe_load or allow_env

    try:
        try:
            state_dict = torch.load(model_path, weights_only=True, **load_kwargs)
        except TypeError:
            logger.warning(
                "torch.load() does not support weights_only on this installation; "
                "falling back to the legacy loader. Ensure checkpoints are trusted.",
            )
            state_dict = torch.load(model_path, **load_kwargs)
    except (RuntimeError, ValueError) as exc:
        if allow_unsafe:
            logger.warning(
                "Unsafe checkpoint loading enabled; falling back to "
                "torch.load(..., weights_only=False). Only use trusted checkpoints.",
            )
            try:
                try:
                    state_dict = torch.load(
                        model_path, weights_only=False, **load_kwargs
                    )
                except TypeError:
                    state_dict = torch.load(model_path, **load_kwargs)
            except Exception as unsafe_exc:  # noqa: BLE001
                raise OSError(
                    f"Failed to load model from {model_path}: {unsafe_exc}"
                ) from unsafe_exc
        else:
            raise OSError(
                "Failed to load model safely. Re-run with --allow-unsafe-load or set "
                f"{UNSAFE_ENV_VAR}=1 after verifying the checkpoint is trusted: {exc}"
            ) from exc
    except Exception as exc:  # noqa: BLE001
        raise OSError(f"Failed to load model from {model_path}: {exc}") from exc
    model.load_state_dict(state_dict)
    return preprocessor, model, device, resolved_config


def load_encoded_dataset(
    preprocessor: TextPreprocessor,
    data_path: str,
    seq_len: int,
    lazy: bool,
    texts: list[str] | None = None,
) -> NewsDataset:
    """Read ``data_path`` and return an encoded ``NewsDataset``.

    Args:
        preprocessor: Preprocessor with an existing vocabulary.
        data_path: Text file containing one document per line.
        seq_len: Sequence length for dataset examples.
        lazy: Whether to defer padding to retrieval time.
        texts: Optional list of pre-loaded texts; when provided the file is not
            read from disk again.

    Returns:
        ``NewsDataset`` built from the provided data.

    Raises:
        OSError: If ``data_path`` is missing or unreadable.
    """
    from ..data import NewsDataset

    if texts is None:
        try:
            texts = load_texts(data_path)
        except (OSError, ValueError) as exc:
            raise OSError(str(exc)) from exc
    encoded = [preprocessor.encode(t) for t in texts]
    return NewsDataset(encoded, seq_len=seq_len, in_memory=not lazy)
