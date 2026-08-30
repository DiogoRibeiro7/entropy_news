import torch
from torch.utils.data import Dataset

from .entropy_calculator import EntropyCalculator
from entropy_news.utils import get_device


class NewsModelUpdateCalculator:
    """Compare entropy levels before and after a model update.

    This class is deliberately *not* the paper's ENT/ENT_NEWS/ENT_MODEL
    decomposition.  Those quantities require a 12-month lagged model and are
    implemented in :mod:`entropy_news.paper_reproduction` and
    :mod:`entropy_news.paper_rolling`.
    """

    def __init__(
        self,
        old_model: torch.nn.Module,
        new_model: torch.nn.Module,
        device: torch.device | None = None,
    ) -> None:
        self.device = device or get_device()
        self.old_model = old_model.to(self.device)
        self.new_model = new_model.to(self.device)

    def compute_entropies(
        self,
        new_dataset: Dataset,
        batch_size: int = 1,
        show_progress: bool = False,
    ) -> dict[str, float]:
        """Return one-month entropy-level diagnostics for two models."""

        baseline_entropy = EntropyCalculator(
            self.old_model, device=self.device
        ).compute_entropy(
            new_dataset, batch_size=batch_size, show_progress=show_progress
        )
        updated_entropy = EntropyCalculator(
            self.new_model, device=self.device
        ).compute_entropy(
            new_dataset, batch_size=batch_size, show_progress=show_progress
        )
        return {
            "baseline_entropy": baseline_entropy,
            "updated_entropy": updated_entropy,
            "model_update_delta": updated_entropy - baseline_entropy,
        }
