# entropy_news/evaluation/news_model_update.py

import torch
from torch.utils.data import Dataset
from .entropy_calculator import EntropyCalculator

class NewsModelUpdateCalculator:
    """Decompose entropy into news and model components."""

    def __init__(self, old_model: torch.nn.Module, new_model: torch.nn.Module):
        """Store ``old_model`` and ``new_model`` for comparison."""
        self.old_model = old_model
        self.new_model = new_model
        self.device = new_model.device

    def compute_entropies(
        self, new_dataset: Dataset, batch_size: int = 1
    ) -> dict[str, float]:
        """Compute entropy decomposition using both models.

        Args:
            new_dataset: Dataset containing tokenised sequences.
            batch_size: Size of each evaluation batch.

        Returns:
            Dictionary with keys ``ENT``, ``ENT_news`` and ``ENT_model``.
        """
        # Compute entropy under both the old and the updated model
        old_entropy_calculator = EntropyCalculator(self.old_model)
        new_entropy_calculator = EntropyCalculator(self.new_model)

        ENT_news = old_entropy_calculator.compute_entropy(new_dataset, batch_size=batch_size)
        ENT = new_entropy_calculator.compute_entropy(new_dataset, batch_size=batch_size)
        ENT_model = ENT - ENT_news

        return {
            "ENT": ENT,
            "ENT_news": ENT_news,
            "ENT_model": ENT_model,
        }


# Example usage:
# update_calculator = NewsModelUpdateCalculator(old_model, new_model)
# entropies = update_calculator.compute_entropies(new_dataset)
