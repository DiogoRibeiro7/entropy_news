# entropy_news/evaluation/news_model_update.py

import torch
from torch.utils.data import DataLoader, Dataset
from entropy_calculator import EntropyCalculator

class NewsModelUpdateCalculator:
    def __init__(self, old_model: torch.nn.Module, new_model: torch.nn.Module):
        self.old_model = old_model
        self.new_model = new_model
        self.device = new_model.device

    def compute_entropies(self, new_dataset: Dataset, batch_size: int = 1) -> dict:
        """Decompose the entropy change after a model update.

        Parameters
        ----------
        new_dataset : Dataset
            Dataset containing the new data on which the models will be
            evaluated.
        batch_size : int, optional
            Batch size for the entropy computation, by default 1.

        Returns
        -------
        dict
            Dictionary with the keys ``ENT``, ``ENT_news`` and ``ENT_model``.
        """
        old_entropy_calculator = EntropyCalculator(self.old_model)
        new_entropy_calculator = EntropyCalculator(self.new_model)

        ENT_news = old_entropy_calculator.compute_entropy(new_dataset, batch_size=batch_size)
        ENT = new_entropy_calculator.compute_entropy(new_dataset, batch_size=batch_size)
        ENT_model = ENT - ENT_news

        return {
            'ENT': ENT,
            'ENT_news': ENT_news,
            'ENT_model': ENT_model
        }


# Example usage:
# update_calculator = NewsModelUpdateCalculator(old_model, new_model)
# entropies = update_calculator.compute_entropies(new_dataset)
