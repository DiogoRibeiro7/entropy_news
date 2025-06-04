# entropy_news/evaluation/entropy_calculator.py

import torch
from torch.utils.data import DataLoader, Dataset

from ..utils import perplexity

class EntropyCalculator:
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.device = model.device

    def compute_entropy(self, dataset: Dataset, batch_size: int = 1) -> float:
        """Return cross-entropy of ``model`` on ``dataset``.

        Parameters
        ----------
        dataset : Dataset
            Dataset yielding input and target token sequences.
        batch_size : int, optional
            Size of each evaluation batch, by default ``1``.

        Returns
        -------
        float
            Average token-wise cross-entropy ignoring padding tokens.
        """
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        self.model.eval()

        total_log_prob = 0.0
        total_tokens = 0

        with torch.no_grad():
            for x_batch, y_batch in loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                logits = self.model(x_batch)
                probs = torch.softmax(logits, dim=-1)

                token_probs = probs.gather(2, y_batch.unsqueeze(-1)).squeeze()
                log_token_probs = torch.log(token_probs + 1e-10)

                mask = (y_batch != 0)  # Ignore <PAD> tokens
                total_log_prob += (log_token_probs * mask).sum().item()
                total_tokens += mask.sum().item()

        entropy = -total_log_prob / total_tokens if total_tokens > 0 else float('inf')
        return entropy

    def compute_perplexity(self, dataset: Dataset, batch_size: int = 1) -> float:
        """Return perplexity computed from cross-entropy."""
        ent = self.compute_entropy(dataset, batch_size=batch_size)
        return perplexity(ent)

# Exemplo de uso:
# calculator = EntropyCalculator(model)
# entropy_value = calculator.compute_entropy(dataset)
