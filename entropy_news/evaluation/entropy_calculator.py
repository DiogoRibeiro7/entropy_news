# entropy_news/evaluation/entropy_calculator.py

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from ..utils import perplexity, get_device

class EntropyCalculator:
    """Compute cross-entropy and perplexity for a language model."""

    def __init__(
        self, model: torch.nn.Module, device: torch.device | None = None
    ) -> None:
        """Initialise the calculator with a model.

        Args:
            model: Language model used for prediction.
            device: Optional ``torch`` device for computation.
        """
        self.device = device or get_device()
        self.model = model.to(self.device)

    def compute_entropy(
        self, dataset: Dataset, batch_size: int = 1, show_progress: bool = False
    ) -> float:
        """Calculate the average cross-entropy on ``dataset``.

        Args:
            dataset: Dataset yielding input and target token sequences.
            batch_size: Number of samples per evaluation batch.
            show_progress: Whether to display a progress bar while iterating.

        Returns:
            Average token-wise cross-entropy, ignoring padding tokens.
        """
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        self.model.eval()

        total_log_prob = 0.0
        total_tokens = 0

        with torch.no_grad():
            iterator = tqdm(loader, desc="Evaluating", leave=False) if show_progress else loader
            for x_batch, y_batch in iterator:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                logits = self.model(x_batch)
                log_probs = torch.log_softmax(logits, dim=-1)
                token_log_probs = log_probs.gather(2, y_batch.unsqueeze(-1)).squeeze()

                mask = (y_batch != 0)  # Ignore <PAD> tokens
                total_log_prob += (token_log_probs * mask).sum().item()
                total_tokens += mask.sum().item()

        entropy = -total_log_prob / total_tokens if total_tokens > 0 else float("inf")
        return entropy

    def compute_perplexity(
        self, dataset: Dataset, batch_size: int = 1, show_progress: bool = False
    ) -> float:
        """Compute perplexity from cross-entropy on ``dataset``.

        Args:
            dataset: Dataset for evaluation.
            batch_size: Number of samples per evaluation batch.
            show_progress: Whether to display a progress bar while iterating.

        Returns:
            Perplexity value.
        """
        ent = self.compute_entropy(
            dataset, batch_size=batch_size, show_progress=show_progress
        )
        return perplexity(ent)
