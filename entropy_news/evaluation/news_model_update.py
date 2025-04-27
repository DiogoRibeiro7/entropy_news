# entropy_news/evaluation/news_model_update.py

import torch
from torch.utils.data import DataLoader, Dataset

class NewsModelUpdateCalculator:
    def __init__(self, old_model: torch.nn.Module, new_model: torch.nn.Module):
        self.old_model = old_model
        self.new_model = new_model
        self.device = new_model.device

    def compute_entropies(self, old_dataset: Dataset, new_dataset: Dataset, batch_size: int = 1) -> dict:
        """
        Decomposes total entropy change into:
        - ENT: Entropy of new model on new dataset
        - ENT_news: Entropy of old model on new dataset
        - ENT_model: Difference due to model update
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

class EntropyCalculator:
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.device = model.device

    def compute_entropy(self, dataset: Dataset, batch_size: int = 1) -> float:
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

# Exemplo de uso:
# update_calculator = NewsModelUpdateCalculator(old_model, new_model)
# entropies = update_calculator.compute_entropies(old_dataset, new_dataset)
