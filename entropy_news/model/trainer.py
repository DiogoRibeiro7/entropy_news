# entropy_news/model/trainer.py

import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from typing import Optional
from tqdm import tqdm
from entropy_news.utils import get_device

logger = logging.getLogger(__name__)

class Trainer:
    """Utility class to train and fine-tune language models."""

    def __init__(
        self, model: nn.Module, learning_rate: float = 0.001, device: torch.device | None = None
    ) -> None:
        """Initialise the trainer.

        Args:
            model: Model to optimise.
            learning_rate: Step size for the Adam optimiser.
            device: Optional ``torch`` device for computation.
        """
        self.device = device or get_device()
        self.model = model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)

    def train(
        self,
        dataset: Dataset,
        epochs: int = 50,
        batch_size: int = 128,
        val_dataset: Optional[Dataset] = None,
        early_stopping: bool = False,
        patience: int = 5,
        start_epoch: int = 0,
        checkpoint_path: str | Path | None = None,
        show_progress: bool = True,
    ) -> None:
        """Train ``self.model`` using ``dataset``.

        Args:
            dataset: Dataset of token sequences.
            epochs: Number of training epochs.
            batch_size: Samples per training batch.
            val_dataset: Optional dataset for validation loss calculation.
            early_stopping: Whether to stop when validation loss stops
                improving.
            patience: Epochs to wait for improvement before stopping.
            start_epoch: Epoch to resume training from (zero-indexed).
            checkpoint_path: Optional file path to store checkpoints after each
                epoch.
            show_progress: Whether to display a progress bar for each epoch.
        """
        self._ensure_trainable_embeddings()

        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        val_loader = (
            DataLoader(val_dataset, batch_size=batch_size) if val_dataset else None
        )
        self.model.train()

        best_val_loss = float("inf")
        epochs_no_improve = 0

        for epoch in range(start_epoch + 1, epochs + 1):
            total_loss = 0
            loop = (
                tqdm(loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
                if show_progress
                else loader
            )
            for x_batch, y_batch in loop:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                self.optimizer.zero_grad()
                logits = self.model(x_batch)
                loss = self.criterion(
                    logits.view(-1, logits.size(-1)), y_batch.view(-1)
                )
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                if show_progress:
                    loop.set_postfix(loss=loss.item())

            avg_loss = total_loss / len(loader)
            val_loss = (
                self.evaluate(val_loader) if val_loader is not None else None
            )
            if epoch % 10 == 0 or epoch == 1 or epoch == epochs:
                msg = f"Epoch {epoch}/{epochs} - Loss: {avg_loss:.4f}"
                if val_loss is not None:
                    msg += f" - Val Loss: {val_loss:.4f}"
                logger.info(msg)

            if val_loss is not None and early_stopping:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    logger.info(
                        f"Early stopping at epoch {epoch} (best val loss {best_val_loss:.4f})"
                    )
                    break

            if checkpoint_path:
                self.save_checkpoint(checkpoint_path, epoch)

    def _ensure_trainable_embeddings(self) -> None:
        """Jitter embedding weights when they are entirely zero."""

        embed = getattr(self.model, "embed", None) or getattr(
            self.model, "embedding", None
        )
        weight = getattr(embed, "weight", None) if embed is not None else None
        if weight is None or not isinstance(weight, torch.Tensor):
            return
        if not weight.requires_grad:
            return
        if torch.count_nonzero(weight).item() == 0:
            with torch.no_grad():
                weight.uniform_(-1e-3, 1e-3)

    def fine_tune(
        self,
        new_dataset: Dataset,
        epochs: int = 50,
        batch_size: int = 128,
        show_progress: bool = True,
    ) -> None:
        """Continue training ``self.model`` on ``new_dataset``.

        Args:
            new_dataset: Additional data used for fine-tuning.
            epochs: Number of fine-tuning epochs.
            batch_size: Samples per batch.
            show_progress: Whether to display a progress bar during training.
        """
        logger.info("Fine-tuning model on new data...")
        self.train(
            new_dataset,
            epochs=epochs,
            batch_size=batch_size,
            show_progress=show_progress,
        )

    def evaluate(self, loader: DataLoader) -> float:
        """Compute average loss on ``loader`` without updating gradients.

        Args:
            loader: Data loader used for evaluation.

        Returns:
            Average loss across all batches.
        """
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch in loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                logits = self.model(x_batch)
                loss = self.criterion(logits.view(-1, logits.size(-1)), y_batch.view(-1))
                total_loss += loss.item()
        self.model.train()
        return total_loss / len(loader)

    def save_checkpoint(self, path: str | Path, epoch: int) -> None:
        """Persist model and optimiser state to ``path``.

        Args:
            path: Destination file for the checkpoint.
            epoch: Epoch number to record within the checkpoint.
        """
        path = Path(path)
        if path.parent:
            path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "epoch": epoch,
            },
            path,
        )

    def load_checkpoint(self, path: str | Path) -> int:
        """Load model and optimiser state from ``path``.

        Args:
            path: Source checkpoint file.

        Returns:
            Epoch number stored in the checkpoint.
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        return int(checkpoint.get("epoch", 0))
