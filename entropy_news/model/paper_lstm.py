"""Exact LSTM parameterisation reported in *New News is Bad News*."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from entropy_news.types import EmbeddingMatrix


class PaperEntropyLSTM(nn.Module):
    """Paper-only LSTM with one bias vector per gate block.

    PyTorch's :class:`~torch.nn.LSTM` stores two trainable bias vectors per
    layer. The paper's appendix specifies the standard four input matrices,
    four recurrent matrices and four bias vectors, which corresponds to one
    concatenated bias vector of length ``4 * hidden_dim``. This implementation
    matches that parameterisation while keeping the GloVe embedding frozen.
    """

    def __init__(
        self,
        predictive_vocab_size: int,
        *,
        embed_dim: int = 100,
        hidden_dim: int = 16,
        embedding_matrix: EmbeddingMatrix | None = None,
        padding_idx: int | None = None,
    ) -> None:
        super().__init__()
        if predictive_vocab_size <= 1:
            raise ValueError("predictive_vocab_size must exceed 1")
        self.predictive_vocab_size = predictive_vocab_size
        self.hidden_dim = hidden_dim
        self.padding_idx = padding_idx

        if embedding_matrix is not None:
            input_vocab_size = int(embedding_matrix.shape[0])
        elif padding_idx is not None:
            input_vocab_size = max(predictive_vocab_size, padding_idx + 1)
        else:
            input_vocab_size = predictive_vocab_size

        self.embedding = nn.Embedding(
            input_vocab_size,
            embed_dim,
            padding_idx=padding_idx,
        )
        self.embed = self.embedding
        if embedding_matrix is not None:
            matrix = torch.as_tensor(embedding_matrix, dtype=self.embedding.weight.dtype)
            if tuple(matrix.shape) != tuple(self.embedding.weight.shape):
                raise ValueError(
                    "embedding matrix shape does not match paper model input vocabulary"
                )
            self.embedding.weight.data.copy_(matrix)
            self.embedding.weight.requires_grad = False

        self.weight_ih = nn.Parameter(torch.empty(4 * hidden_dim, embed_dim))
        self.weight_hh = nn.Parameter(torch.empty(4 * hidden_dim, hidden_dim))
        self.bias = nn.Parameter(torch.zeros(4 * hidden_dim))
        self.fc = nn.Linear(hidden_dim, predictive_vocab_size)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = 1.0 / math.sqrt(self.hidden_dim)
        nn.init.uniform_(self.weight_ih, -bound, bound)
        nn.init.uniform_(self.weight_hh, -bound, bound)
        nn.init.uniform_(self.bias, -bound, bound)
        nn.init.uniform_(self.fc.weight, -bound, bound)
        nn.init.uniform_(self.fc.bias, -bound, bound)

    def forward_embeddings(
        self,
        embeddings: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Run the recurrent cell over precomputed input vectors."""
        batch_size, steps, _ = embeddings.shape
        if state is None:
            h = torch.zeros(
                batch_size,
                self.hidden_dim,
                dtype=embeddings.dtype,
                device=embeddings.device,
            )
            c = torch.zeros_like(h)
        else:
            h, c = state

        outputs: list[torch.Tensor] = []
        for step in range(steps):
            gates = (
                torch.nn.functional.linear(embeddings[:, step, :], self.weight_ih)
                + torch.nn.functional.linear(h, self.weight_hh, self.bias)
            )
            input_gate, forget_gate, candidate, output_gate = gates.chunk(4, dim=-1)
            input_gate = torch.sigmoid(input_gate)
            forget_gate = torch.sigmoid(forget_gate)
            candidate = torch.tanh(candidate)
            output_gate = torch.sigmoid(output_gate)
            c = forget_gate * c + input_gate * candidate
            h = output_gate * torch.tanh(c)
            outputs.append(h)

        output = torch.stack(outputs, dim=1)
        return output, (h, c)

    def forward_with_state(
        self,
        x: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        output, next_state = self.forward_embeddings(self.embedding(x), state)
        return self.fc(output), next_state

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward_with_state(x)
        return logits

    def trainable_parameter_count(self) -> int:
        """Return the number of trainable parameters, excluding frozen GloVe."""
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
