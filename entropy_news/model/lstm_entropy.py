# entropy_news/model/lstm_entropy.py

import torch
import torch.nn as nn

class EntropyLSTM(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 100, hidden_dim: int = 16, embedding_matrix=None):
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if embedding_matrix is not None:
            self.embedding.weight.data.copy_(torch.Tensor(embedding_matrix))
            self.embedding.weight.requires_grad = False  # Freezes embeddings

        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = x.to(self.device)
        emb = self.embedding(x)
        output, _ = self.lstm(emb)
        logits = self.fc(output)
        return logits

# Exemplo de uso:
# model = EntropyLSTM(vocab_size=len(vocab), embed_dim=100, hidden_dim=16, embedding_matrix=preprocessor.embedding_matrix)
# model = model.to(model.device)
