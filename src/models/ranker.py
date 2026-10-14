import torch
from torch import nn


class NeuralRanker(nn.Module):
    """Small cross-feature MLP that re-scores retrieved candidates."""
    def __init__(self, embedding_dim: int, hidden_dim: int = 96):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(embedding_dim * 3 + 1, hidden_dim), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, users: torch.Tensor, items: torch.Tensor, retrieval_scores: torch.Tensor) -> torch.Tensor:
        features = torch.cat([users, items, users * items, retrieval_scores.unsqueeze(-1)], dim=-1)
        return self.network(features).squeeze(-1)
