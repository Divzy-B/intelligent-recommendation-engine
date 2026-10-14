from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F


class ItemTower(nn.Module):
    def __init__(self, num_items: int, genre_features: torch.Tensor, dim: int, hidden: int):
        super().__init__()
        self.id_embedding = nn.Embedding(num_items, dim)
        self.register_buffer("genres", genre_features)
        self.genre_projection = nn.Linear(genre_features.shape[1], dim)
        self.network = nn.Sequential(nn.Linear(dim * 2, hidden), nn.ReLU(), nn.Linear(hidden, dim))

    def forward(self, item_ids: torch.Tensor) -> torch.Tensor:
        id_vector = self.id_embedding(item_ids)
        genre_vector = self.genre_projection(self.genres[item_ids])
        return F.normalize(self.network(torch.cat([id_vector, genre_vector], dim=-1)), dim=-1)


class TwoTower(nn.Module):
    """Separate, L2-normalized user and item encoders for contrastive retrieval."""
    def __init__(self, num_users: int, num_items: int, genre_features: torch.Tensor, dim: int = 48, hidden: int = 96):
        super().__init__()
        self.item_tower = ItemTower(num_items, genre_features, dim, hidden)
        self.user_embedding = nn.Embedding(num_users, dim)
        self.user_network = nn.Sequential(nn.Linear(dim * 2, hidden), nn.ReLU(), nn.Linear(hidden, dim))

    def encode_items(self, item_ids: torch.Tensor) -> torch.Tensor:
        return self.item_tower(item_ids)

    def encode_users(self, user_ids: torch.Tensor, histories: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        history_vectors = self.encode_items(histories)
        pooled = (history_vectors * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)
        identity = self.user_embedding(user_ids)
        return F.normalize(self.user_network(torch.cat([identity, pooled], dim=-1)), dim=-1)

    def forward(self, user_ids: torch.Tensor, histories: torch.Tensor, mask: torch.Tensor, item_ids: torch.Tensor):
        return self.encode_users(user_ids, histories, mask), self.encode_items(item_ids)
