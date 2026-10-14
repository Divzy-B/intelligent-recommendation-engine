"""Transparent baselines used to contextualize neural-model performance."""
from __future__ import annotations
import numpy as np


class PopularityRecommender:
    def fit(self, train_events, num_items: int):
        self.scores = np.bincount([item for _, item in train_events], minlength=num_items).astype(np.float32)
        return self

    def score(self, _user: int, _history: list[int]) -> np.ndarray:
        return self.scores.copy()


class ItemCFRecommender:
    """Cosine-normalized item co-occurrence recommender."""
    def fit(self, histories: dict[int, list[int]], num_items: int):
        cooccurrence = np.zeros((num_items, num_items), dtype=np.float32)
        for history in histories.values():
            unique = np.unique(history[-50:])
            cooccurrence[np.ix_(unique, unique)] += 1
        np.fill_diagonal(cooccurrence, 0)
        norm = np.sqrt(np.maximum(np.diag(cooccurrence + np.diag(np.diag(cooccurrence))), 1))
        # Item frequency is the row sum before normalization; co-occurrence diagonal was removed.
        frequency = np.maximum((cooccurrence > 0).sum(axis=1), 1)
        self.similarity = cooccurrence / np.sqrt(frequency[:, None] * frequency[None, :])
        self.num_items = num_items
        return self

    def score(self, _user: int, history: list[int]) -> np.ndarray:
        if not history:
            return np.zeros(self.num_items, dtype=np.float32)
        return self.similarity[np.unique(history[-30:])].mean(axis=0)


class ImplicitMF:
    """A compact BPR matrix-factorization baseline trained with sampled negatives."""
    def __init__(self, factors: int = 32, epochs: int = 3, learning_rate: float = 0.05, seed: int = 42):
        self.factors, self.epochs, self.learning_rate, self.seed = factors, epochs, learning_rate, seed

    def fit(self, train_events, num_users: int, num_items: int):
        rng = np.random.default_rng(self.seed)
        self.user_factors = rng.normal(0, 0.05, (num_users, self.factors)).astype(np.float32)
        self.item_factors = rng.normal(0, 0.05, (num_items, self.factors)).astype(np.float32)
        known = {user: set() for user in range(num_users)}
        for user, item in train_events:
            known[user].add(item)
        triples = np.asarray(train_events, dtype=np.int64)
        for _ in range(self.epochs):
            rng.shuffle(triples)
            for user, positive in triples:
                negative = rng.integers(num_items)
                while negative in known[int(user)]:
                    negative = rng.integers(num_items)
                u, pos, neg = self.user_factors[user].copy(), self.item_factors[positive].copy(), self.item_factors[negative].copy()
                sigmoid = 1 / (1 + np.exp(np.clip(np.dot(u, pos - neg), -30, 30)))
                self.user_factors[user] += self.learning_rate * (sigmoid * (pos - neg) - 1e-4 * u)
                self.item_factors[positive] += self.learning_rate * (sigmoid * u - 1e-4 * pos)
                self.item_factors[negative] += self.learning_rate * (-sigmoid * u - 1e-4 * neg)
        return self

    def score(self, user: int, _history: list[int]) -> np.ndarray:
        return self.item_factors @ self.user_factors[user]
