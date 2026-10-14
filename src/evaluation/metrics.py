"""Ranking metrics for a leave-one-out, full-catalog evaluation protocol."""
from __future__ import annotations
import numpy as np


def _dcg(rank: int) -> float:
    return 1.0 / np.log2(rank + 2)


def evaluate_scores(score_function, test_events, histories: dict[int, list[int]], num_items: int, ks=(5, 10)) -> dict:
    hits = {k: [] for k in ks}
    ndcgs = {k: [] for k in ks}
    recommended = set()
    for user, target in test_events:
        scores = np.asarray(score_function(user, histories[user]), dtype=np.float64).copy()
        scores[histories[user]] = -np.inf
        top = np.argsort(-scores)[: max(ks)]
        recommended.update(top.tolist())
        for k in ks:
            positions = np.where(top[:k] == target)[0]
            hit = len(positions) > 0
            hits[k].append(float(hit))
            ndcgs[k].append(_dcg(int(positions[0])) if hit else 0.0)
    result = {f"recall@{k}": float(np.mean(hits[k])) for k in ks}
    result.update({f"ndcg@{k}": float(np.mean(ndcgs[k])) for k in ks})
    result["hit_rate@10"] = result.get("recall@10", result[f"recall@{max(ks)}"])
    result["catalog_coverage"] = len(recommended) / num_items
    result["users_evaluated"] = len(test_events)
    return result
