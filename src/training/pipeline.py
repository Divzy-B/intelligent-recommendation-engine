"""End-to-end training entry point for the laptop-scale MovieLens implementation."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import logging
import pickle
from pathlib import Path
import random
import os
from time import perf_counter

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from src.config import Settings
from src.data.movielens import load_movielens, make_synthetic_ratings, prepare_data
from src.evaluation.metrics import evaluate_scores
from src.models import NeuralRanker, TwoTower
from src.models.baselines import ImplicitMF, ItemCFRecommender, PopularityRecommender
from src.retrieval import FaissRetriever
from src.training.tracking import log_run

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(max(1, int(os.getenv("TORCH_NUM_THREADS", "1"))))


class SequenceDataset(Dataset):
    def __init__(self, samples): self.samples = samples
    def __len__(self): return len(self.samples)
    def __getitem__(self, index): return self.samples[index]


def collate_sequences(batch, max_history: int):
    users = torch.tensor([row[0] for row in batch], dtype=torch.long)
    targets = torch.tensor([row[2] for row in batch], dtype=torch.long)
    histories = torch.zeros((len(batch), max_history), dtype=torch.long)
    mask = torch.zeros((len(batch), max_history), dtype=torch.float32)
    for row, history, active in zip(batch, histories, mask):
        values = row[1][-max_history:]
        if values:
            history[-len(values):] = torch.tensor(values, dtype=torch.long)
            active[-len(values):] = 1
    return users, histories, mask, targets


def encode_user(model: TwoTower, user: int, history: list[int], settings: Settings) -> torch.Tensor:
    h = torch.zeros((1, settings.max_history), dtype=torch.long)
    mask = torch.zeros((1, settings.max_history), dtype=torch.float32)
    values = history[-settings.max_history:]
    if values:
        h[0, -len(values):] = torch.tensor(values)
        mask[0, -len(values):] = 1
    with torch.no_grad():
        return model.encode_users(torch.tensor([user]), h, mask).cpu()


def train_retrieval(model: TwoTower, samples, settings: Settings) -> list[float]:
    rng = np.random.default_rng(settings.seed)
    if len(samples) > settings.max_train_examples:
        samples = [samples[i] for i in rng.choice(len(samples), settings.max_train_examples, replace=False)]
    loader = DataLoader(SequenceDataset(samples), batch_size=settings.batch_size, shuffle=True,
                        collate_fn=lambda batch: collate_sequences(batch, settings.max_history))
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay)
    losses = []
    model.train()
    for _ in range(settings.retrieval_epochs):
        epoch_losses = []
        for users, histories, mask, targets in loader:
            user_vectors, item_vectors = model(users, histories, mask, targets)
            # In-batch targets are negative samples for every other user. This contrastive
            # objective is reproducible and avoids materializing an all-item softmax.
            logits = user_vectors @ item_vectors.T / 0.10
            loss = F.cross_entropy(logits, torch.arange(len(users)))
            optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
            epoch_losses.append(float(loss.detach()))
        losses.append(float(np.mean(epoch_losses)))
    return losses


def train_ranker(model: TwoTower, ranker: NeuralRanker, samples, num_items: int, settings: Settings) -> list[float]:
    """Train pointwise relevance labels with explicitly sampled non-interacted items."""
    rng = np.random.default_rng(settings.seed)
    if len(samples) > min(25000, settings.max_train_examples):
        samples = [samples[i] for i in rng.choice(len(samples), min(25000, settings.max_train_examples), replace=False)]
    users, item_ids, labels, retrieval_scores = [], [], [], []
    model.eval()
    item_embeddings = encode_items_array(model)
    for user, history, target in samples:
        vector = encode_user(model, user, history, settings).numpy()[0]
        negatives = [int(rng.integers(num_items)), int(rng.integers(num_items))]
        while target in negatives:
            negatives = [int(rng.integers(num_items)), int(rng.integers(num_items))]
        chosen = [target, *negatives]
        users.extend([vector, vector, vector]); item_ids.extend(chosen)
        labels.extend([1.0, 0.0, 0.0]); retrieval_scores.extend([float(vector @ item_embeddings[item]) for item in chosen])
    user_tensor = torch.tensor(np.asarray(users), dtype=torch.float32)
    item_tensor = torch.tensor(item_embeddings[np.asarray(item_ids)], dtype=torch.float32)
    score_tensor = torch.tensor(np.asarray(retrieval_scores), dtype=torch.float32)
    label_tensor = torch.tensor(np.asarray(labels), dtype=torch.float32)
    loader = DataLoader(list(zip(user_tensor, item_tensor, score_tensor, label_tensor)), batch_size=settings.batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(ranker.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay)
    loss_function = nn.BCEWithLogitsLoss()
    losses = []
    ranker.train()
    for _ in range(settings.ranking_epochs):
        epoch_losses = []
        for u, i, retrieval, label in loader:
            loss = loss_function(ranker(u, i, retrieval), label)
            optimizer.zero_grad(); loss.backward(); optimizer.step(); epoch_losses.append(float(loss.detach()))
        losses.append(float(np.mean(epoch_losses)))
    return losses


def encode_items_array(model: TwoTower) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        ids = torch.arange(model.item_tower.id_embedding.num_embeddings)
        return model.encode_items(ids).cpu().numpy().astype("float32")


def make_neural_scorer(model: TwoTower, item_embeddings: np.ndarray, settings: Settings, ranker: NeuralRanker | None = None):
    items = torch.tensor(item_embeddings, dtype=torch.float32)
    def scorer(user: int, history: list[int]) -> np.ndarray:
        vector = encode_user(model, user, history, settings)
        retrieval = vector @ items.T
        if ranker is None:
            return retrieval.numpy()[0]
        ranker.eval()
        expanded = vector.repeat(len(items), 1)
        with torch.no_grad():
            return torch.sigmoid(ranker(expanded, items, retrieval[0])).numpy()
    return scorer


def _save_metrics_markdown(results: dict, path: Path) -> None:
    lines = ["# Executed evaluation results", "", "Metrics are generated by `python -m src.training.pipeline`; they are never hand-entered.", "", "| Model | Recall@5 | Recall@10 | NDCG@5 | NDCG@10 | Hit rate@10 | Coverage |", "|---|---:|---:|---:|---:|---:|---:|"]
    for name, metric in results.items():
        lines.append(f"| {name} | {metric['recall@5']:.4f} | {metric['recall@10']:.4f} | {metric['ndcg@5']:.4f} | {metric['ndcg@10']:.4f} | {metric['hit_rate@10']:.4f} | {metric['catalog_coverage']:.4f} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _save_visualization(report: dict, path: Path) -> None:
    """Write a compact, generated chart for the README rather than a static mockup."""
    import matplotlib.pyplot as plt
    metrics = report["metrics"]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(range(1, len(report["training"]["retrieval_losses"]) + 1), report["training"]["retrieval_losses"], marker="o", label="Two-Tower")
    axes[0].plot(range(1, len(report["training"]["ranking_losses"]) + 1), report["training"]["ranking_losses"], marker="o", label="Neural ranker")
    axes[0].set(title="Executed training loss", xlabel="Epoch", ylabel="Loss")
    axes[0].legend(); axes[0].grid(alpha=0.25)
    names, values = list(metrics), [entry["recall@10"] for entry in metrics.values()]
    axes[1].bar(range(len(names)), values, color="#2563eb")
    axes[1].set(title="Executed Recall@10", ylabel="Recall@10", xticks=range(len(names)), xticklabels=[name.replace(" matrix factorization", " MF").replace("Two-Tower + neural ranker", "Two-Tower + ranker") for name in names])
    axes[1].tick_params(axis="x", rotation=28); axes[1].grid(axis="y", alpha=0.25)
    figure.tight_layout(); figure.savefig(path, dpi=160, bbox_inches="tight"); plt.close(figure)


def run_pipeline(project_root: Path, settings: Settings, synthetic: bool = False) -> dict:
    set_seed(settings.seed)
    raw, processed, artifacts, docs = (project_root / "data/raw", project_root / "data/processed", project_root / "artifacts", project_root / "docs")
    for directory in (raw, processed, artifacts, docs): directory.mkdir(parents=True, exist_ok=True)
    logger.info("loading dataset synthetic=%s", synthetic)
    ratings, items = make_synthetic_ratings(seed=settings.seed) if synthetic else load_movielens(raw)
    data = prepare_data(ratings, items, settings.min_rating)
    num_users, num_items = len(data["user_map"]), len(data["item_map"])
    logger.info("prepared %s users, %s items, %s retrieval samples", num_users, num_items, len(data["train_samples"]))
    np.save(processed / "genre_features.npy", data["genre_features"])
    data["item_table"].to_csv(processed / "items.csv", index=False)
    with (processed / "histories.pkl").open("wb") as handle: pickle.dump({"train": data["train_histories"], "test": data["test_histories"]}, handle)

    model = TwoTower(num_users, num_items, torch.tensor(data["genre_features"]), settings.embedding_dim, settings.hidden_dim)
    logger.info("training two-tower retrieval model")
    retrieval_losses = train_retrieval(model, data["train_samples"], settings)
    embeddings = encode_items_array(model)
    retriever = FaissRetriever(embeddings)
    neural_scorer = make_neural_scorer(model, embeddings, settings)
    ranker = NeuralRanker(settings.embedding_dim, settings.hidden_dim)
    logger.info("training neural ranker")
    ranking_losses = train_ranker(model, ranker, data["train_samples"], num_items, settings)
    ranked_scorer = make_neural_scorer(model, embeddings, settings, ranker)

    logger.info("training baselines and evaluating models")
    popularity = PopularityRecommender().fit(data["train_events"], num_items)
    item_cf = ItemCFRecommender().fit(data["train_histories"], num_items)
    baseline_events = data["train_events"]
    if len(baseline_events) > settings.max_baseline_events:
        rng = np.random.default_rng(settings.seed)
        chosen = rng.choice(len(baseline_events), settings.max_baseline_events, replace=False)
        baseline_events = [baseline_events[index] for index in chosen]
    matrix_factorization = ImplicitMF(epochs=2, seed=settings.seed).fit(baseline_events, num_users, num_items)
    scorers = {"Popularity": popularity.score, "ItemCF": item_cf.score, "BPR matrix factorization": matrix_factorization.score,
               "Two-Tower": neural_scorer, "Two-Tower + neural ranker": ranked_scorer}
    evaluation_events = data["test_events"]
    if len(evaluation_events) > settings.max_evaluation_users:
        rng = np.random.default_rng(settings.seed)
        chosen = rng.choice(len(evaluation_events), settings.max_evaluation_users, replace=False)
        evaluation_events = [evaluation_events[index] for index in chosen]
    evaluation_histories = {user: data["test_histories"][user] for user, _ in evaluation_events}
    metrics = {name: evaluate_scores(score, evaluation_events, evaluation_histories, num_items) for name, score in scorers.items()}
    query_embeddings = np.vstack([encode_user(model, user, history, settings).numpy()[0] for user, history in evaluation_histories.items()])
    latency = retriever.benchmark(query_embeddings, repetitions=15)
    start = perf_counter(); [ranked_scorer(user, history) for user, history in evaluation_histories.items()]; latency["ranked_full_catalog_ms_per_query"] = (perf_counter() - start) * 1000 / len(evaluation_histories)

    logger.info("saving artifacts and report")
    settings.save(artifacts / "settings.json")
    torch.save(model.state_dict(), artifacts / "two_tower.pt")
    torch.save(ranker.state_dict(), artifacts / "ranker.pt")
    np.save(artifacts / "item_embeddings.npy", embeddings)
    if retriever.index is not None:
        import faiss
        faiss.write_index(retriever.index, str(artifacts / "items.faiss"))
    with (artifacts / "baselines.pkl").open("wb") as handle: pickle.dump({"popularity": popularity, "item_cf": item_cf, "mf": matrix_factorization}, handle)
    metadata = {"num_users": num_users, "num_items": num_items, "user_map": {str(k): int(v) for k, v in data["user_map"].items()},
                "item_map": {str(k): int(v) for k, v in data["item_map"].items()}, "user_reverse": data["user_reverse"], "item_reverse": data["item_reverse"],
                "dataset": "synthetic test fixture" if synthetic else "MovieLens 100K", "split": "per-user temporal last=holdout; penultimate=validation",
                "retrieval_training_examples": min(len(data["train_samples"]), settings.max_train_examples), "mf_training_events": len(baseline_events),
                "evaluation_users": len(evaluation_events)}
    (artifacts / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    report = {"dataset": metadata["dataset"], "training": {"retrieval_losses": retrieval_losses, "ranking_losses": ranking_losses,
              "retrieval_examples": metadata["retrieval_training_examples"], "mf_events": metadata["mf_training_events"],
              "evaluation_users": metadata["evaluation_users"]}, "metrics": metrics, "latency": latency}
    report["mlflow"] = log_run(asdict(settings), report)
    (artifacts / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    _save_metrics_markdown(metrics, docs / "benchmark_results.md")
    _save_visualization(report, docs / "training_visualization.png")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate the recommendation system.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--synthetic", action="store_true", help="Use deterministic test-only fixture, not MovieLens.")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()
    settings = Settings()
    if args.epochs is not None: settings.retrieval_epochs = settings.ranking_epochs = args.epochs
    report = run_pipeline(args.project_root, settings, args.synthetic)
    print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
