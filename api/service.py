"""Cached artifact loading and inference for the REST API and Streamlit app."""
from __future__ import annotations
import json
import os
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from src.config import Settings
from src.models import NeuralRanker, TwoTower
from src.retrieval import FaissRetriever
from src.training.pipeline import encode_user

# Tiny online batches are substantially faster and more predictable with one CPU thread.
torch.set_num_threads(max(1, int(os.getenv("TORCH_NUM_THREADS", "1"))))


class RecommendationService:
    def __init__(self, artifact_dir: Path | None = None, project_root: Path | None = None):
        root = project_root or Path(__file__).resolve().parents[1]
        self.artifact_dir = artifact_dir or Path(os.getenv("ARTIFACT_DIR", root / "artifacts"))
        required = ["settings.json", "metadata.json", "two_tower.pt", "ranker.pt", "item_embeddings.npy"]
        missing = [name for name in required if not (self.artifact_dir / name).exists()]
        if missing:
            raise RuntimeError(f"Model artifacts are unavailable ({', '.join(missing)}). Run `python -m src.training.pipeline` first.")
        values = json.loads((self.artifact_dir / "settings.json").read_text())
        self.settings = Settings(**values)
        self.metadata = json.loads((self.artifact_dir / "metadata.json").read_text())
        self.items = pd.read_csv(root / "data/processed/items.csv")
        features = torch.tensor(np.load(root / "data/processed/genre_features.npy"), dtype=torch.float32)
        self.model = TwoTower(self.metadata["num_users"], self.metadata["num_items"], features, self.settings.embedding_dim, self.settings.hidden_dim)
        self.model.load_state_dict(torch.load(self.artifact_dir / "two_tower.pt", map_location="cpu", weights_only=True)); self.model.eval()
        self.ranker = NeuralRanker(self.settings.embedding_dim, self.settings.hidden_dim)
        self.ranker.load_state_dict(torch.load(self.artifact_dir / "ranker.pt", map_location="cpu", weights_only=True)); self.ranker.eval()
        self.item_embeddings = np.load(self.artifact_dir / "item_embeddings.npy")
        self.retriever = FaissRetriever(self.item_embeddings)
        with (root / "data/processed/histories.pkl").open("rb") as handle:
            self.histories = pickle.load(handle)["test"]

    def _ids(self, user_id: str, history_item_ids: list[str] | None = None) -> tuple[int, list[int]]:
        user = int(self.metadata["user_map"].get(str(user_id), 0))
        if history_item_ids is None:
            return user, list(self.histories.get(user, []))
        history = [self.metadata["item_map"][str(item)] for item in history_item_ids if str(item) in self.metadata["item_map"]]
        return user, history

    def recommend(self, user_id: str, k: int = 10, history_item_ids: list[str] | None = None) -> list[dict]:
        if k < 1 or k > 100: raise ValueError("k must be between 1 and 100")
        user, history = self._ids(user_id, history_item_ids)
        user_vector = encode_user(self.model, user, history, self.settings).numpy()
        retrieval_scores, indices = self.retriever.search(user_vector, min(max(k * 5, 30), self.metadata["num_items"]))
        candidate_ids = [int(item) for item in indices[0] if int(item) not in set(history)]
        if not candidate_ids: return []
        candidates = torch.tensor(self.item_embeddings[candidate_ids], dtype=torch.float32)
        users = torch.tensor(user_vector, dtype=torch.float32).repeat(len(candidate_ids), 1)
        candidate_retrieval = torch.tensor((user_vector[0] @ self.item_embeddings[candidate_ids].T), dtype=torch.float32)
        with torch.no_grad(): scores = torch.sigmoid(self.ranker(users, candidates, candidate_retrieval)).numpy()
        ordered = np.argsort(-scores)[:k]
        results = []
        for position in ordered:
            item_index = candidate_ids[int(position)]
            item = self.items.iloc[item_index]
            results.append({"item_id": self.metadata["item_reverse"][str(item_index)], "title": str(item.title),
                            "score": float(scores[position]), "retrieval_score": float(candidate_retrieval[position])})
        return results

    def similar_items(self, item_id: str, k: int = 10) -> list[dict]:
        if str(item_id) not in self.metadata["item_map"]: raise KeyError(item_id)
        item_index = self.metadata["item_map"][str(item_id)]
        scores, indices = self.retriever.search(self.item_embeddings[[item_index]], min(k + 1, self.metadata["num_items"]))
        output = []
        for score, nearby in zip(scores[0], indices[0]):
            if int(nearby) == item_index: continue
            item = self.items.iloc[int(nearby)]
            output.append({"item_id": self.metadata["item_reverse"][str(int(nearby))], "title": str(item.title), "score": float(score)})
            if len(output) == k: break
        return output

    def model_info(self) -> dict:
        report_path = self.artifact_dir / "report.json"
        report = json.loads(report_path.read_text()) if report_path.exists() else {}
        return {"model": "two-tower retrieval + neural ranker", "dataset": self.metadata["dataset"], "num_users": self.metadata["num_users"],
                "num_items": self.metadata["num_items"], "embedding_dim": self.settings.embedding_dim, "retrieval_backend": self.retriever.backend,
                "evaluation": report.get("metrics", {}), "latency": report.get("latency", {})}
