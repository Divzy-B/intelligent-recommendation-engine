from pathlib import Path
import json
import pickle
import numpy as np
import pandas as pd
import torch

from api import main
from api.service import RecommendationService
from src.config import Settings
from src.models import NeuralRanker, TwoTower


def test_api_endpoints_with_model_artifacts(tmp_path, monkeypatch):
    root = Path(tmp_path); artifacts = root / "artifacts"; processed = root / "data" / "processed"
    artifacts.mkdir(); processed.mkdir(parents=True)
    settings = Settings(embedding_dim=8, hidden_dim=16, max_history=5)
    settings.save(artifacts / "settings.json")
    features = np.zeros((50, 3), dtype=np.float32)
    np.save(processed / "genre_features.npy", features)
    items = pd.DataFrame({"item_raw": range(1, 51), "title": [f"Film {i}" for i in range(1, 51)]})
    items.to_csv(processed / "items.csv", index=False)
    with (processed / "histories.pkl").open("wb") as handle: pickle.dump({"test": {0: [1, 2, 3]}}, handle)
    metadata = {"num_users": 1, "num_items": 50, "dataset": "test fixture", "user_map": {"1": 0},
                "item_map": {str(i): i - 1 for i in range(1, 51)}, "item_reverse": {str(i - 1): str(i) for i in range(1, 51)}}
    (artifacts / "metadata.json").write_text(json.dumps(metadata))
    model = TwoTower(1, 50, torch.tensor(features), 8, 16); ranker = NeuralRanker(8, 16)
    torch.save(model.state_dict(), artifacts / "two_tower.pt"); torch.save(ranker.state_dict(), artifacts / "ranker.pt")
    embeddings = model.encode_items(torch.arange(50)).detach().numpy().astype("float32")
    np.save(artifacts / "item_embeddings.npy", embeddings)
    (artifacts / "report.json").write_text(json.dumps({"metrics": {}, "latency": {}}))
    service = RecommendationService(artifacts, root)
    monkeypatch.setattr(main, "get_service", lambda: service)
    assert main.health()["status"] == "ok"
    response = main.recommend_user("1", 4)
    assert len(response) == 4
    item_id = response[0]["item_id"]
    assert len(main.similar_items(item_id, 3)) == 3
    assert len(main.recommend_post(main.RecommendationRequest(user_id="1", k=3, history_item_ids=[item_id]))) == 3
    assert main.model_info()["num_items"] == 50
