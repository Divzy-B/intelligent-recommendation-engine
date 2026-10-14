from dataclasses import dataclass, asdict
from pathlib import Path
import json


@dataclass
class Settings:
    seed: int = 42
    embedding_dim: int = 48
    hidden_dim: int = 96
    max_history: int = 30
    batch_size: int = 512
    retrieval_epochs: int = 2
    ranking_epochs: int = 2
    learning_rate: float = 0.003
    weight_decay: float = 1e-5
    min_rating: float = 4.0
    max_train_examples: int = 5000
    max_baseline_events: int = 3000
    max_evaluation_users: int = 300
    candidates: int = 100

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
