"""MovieLens 100K ingestion with a leakage-safe per-user temporal split."""
from __future__ import annotations

from pathlib import Path
from urllib.request import urlopen
import shutil
import ssl
import zipfile
import numpy as np
import pandas as pd

MOVIELENS_100K_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"


def download_movielens(raw_dir: Path) -> tuple[Path, Path]:
    """Download the public MovieLens 100K archive only when it is absent."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    ratings = raw_dir / "ml-100k" / "u.data"
    items = raw_dir / "ml-100k" / "u.item"
    if ratings.exists() and items.exists():
        return ratings, items
    archive = raw_dir / "ml-100k.zip"
    # The desktop-bundled Python does not always inherit the Windows CA store.
    # certifi keeps the HTTPS connection verified instead of disabling TLS checks.
    import certifi
    context = ssl.create_default_context(cafile=certifi.where())
    with urlopen(MOVIELENS_100K_URL, context=context) as response, archive.open("wb") as destination:
        shutil.copyfileobj(response, destination)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(raw_dir)
    return ratings, items


def load_movielens(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    ratings_path, items_path = download_movielens(raw_dir)
    ratings = pd.read_csv(
        ratings_path, sep="\t", names=["user_raw", "item_raw", "rating", "timestamp"]
    )
    columns = ["item_raw", "title", "release_date", "video_release", "imdb_url"] + [f"genre_{i}" for i in range(19)]
    items = pd.read_csv(items_path, sep="|", names=columns, encoding="latin-1")
    genre_cols = [c for c in items if c.startswith("genre_")]
    items["genre_vector"] = items[genre_cols].astype("float32").values.tolist()
    return ratings, items[["item_raw", "title", "genre_vector"]]


def make_synthetic_ratings(users: int = 24, items: int = 50, events_per_user: int = 10, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Small deterministic fixture used only by automated tests."""
    rng = np.random.default_rng(seed)
    rows = []
    for user in range(1, users + 1):
        choices = rng.choice(np.arange(1, items + 1), size=events_per_user, replace=False)
        for step, item in enumerate(choices):
            rows.append((user, int(item), 5.0 if step % 3 else 4.0, 1_600_000_000 + step + user * 100))
    ratings = pd.DataFrame(rows, columns=["user_raw", "item_raw", "rating", "timestamp"])
    meta = pd.DataFrame({"item_raw": np.arange(1, items + 1), "title": [f"Synthetic film {i}" for i in range(1, items + 1)]})
    meta["genre_vector"] = [[float(i % 2), float((i // 2) % 2), 1.0] for i in range(items)]
    return ratings, meta


def prepare_data(ratings: pd.DataFrame, items: pd.DataFrame, min_rating: float = 4.0) -> dict:
    """Clean data, reserve each user's final two events, and create model samples.

    Validation contexts contain training interactions only; test contexts contain training
    plus validation interactions. This mimics the information that would be available at
    the corresponding point in time and avoids future-item leakage.
    """
    events = ratings.dropna(subset=["user_raw", "item_raw", "rating", "timestamp"]).copy()
    events = events[events["rating"] >= min_rating]
    events = events.sort_values("timestamp").drop_duplicates(["user_raw", "item_raw"], keep="last")
    valid_users = events.groupby("user_raw").size()
    events = events[events.user_raw.isin(valid_users[valid_users >= 4].index)].sort_values(["user_raw", "timestamp"])
    user_map = {value: idx for idx, value in enumerate(sorted(events.user_raw.unique()))}
    item_map = {value: idx for idx, value in enumerate(sorted(events.item_raw.unique()))}
    events["user_id"] = events.user_raw.map(user_map).astype("int64")
    events["item_id"] = events.item_raw.map(item_map).astype("int64")
    item_table = items.drop_duplicates("item_raw").set_index("item_raw").reindex(sorted(item_map)).reset_index()
    item_table["title"] = item_table.title.fillna("Unknown title")
    item_table["genre_vector"] = item_table.genre_vector.apply(lambda x: x if isinstance(x, list) else [0.0] * 19)
    genre_features = np.asarray(item_table.genre_vector.tolist(), dtype=np.float32)

    train_rows, val_rows, test_rows, train_samples = [], [], [], []
    train_histories, test_histories = {}, {}
    for user_id, group in events.groupby("user_id", sort=False):
        sequence = group.sort_values("timestamp").item_id.tolist()
        train, val, test = sequence[:-2], sequence[-2], sequence[-1]
        train_histories[int(user_id)] = train
        test_histories[int(user_id)] = train + [val]
        train_rows.extend((int(user_id), item) for item in train)
        val_rows.append((int(user_id), val))
        test_rows.append((int(user_id), test))
        for position in range(1, len(train)):
            train_samples.append((int(user_id), train[:position], train[position]))
    return {
        "train_events": train_rows, "val_events": val_rows, "test_events": test_rows,
        "train_samples": train_samples, "train_histories": train_histories, "test_histories": test_histories,
        "user_map": user_map, "item_map": item_map, "user_reverse": {v: str(k) for k, v in user_map.items()},
        "item_reverse": {v: str(k) for k, v in item_map.items()}, "item_table": item_table,
        "genre_features": genre_features,
    }
