"""FAISS HNSW retrieval with an explicit exact-search benchmark."""
from __future__ import annotations
from time import perf_counter
from functools import lru_cache
import os
import subprocess
import sys
import numpy as np


@lru_cache(maxsize=1)
def _faiss_is_usable() -> bool:
    """Protect a serving process from optional native-extension import failures.

    Some Windows environments can install a FAISS wheel whose DLL cannot load. Probing
    in a child process preserves a deterministic NumPy exact fallback rather than
    crashing FastAPI or a training run. Set IRE_DISABLE_FAISS=1 to deliberately test
    the fallback; compatible Linux/macOS/Windows environments use FAISS normally.
    """
    if os.getenv("IRE_DISABLE_FAISS") == "1":
        return False
    # The current PyPI FAISS Windows wheel intermittently crashes on HNSW construction
    # in the constrained desktop sandbox. Prefer the stable exact fallback there;
    # developers with a validated Windows FAISS installation may opt in explicitly.
    if os.name == "nt" and os.getenv("IRE_ENABLE_FAISS") != "1":
        return False
    probe = subprocess.run([sys.executable, "-c", "import faiss"], capture_output=True, timeout=20, check=False)
    return probe.returncode == 0


class FaissRetriever:
    def __init__(self, item_embeddings: np.ndarray, hnsw_m: int = 32):
        self.item_embeddings = np.ascontiguousarray(item_embeddings.astype("float32"))
        self.backend = "numpy-exact-fallback"
        self.index = None
        try:
            if not _faiss_is_usable():
                return
            import faiss
            self.index = faiss.IndexHNSWFlat(self.item_embeddings.shape[1], hnsw_m, faiss.METRIC_INNER_PRODUCT)
            self.index.hnsw.efSearch = min(128, len(self.item_embeddings))
            self.index.add(self.item_embeddings)
            self.backend = "faiss-hnsw"
        except (ImportError, OSError):
            pass

    def search(self, query_embeddings: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        queries = np.ascontiguousarray(query_embeddings.astype("float32"))
        k = min(k, len(self.item_embeddings))
        if self.index is not None:
            scores, indices = self.index.search(queries, k)
            return scores, indices
        scores = queries @ self.item_embeddings.T
        indices = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
        ordered = np.take_along_axis(indices, np.argsort(-np.take_along_axis(scores, indices, axis=1), axis=1), axis=1)
        return np.take_along_axis(scores, ordered, axis=1), ordered

    def exact_search(self, query_embeddings: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        scores = np.ascontiguousarray(query_embeddings.astype("float32")) @ self.item_embeddings.T
        k = min(k, len(self.item_embeddings))
        indices = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
        ordered = np.take_along_axis(indices, np.argsort(-np.take_along_axis(scores, indices, axis=1), axis=1), axis=1)
        return np.take_along_axis(scores, ordered, axis=1), ordered

    def benchmark(self, query_embeddings: np.ndarray, k: int = 10, repetitions: int = 15) -> dict:
        queries = query_embeddings[: min(100, len(query_embeddings))]
        start = perf_counter()
        for _ in range(repetitions): self.search(queries, k)
        ann_ms = (perf_counter() - start) * 1000 / repetitions / len(queries)
        start = perf_counter()
        for _ in range(repetitions): self.exact_search(queries, k)
        exact_ms = (perf_counter() - start) * 1000 / repetitions / len(queries)
        return {"backend": self.backend, "queries": int(len(queries)), "ann_ms_per_query": ann_ms, "exact_ms_per_query": exact_ms}
