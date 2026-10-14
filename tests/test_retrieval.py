import numpy as np
from src.retrieval import FaissRetriever


def test_retriever_returns_ranked_candidates_and_benchmark():
    embeddings = np.eye(8, dtype=np.float32)
    retriever = FaissRetriever(embeddings)
    scores, indices = retriever.search(embeddings[[3]], k=3)
    assert indices.shape == (1, 3)
    assert int(indices[0, 0]) == 3
    assert scores[0, 0] >= scores[0, 1]
    assert retriever.benchmark(embeddings, repetitions=2)["queries"] == 8
