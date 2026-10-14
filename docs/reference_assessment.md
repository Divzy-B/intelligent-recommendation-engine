# Reference assessment and attribution

The project was informed by the architecture described in
[`abdullahhrashid/industrial-recommender-system`](https://github.com/abdullahhrashid/industrial-recommender-system), inspected on 2026-10-13.
That repository organizes a production two-stage pipeline around Amazon Reviews 2023: preprocessing, a PyTorch two-tower retrieval model, FAISS candidate generation, a neural ranker, FastAPI, Docker, and a MIT license.

This repository is an independent implementation. It deliberately uses the smaller public MovieLens 100K dataset, a genre-feature item tower and pooled-history user tower, compact BPR and co-occurrence baselines, a separate artifact format, and a different serving/dashboard implementation. No source code, checkpoints, data, or benchmark claims were copied from the reference.

The reference repository's MIT license permits reuse with notice. No code from it is redistributed here, but the attribution is preserved as professional credit for the architectural inspiration.
