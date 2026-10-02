"""Vector helpers shared by every index implementation."""

import numpy as np


def normalize(x: np.ndarray) -> np.ndarray:
    """L2-normalize along the last axis, so cosine similarity equals a dot product."""
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norms, 1e-12)


def top_k(scores: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the k highest scores per row of `scores`, best first, as (ids, scores).

    Works on a 1-D score vector or a (queries, items) matrix. If k exceeds the
    number of items, every item is returned.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    n = scores.shape[-1]
    k = min(k, n)
    if k < n:
        candidates = np.argpartition(-scores, k - 1, axis=-1)[..., :k]
    else:
        candidates = np.broadcast_to(np.arange(n), scores.shape)
    candidate_scores = np.take_along_axis(scores, candidates, axis=-1)
    order = np.argsort(-candidate_scores, axis=-1)
    return (
        np.take_along_axis(candidates, order, axis=-1),
        np.take_along_axis(candidate_scores, order, axis=-1),
    )
