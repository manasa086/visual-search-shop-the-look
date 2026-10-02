"""Exact search: compare the query against every vector.

Slow at scale, but always correct, so it doubles as the ground truth that the
approximate indexes are measured against.
"""

import numpy as np

from visualsearch.index.base import Index
from visualsearch.index.utils import check_matrix, normalize, top_k

_BATCH_CHUNK = 256


class BruteForceIndex(Index):
    def __init__(self) -> None:
        self._vectors: np.ndarray | None = None

    def __len__(self) -> int:
        return 0 if self._vectors is None else len(self._vectors)

    def build(self, vectors: np.ndarray) -> None:
        self._vectors = normalize(check_matrix(vectors))

    def memory_bytes(self) -> int:
        return 0 if self._vectors is None else self._vectors.nbytes

    def search(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        vectors = self._require_built()
        query = np.asarray(query)
        self._check_dim(query.shape)
        return top_k(vectors @ normalize(query), k)

    def search_batch(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        vectors = self._require_built()
        queries = np.asarray(queries)
        if queries.ndim != 2:
            raise ValueError(f"expected a (q, d) matrix, got shape {queries.shape}")
        self._check_dim(queries.shape[1:])
        queries = normalize(queries)
        results = [
            top_k(queries[start : start + _BATCH_CHUNK] @ vectors.T, k)
            for start in range(0, len(queries), _BATCH_CHUNK)
        ]
        return (
            np.concatenate([ids for ids, _ in results]),
            np.concatenate([scores for _, scores in results]),
        )

    def _require_built(self) -> np.ndarray:
        if self._vectors is None:
            raise RuntimeError("index is empty; call build() first")
        return self._vectors

    def _check_dim(self, query_shape: tuple[int, ...]) -> None:
        expected = self._vectors.shape[1]
        if query_shape != (expected,):
            raise ValueError(f"expected query of dimension {expected}, got shape {query_shape}")
