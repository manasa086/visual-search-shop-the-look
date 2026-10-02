"""The contract every nearest-neighbor index implements."""

from abc import ABC, abstractmethod

import numpy as np


class Index(ABC):
    """Cosine-similarity nearest-neighbor index over a fixed set of vectors.

    Implementations normalize vectors themselves, so callers can pass raw
    embeddings. Row numbers in the build matrix are the ids that `search` returns.
    """

    @abstractmethod
    def build(self, vectors: np.ndarray) -> None:
        """Index an (n, d) matrix of vectors."""

    @abstractmethod
    def search(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (ids, scores) of the best matches for one (d,) query, best first.

        Approximate indexes may return fewer than k results.
        """

    def search_batch(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Search a (q, d) matrix of queries; returns (q, k) ids and scores.

        Rows with fewer than k results are padded with id -1 and score -inf.
        """
        ids = np.full((len(queries), k), -1, dtype=np.int64)
        scores = np.full((len(queries), k), -np.inf, dtype=np.float32)
        for row, query in enumerate(queries):
            found_ids, found_scores = self.search(query, k)
            ids[row, : len(found_ids)] = found_ids
            scores[row, : len(found_scores)] = found_scores
        return ids, scores
