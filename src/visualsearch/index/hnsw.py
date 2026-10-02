"""Approximate search with a Hierarchical Navigable Small World graph (via hnswlib).

Every vector is a node linked to about `M` near neighbors. A query enters at the top of a
sparse layered graph and greedily hops toward closer nodes, then keeps a beam of
`ef_search` candidates on the dense bottom layer. A larger beam is slower but finds more of
the true neighbors. This is the industry-standard reference the from-scratch LSH index is
compared against.
"""

import os
import tempfile

import hnswlib
import numpy as np

from visualsearch.index.base import Index
from visualsearch.index.utils import check_matrix, normalize


class HNSWIndex(Index):
    def __init__(
        self,
        M: int = 16,  # noqa: N803 - the name used throughout the HNSW literature
        ef_construction: int = 200,
        ef_search: int = 64,
        seed: int = 0,
    ) -> None:
        if M < 2:
            raise ValueError(f"M must be >= 2, got {M}")
        if ef_construction < 1 or ef_search < 1:
            raise ValueError("ef_construction and ef_search must be >= 1")
        self.M = M
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.seed = seed
        self._index: hnswlib.Index | None = None
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def build(self, vectors: np.ndarray) -> None:
        vectors = normalize(check_matrix(vectors))
        index = hnswlib.Index(space="cosine", dim=vectors.shape[1])
        index.init_index(
            max_elements=len(vectors),
            ef_construction=self.ef_construction,
            M=self.M,
            random_seed=self.seed,
        )
        index.add_items(vectors, np.arange(len(vectors)))
        self._index = index
        self._size = len(vectors)

    def set_ef_search(self, ef_search: int) -> None:
        """Change the search beam width without rebuilding the graph."""
        if ef_search < 1:
            raise ValueError(f"ef_search must be >= 1, got {ef_search}")
        self.ef_search = ef_search

    def search(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        if self._index is None:
            raise RuntimeError("index is empty; call build() first")
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        query = np.asarray(query)
        if query.shape != (self._index.dim,):
            raise ValueError(
                f"expected query of dimension {self._index.dim}, got shape {query.shape}"
            )
        k = min(k, self._size)
        self._index.set_ef(max(self.ef_search, k))  # hnswlib requires ef >= k
        labels, distances = self._index.knn_query(normalize(query), k=k)
        return labels[0].astype(np.int64), 1.0 - distances[0]

    def memory_bytes(self) -> int:
        """Size of the serialized graph, which closely tracks its in-memory size."""
        if self._index is None:
            return 0
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "hnsw.bin")
            self._index.save_index(path)
            return os.path.getsize(path)
