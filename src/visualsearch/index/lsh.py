"""Locality-sensitive hashing for cosine similarity (random hyperplanes / SimHash).

Each table draws `bits_per_table` random hyperplanes. A vector's hash is the pattern of
which side of each hyperplane it falls on, so two vectors with a small angle between them
agree on most bits and often share a whole code. Search hashes the query in every table,
collects the vectors that share its code (the candidates), and ranks only those by exact
cosine similarity. Using several tables gives a close neighbor several chances to collide
with the query.

More bits make buckets smaller (faster, but neighbors are more likely to be missed); more
tables recover recall at the cost of memory and hashing time. With `hamming_radius=1` the
search also looks in buckets whose code differs from the query's in a single bit, which
trades extra lookups for higher recall.
"""

import numpy as np

from visualsearch.index.base import Index
from visualsearch.index.utils import check_matrix, normalize, top_k

_MAX_BITS = 62  # codes are packed into int64


class LSHIndex(Index):
    def __init__(
        self,
        num_tables: int = 16,
        bits_per_table: int = 12,
        hamming_radius: int = 0,
        seed: int = 0,
    ) -> None:
        if num_tables < 1:
            raise ValueError(f"num_tables must be >= 1, got {num_tables}")
        if not 1 <= bits_per_table <= _MAX_BITS:
            raise ValueError(f"bits_per_table must be in [1, {_MAX_BITS}], got {bits_per_table}")
        if hamming_radius not in (0, 1):
            raise ValueError(f"hamming_radius must be 0 or 1, got {hamming_radius}")
        self.num_tables = num_tables
        self.bits_per_table = bits_per_table
        self.hamming_radius = hamming_radius
        self.seed = seed
        self._bit_weights = 1 << np.arange(bits_per_table, dtype=np.int64)
        self._vectors: np.ndarray | None = None
        self._planes: np.ndarray | None = None  # (tables, d, bits)
        # Per table: all codes in sorted order, and the vector ids in that same order.
        # A bucket is then a contiguous slice that searchsorted can locate.
        self._tables: list[tuple[np.ndarray, np.ndarray]] = []

    def __len__(self) -> int:
        return 0 if self._vectors is None else len(self._vectors)

    def build(self, vectors: np.ndarray) -> None:
        self._vectors = normalize(check_matrix(vectors))
        dim = self._vectors.shape[1]
        rng = np.random.default_rng(self.seed)
        self._planes = rng.standard_normal(
            (self.num_tables, dim, self.bits_per_table), dtype=np.float32
        )
        self._tables = []
        for planes in self._planes:
            codes = self._codes(self._vectors @ planes)
            order = np.argsort(codes, kind="stable")
            self._tables.append((codes[order], order))

    def memory_bytes(self) -> int:
        if self._vectors is None:
            return 0
        tables = sum(codes.nbytes + ids.nbytes for codes, ids in self._tables)
        return self._vectors.nbytes + self._planes.nbytes + tables

    def candidates(self, query: np.ndarray) -> np.ndarray:
        """Ids of every vector that shares a bucket with the query in at least one table."""
        return self._candidates(self._prepare(query))

    def search(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        query = self._prepare(query)
        candidates = self._candidates(query)
        if len(candidates) == 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
        ranked, scores = top_k(self._vectors[candidates] @ query, k)
        return candidates[ranked], scores

    def _prepare(self, query: np.ndarray) -> np.ndarray:
        if self._vectors is None:
            raise RuntimeError("index is empty; call build() first")
        query = np.asarray(query)
        expected = self._vectors.shape[1]
        if query.shape != (expected,):
            raise ValueError(f"expected query of dimension {expected}, got shape {query.shape}")
        return normalize(query)

    def _candidates(self, query: np.ndarray) -> np.ndarray:
        query_codes = self._codes(query @ self._planes)  # one code per table
        buckets = []
        for (sorted_codes, ids), code in zip(self._tables, query_codes, strict=True):
            probes = self._probe_codes(code)
            starts = np.searchsorted(sorted_codes, probes, side="left")
            ends = np.searchsorted(sorted_codes, probes, side="right")
            buckets.extend(ids[start:end] for start, end in zip(starts, ends, strict=True))
        return np.unique(np.concatenate(buckets))

    def _codes(self, projections: np.ndarray) -> np.ndarray:
        """Pack the sign pattern of each row of projections into one integer code."""
        return (projections > 0).astype(np.int64) @ self._bit_weights

    def _probe_codes(self, code: np.int64) -> np.ndarray:
        if self.hamming_radius == 0:
            return np.array([code], dtype=np.int64)
        return np.concatenate(([code], code ^ self._bit_weights))
