"""Build an index, run held-out queries through it, and report speed, memory and quality."""

from time import perf_counter

import numpy as np

from visualsearch.bench.metrics import category_precision, recall_at_k
from visualsearch.index import BruteForceIndex, Index

_WARMUP_QUERIES = 20
_CANDIDATE_SAMPLE = 200


def split_queries(num_items: int, num_queries: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Randomly hold out `num_queries` rows as queries; the rest are indexed.

    Held-out queries are not in the index, so a perfect score means finding true
    neighbors, not just finding the query itself. Returns (index_ids, query_ids).
    """
    if not 0 < num_queries < num_items:
        raise ValueError(f"need 0 < num_queries < {num_items}, got {num_queries}")
    permutation = np.random.default_rng(seed).permutation(num_items)
    return np.sort(permutation[num_queries:]), np.sort(permutation[:num_queries])


def exact_neighbors(index_vectors: np.ndarray, queries: np.ndarray, k: int) -> np.ndarray:
    """Ground truth: the exact top-k ids for every query."""
    brute_force = BruteForceIndex()
    brute_force.build(index_vectors)
    return brute_force.search_batch(queries, k)[0]


def evaluate(
    index: Index,
    index_vectors: np.ndarray,
    queries: np.ndarray,
    truth: np.ndarray,
    k: int,
    labels: np.ndarray | None = None,
    query_labels: np.ndarray | None = None,
) -> dict:
    """Measure one index. Queries run one at a time, as an API would serve them."""
    start = perf_counter()
    index.build(index_vectors)
    build_seconds = perf_counter() - start

    for query in queries[:_WARMUP_QUERIES]:
        index.search(query, k)

    found = np.full((len(queries), k), -1, dtype=np.int64)
    latencies_ms = np.empty(len(queries))
    for row, query in enumerate(queries):
        start = perf_counter()
        ids, _ = index.search(query, k)
        latencies_ms[row] = (perf_counter() - start) * 1000
        found[row, : len(ids)] = ids

    result = {
        "build_seconds": build_seconds,
        "memory_mb": index.memory_bytes() / 1e6,
        "p50_ms": float(np.percentile(latencies_ms, 50)),
        "p95_ms": float(np.percentile(latencies_ms, 95)),
        "qps": float(1000 / latencies_ms.mean()),
        f"recall@{k}": recall_at_k(truth, found),
        "candidate_fraction": _candidate_fraction(index, queries, len(index_vectors)),
    }
    if labels is not None and query_labels is not None:
        result[f"category_precision@{k}"] = category_precision(found, labels, query_labels)
    return result


def _candidate_fraction(index: Index, queries: np.ndarray, num_items: int) -> float:
    """Share of the dataset the index ranks per query (1.0 for exhaustive search)."""
    candidates = getattr(index, "candidates", None)
    if candidates is None:
        return 1.0
    sample = queries[:_CANDIDATE_SAMPLE]
    return float(np.mean([len(candidates(query)) for query in sample]) / num_items)
