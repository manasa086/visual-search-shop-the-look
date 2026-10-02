"""Quality metrics for comparing an approximate index against exact search.

Result matrices are (queries, k) arrays of ids, padded with -1 where an index returned
fewer than k results. Padding counts against the index in both metrics.
"""

import numpy as np


def recall_at_k(true_ids: np.ndarray, found_ids: np.ndarray) -> float:
    """Average fraction of each query's exact top-k that the index also returned."""
    if true_ids.shape[0] != found_ids.shape[0]:
        raise ValueError("true_ids and found_ids must have the same number of queries")
    hits = sum(np.isin(true, found).sum() for true, found in zip(true_ids, found_ids, strict=True))
    return float(hits / true_ids.size)


def category_precision(
    found_ids: np.ndarray, labels: np.ndarray, query_labels: np.ndarray
) -> float:
    """Average fraction of the k result slots holding an item with the query's category.

    `labels[i]` is the category of indexed item i. This measures usefulness to a person,
    which can stay high even when an approximate index misses some exact neighbors.
    """
    found = np.asarray(found_ids)
    matches = (labels[np.where(found >= 0, found, 0)] == query_labels[:, None]) & (found >= 0)
    return float(matches.mean())
