import numpy as np
import pytest

from visualsearch.bench.metrics import category_precision, recall_at_k
from visualsearch.bench.runner import evaluate, exact_neighbors, split_queries
from visualsearch.index import BruteForceIndex, HNSWIndex, LSHIndex


def test_recall_counts_overlap_and_penalizes_padding():
    truth = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    found = np.array([[4, 3, 9, 10], [5, 6, 7, -1]])

    assert recall_at_k(truth, found) == pytest.approx((2 + 3) / 8)


def test_perfect_recall_ignores_order():
    truth = np.array([[1, 2, 3]])

    assert recall_at_k(truth, np.array([[3, 1, 2]])) == 1.0


def test_category_precision_counts_matching_slots_and_padding_as_misses():
    labels = np.array(["cat", "dog", "cat", "bird"])
    found = np.array([[0, 2, 1, 3], [1, -1, -1, -1]])
    query_labels = np.array(["cat", "dog"])

    assert category_precision(found, labels, query_labels) == pytest.approx((2 + 1) / 8)


def test_split_is_disjoint_complete_and_deterministic():
    index_ids, query_ids = split_queries(100, 10, seed=3)

    assert len(query_ids) == 10
    assert set(index_ids).isdisjoint(query_ids)
    assert sorted(np.concatenate([index_ids, query_ids])) == list(range(100))
    np.testing.assert_array_equal(query_ids, split_queries(100, 10, seed=3)[1])


def test_split_rejects_impossible_query_counts():
    with pytest.raises(ValueError):
        split_queries(10, 10)


@pytest.fixture(scope="module")
def dataset():
    rng = np.random.default_rng(0)
    vectors = rng.normal(size=(600, 32)).astype(np.float32)
    labels = rng.integers(0, 5, size=600)
    index_ids, query_ids = split_queries(600, 50)
    return vectors[index_ids], vectors[query_ids], labels[index_ids], labels[query_ids]


def test_exact_search_scores_perfect_recall_against_itself(dataset):
    index_vectors, queries, labels, query_labels = dataset
    truth = exact_neighbors(index_vectors, queries, k=5)

    result = evaluate(BruteForceIndex(), index_vectors, queries, truth, 5, labels, query_labels)

    assert result["recall@5"] == 1.0
    assert result["candidate_fraction"] == 1.0
    assert 0 <= result["category_precision@5"] <= 1
    assert result["p50_ms"] <= result["p95_ms"]
    assert result["memory_mb"] > 0


def test_hnsw_has_no_candidate_fraction_and_prebuilt_index_skips_build_time(dataset):
    index_vectors, queries, _, _ = dataset
    truth = exact_neighbors(index_vectors, queries, k=5)
    index = HNSWIndex(ef_construction=50)
    index.build(index_vectors)

    result = evaluate(index, index_vectors, queries, truth, 5, build=False)

    assert np.isnan(result["candidate_fraction"])
    assert np.isnan(result["build_seconds"])
    assert np.isnan(result["memory_mb"])
    assert result["recall@5"] > 0.9


def test_lsh_reports_candidate_fraction_below_one(dataset):
    index_vectors, queries, _, _ = dataset
    truth = exact_neighbors(index_vectors, queries, k=5)

    result = evaluate(LSHIndex(num_tables=2, bits_per_table=10), index_vectors, queries, truth, 5)

    assert 0 < result["candidate_fraction"] < 1
    assert 0 <= result["recall@5"] <= 1
