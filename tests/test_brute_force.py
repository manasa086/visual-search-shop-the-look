import numpy as np
import pytest

from visualsearch.index import BruteForceIndex
from visualsearch.index.utils import normalize, top_k


def naive_top_k(vectors: np.ndarray, query: np.ndarray, k: int) -> np.ndarray:
    scores = normalize(vectors) @ normalize(query)
    return np.argsort(-scores)[:k]


@pytest.fixture
def random_vectors() -> np.ndarray:
    return np.random.default_rng(0).normal(size=(500, 32)).astype(np.float32)


def test_known_example():
    index = BruteForceIndex()
    index.build(np.array([[1, 0], [0, 1], [1, 1]], dtype=np.float32))

    ids, scores = index.search(np.array([1.0, 0.1]), k=2)

    assert ids.tolist() == [0, 2]
    assert scores[0] > scores[1]


def test_matches_naive_ranking(random_vectors):
    index = BruteForceIndex()
    index.build(random_vectors)
    query = np.random.default_rng(1).normal(size=32)

    ids, _ = index.search(query, k=10)

    assert ids.tolist() == naive_top_k(random_vectors, query, 10).tolist()


def test_batch_matches_single_queries(random_vectors):
    index = BruteForceIndex()
    index.build(random_vectors)
    queries = np.random.default_rng(2).normal(size=(300, 32))  # spans several chunks

    batch_ids, batch_scores = index.search_batch(queries, k=5)

    assert batch_ids.shape == (300, 5)
    for row in (0, 150, 299):
        ids, scores = index.search(queries[row], k=5)
        assert batch_ids[row].tolist() == ids.tolist()
        np.testing.assert_allclose(batch_scores[row], scores, rtol=1e-5)


def test_vector_scale_does_not_change_results(random_vectors):
    scales = np.random.default_rng(3).uniform(0.1, 50, size=(500, 1)).astype(np.float32)
    plain, scaled = BruteForceIndex(), BruteForceIndex()
    plain.build(random_vectors)
    scaled.build(random_vectors * scales)
    query = random_vectors[7]

    assert plain.search(query, 10)[0].tolist() == scaled.search(query * 3, 10)[0].tolist()


def test_k_larger_than_index_returns_everything_sorted():
    index = BruteForceIndex()
    index.build(np.eye(3, dtype=np.float32))

    ids, scores = index.search(np.array([0.0, 1.0, 0.5]), k=10)

    assert ids.tolist() == [1, 2, 0]
    assert list(scores) == sorted(scores, reverse=True)


def test_stored_vector_finds_itself_first(random_vectors):
    index = BruteForceIndex()
    index.build(random_vectors)

    ids, scores = index.search(random_vectors[42], k=1)

    assert ids[0] == 42
    assert scores[0] == pytest.approx(1.0, abs=1e-5)


def test_search_before_build_raises():
    with pytest.raises(RuntimeError):
        BruteForceIndex().search(np.zeros(4), k=1)


def test_wrong_query_dimension_raises(random_vectors):
    index = BruteForceIndex()
    index.build(random_vectors)

    with pytest.raises(ValueError):
        index.search(np.zeros(31), k=1)
    with pytest.raises(ValueError):
        index.search_batch(np.zeros((2, 31)), k=1)


def test_build_rejects_bad_shapes():
    with pytest.raises(ValueError):
        BruteForceIndex().build(np.zeros(8))
    with pytest.raises(ValueError):
        BruteForceIndex().build(np.zeros((0, 8)))


def test_top_k_rejects_non_positive_k():
    with pytest.raises(ValueError):
        top_k(np.array([0.1, 0.2]), k=0)
