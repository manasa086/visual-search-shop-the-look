import numpy as np
import pytest

from visualsearch.bench.metrics import recall_at_k
from visualsearch.index import BruteForceIndex, HNSWIndex
from visualsearch.index.utils import normalize


@pytest.fixture(scope="module")
def clustered() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    centers = rng.normal(size=(40, 64))
    points = centers[rng.integers(0, 40, size=2400)] + 0.6 * rng.normal(size=(2400, 64))
    return points[:2000].astype(np.float32), points[2000:].astype(np.float32)


@pytest.fixture(scope="module")
def built(clustered) -> HNSWIndex:
    index = HNSWIndex(M=16, ef_construction=100, ef_search=50)
    index.build(clustered[0])
    return index


def test_recall_on_clustered_data(clustered, built):
    vectors, queries = clustered
    exact = BruteForceIndex()
    exact.build(vectors)
    truth = exact.search_batch(queries, k=10)[0]

    found = built.search_batch(queries, k=10)[0]

    assert recall_at_k(truth, found) > 0.95


def test_scores_are_cosine_similarities_best_first(clustered, built):
    vectors, queries = clustered

    ids, scores = built.search(queries[0], k=10)

    expected = normalize(vectors)[ids] @ normalize(queries[0])
    np.testing.assert_allclose(scores, expected, atol=1e-5)
    assert list(scores) == sorted(scores, reverse=True)


def test_stored_vector_finds_itself_first(clustered, built):
    ids, scores = built.search(clustered[0][5], k=1)

    assert ids[0] == 5
    assert scores[0] == pytest.approx(1.0, abs=1e-5)


def test_k_larger_than_ef_and_dataset_still_works(clustered):
    index = HNSWIndex(ef_search=4)
    index.build(clustered[0][:30])

    assert len(index.search(clustered[1][0], k=20)[0]) == 20
    assert len(index.search(clustered[1][0], k=500)[0]) == 30


def test_wider_beam_does_not_hurt_recall(clustered):
    vectors, queries = clustered
    exact = BruteForceIndex()
    exact.build(vectors)
    truth = exact.search_batch(queries, k=10)[0]
    index = HNSWIndex(M=8, ef_construction=50, ef_search=10)
    index.build(vectors)

    narrow = recall_at_k(truth, index.search_batch(queries, k=10)[0])
    index.set_ef_search(200)
    wide = recall_at_k(truth, index.search_batch(queries, k=10)[0])

    assert wide >= narrow


def test_memory_is_larger_than_raw_vectors(clustered, built):
    assert built.memory_bytes() > clustered[0].nbytes


@pytest.mark.parametrize("kwargs", [{"M": 1}, {"ef_construction": 0}, {"ef_search": 0}])
def test_invalid_parameters_are_rejected(kwargs):
    with pytest.raises(ValueError):
        HNSWIndex(**kwargs)


def test_search_before_build_raises():
    with pytest.raises(RuntimeError):
        HNSWIndex().search(np.zeros(8), k=1)


def test_wrong_query_dimension_raises(built):
    with pytest.raises(ValueError):
        built.search(np.zeros(63), k=1)
