import numpy as np
import pytest

from visualsearch.bench.metrics import recall_at_k
from visualsearch.index import BruteForceIndex, LSHIndex
from visualsearch.index.utils import normalize


@pytest.fixture(scope="module")
def clustered() -> tuple[np.ndarray, np.ndarray]:
    """2000 indexed points and 400 held-out queries drawn around 40 cluster centers."""
    rng = np.random.default_rng(0)
    centers = rng.normal(size=(40, 64))
    points = centers[rng.integers(0, 40, size=2400)] + 0.6 * rng.normal(size=(2400, 64))
    return points[:2000].astype(np.float32), points[2000:].astype(np.float32)


def test_stored_vector_finds_itself_first(clustered):
    vectors, _ = clustered
    index = LSHIndex(num_tables=4, bits_per_table=12)
    index.build(vectors)

    ids, scores = index.search(vectors[123], k=1)

    assert ids[0] == 123
    assert scores[0] == pytest.approx(1.0, abs=1e-5)


def test_returned_scores_are_exact_cosine_similarities(clustered):
    vectors, queries = clustered
    index = LSHIndex(num_tables=8, bits_per_table=8)
    index.build(vectors)

    ids, scores = index.search(queries[0], k=10)

    expected = normalize(vectors)[ids] @ normalize(queries[0])
    np.testing.assert_allclose(scores, expected, rtol=1e-5)
    assert list(scores) == sorted(scores, reverse=True)


def test_recall_on_clustered_data(clustered):
    vectors, queries = clustered
    exact = BruteForceIndex()
    exact.build(vectors)
    truth = exact.search_batch(queries, k=10)[0]
    index = LSHIndex(num_tables=12, bits_per_table=8)
    index.build(vectors)

    found = index.search_batch(queries, k=10)[0]

    assert recall_at_k(truth, found) > 0.7


def test_more_tables_only_add_candidates(clustered):
    vectors, queries = clustered
    few, many = LSHIndex(num_tables=3, bits_per_table=10), LSHIndex(num_tables=9, bits_per_table=10)
    few.build(vectors)
    many.build(vectors)

    for query in queries[:25]:
        assert set(few.candidates(query)) <= set(many.candidates(query))


def test_hamming_radius_one_only_adds_candidates(clustered):
    vectors, queries = clustered
    exact_bucket = LSHIndex(num_tables=4, bits_per_table=10, hamming_radius=0)
    neighboring = LSHIndex(num_tables=4, bits_per_table=10, hamming_radius=1)
    exact_bucket.build(vectors)
    neighboring.build(vectors)

    for query in queries[:25]:
        assert set(exact_bucket.candidates(query)) <= set(neighboring.candidates(query))


def test_more_bits_means_fewer_candidates(clustered):
    vectors, queries = clustered
    coarse, fine = (
        LSHIndex(num_tables=4, bits_per_table=4),
        LSHIndex(num_tables=4, bits_per_table=14),
    )
    coarse.build(vectors)
    fine.build(vectors)

    coarse_size = np.mean([len(coarse.candidates(query)) for query in queries[:50]])
    fine_size = np.mean([len(fine.candidates(query)) for query in queries[:50]])

    assert fine_size < coarse_size < len(vectors)


def test_same_seed_gives_same_index(clustered):
    vectors, queries = clustered
    first, second = LSHIndex(seed=5), LSHIndex(seed=5)
    first.build(vectors)
    second.build(vectors)

    np.testing.assert_array_equal(first.candidates(queries[0]), second.candidates(queries[0]))


def test_memory_includes_vectors_and_tables(clustered):
    vectors, _ = clustered
    brute_force, lsh = BruteForceIndex(), LSHIndex(num_tables=4, bits_per_table=8)
    brute_force.build(vectors)
    lsh.build(vectors)

    assert brute_force.memory_bytes() == vectors.astype(np.float32).nbytes
    assert lsh.memory_bytes() > brute_force.memory_bytes()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_tables": 0},
        {"bits_per_table": 0},
        {"bits_per_table": 63},
        {"hamming_radius": 2},
    ],
)
def test_invalid_parameters_are_rejected(kwargs):
    with pytest.raises(ValueError):
        LSHIndex(**kwargs)


def test_search_before_build_raises():
    with pytest.raises(RuntimeError):
        LSHIndex().search(np.zeros(8), k=1)


def test_wrong_query_dimension_raises(clustered):
    vectors, _ = clustered
    index = LSHIndex()
    index.build(vectors)

    with pytest.raises(ValueError):
        index.search(np.zeros(63), k=1)
