import numpy as np

from visualsearch.index import Index


class ShortResultsIndex(Index):
    """Stub approximate index that returns fewer results than requested."""

    def build(self, vectors):
        pass

    def search(self, query, k):
        return np.array([7, 3]), np.array([0.9, 0.5], dtype=np.float32)

    def memory_bytes(self):
        return 0


def test_search_batch_pads_short_results():
    ids, scores = ShortResultsIndex().search_batch(np.zeros((2, 4)), k=4)

    assert ids.tolist() == [[7, 3, -1, -1], [7, 3, -1, -1]]
    assert scores[0, :2].tolist() == [np.float32(0.9), np.float32(0.5)]
    assert np.isneginf(scores[:, 2:]).all()
