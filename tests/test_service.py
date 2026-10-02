import numpy as np
import pytest
from PIL import Image

from visualsearch.index import BruteForceIndex
from visualsearch.service import ItemNotFoundError, SearchService, UnknownIndexError


def test_text_search_returns_matching_category_best_first(service):
    result = service.search_text("a red thing", k=5)

    assert result.index == "brute-force"
    assert [hit.category for hit in result.hits] == ["red"] * 5
    scores = [hit.score for hit in result.hits]
    assert scores == sorted(scores, reverse=True)
    assert result.embed_ms >= 0
    assert result.search_ms >= 0


def test_image_search_finds_the_same_colour(service):
    result = service.search_image(Image.new("RGB", (50, 50), (20, 30, 230)), k=4)

    assert {hit.category for hit in result.hits} == {"blue"}


def test_every_index_can_be_selected(service):
    for name in service.index_names:
        result = service.search_text("green", k=3, index=name)

        assert result.index == name
        assert len(result.hits) <= 3


def test_unknown_index_is_rejected(service):
    with pytest.raises(UnknownIndexError, match="nope"):
        service.search_text("red", k=3, index="nope")


def test_similar_to_excludes_the_item_itself(service):
    item_id = 3

    result = service.similar_to(item_id, k=5)

    assert item_id not in [hit.id for hit in result.hits]
    assert len(result.hits) == 5
    assert {hit.category for hit in result.hits} == {service.records[item_id].category}
    assert result.embed_ms == 0


@pytest.mark.parametrize("item_id", [-1, 24, 10_000])
def test_missing_ids_raise(service, item_id):
    with pytest.raises(ItemNotFoundError):
        service.similar_to(item_id, k=3)
    with pytest.raises(ItemNotFoundError):
        service.image_path(item_id)


def test_image_size_matches_the_file(service):
    width, height = service.image_size(0)

    with Image.open(service.image_path(0)) as image:
        assert (width, height) == image.size


def test_k_larger_than_catalog_returns_everything(service):
    assert len(service.search_text("red", k=500).hits) == len(service)


def test_mismatched_records_and_vectors_are_rejected(service):
    with pytest.raises(ValueError, match="records"):
        SearchService(
            service.records[:5],
            np.zeros((6, 8), dtype=np.float32),
            service._encoder,
            {"brute-force": BruteForceIndex()},
            service._images_dir,
            default_index="brute-force",
        )


def test_default_index_must_exist(service):
    with pytest.raises(ValueError, match="default index"):
        SearchService(
            service.records,
            np.zeros((len(service), 8), dtype=np.float32),
            service._encoder,
            {"brute-force": BruteForceIndex()},
            service._images_dir,
            default_index="hnsw",
        )
