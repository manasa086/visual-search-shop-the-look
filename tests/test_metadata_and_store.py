from pathlib import Path

import numpy as np
import pytest

from visualsearch.data.caltech256 import (
    build_metadata,
    category_name,
    read_metadata,
    write_metadata,
)
from visualsearch.embedding.store import load_embeddings, save_embeddings


@pytest.fixture
def images_dir(tmp_path: Path) -> Path:
    layout = {
        "001.ak47": ["001_0001.jpg", "001_0002.jpg", "001_0003.jpg"],
        "002.american-flag": ["002_0001.jpg", "002_0002.JPG"],
        "198.spider": ["198_0001.jpg", "RENAME2"],  # the real dataset ships a stray file
        "257.clutter": ["257_0001.jpg", "257_0002.jpg"],
    }
    for folder, names in layout.items():
        (tmp_path / folder).mkdir()
        for name in names:
            (tmp_path / folder / name).write_bytes(b"")
    return tmp_path


def test_category_name_strips_numeric_prefix():
    assert category_name("001.ak47") == "ak47"
    assert category_name("002.american-flag") == "american-flag"


def test_build_metadata_skips_clutter_and_non_images(images_dir):
    records = build_metadata(images_dir)

    assert [record.id for record in records] == list(range(6))
    assert {record.category for record in records} == {"ak47", "american-flag", "spider"}
    assert all("clutter" not in record.path and "RENAME2" not in record.path for record in records)


def test_build_metadata_can_include_clutter(images_dir):
    assert len(build_metadata(images_dir, include_clutter=True)) == 8


def test_limit_is_a_deterministic_sample(images_dir):
    first = build_metadata(images_dir, limit=4, seed=1)
    second = build_metadata(images_dir, limit=4, seed=1)

    assert len(first) == 4
    assert first == second
    assert [record.id for record in first] == [0, 1, 2, 3]


def test_missing_images_dir_gives_a_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="download_data"):
        build_metadata(tmp_path / "nope")


def test_metadata_csv_round_trip(images_dir, tmp_path):
    records = build_metadata(images_dir)
    path = tmp_path / "out" / "metadata.csv"

    write_metadata(records, path)

    assert read_metadata(path) == records


def test_embeddings_round_trip(tmp_path):
    vectors = np.random.default_rng(0).normal(size=(5, 8)).astype(np.float32)
    path = tmp_path / "emb" / "embeddings.npy"

    save_embeddings(path, vectors, {"model": "test", "count": 5})
    loaded, info = load_embeddings(path)

    np.testing.assert_array_equal(loaded, vectors)
    assert info == {"model": "test", "count": 5}
