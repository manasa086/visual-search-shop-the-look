"""A tiny colour-based catalog and encoder, so API tests need no CLIP weights or dataset."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from visualsearch.data.caltech256 import Record
from visualsearch.index import BruteForceIndex, HNSWIndex, LSHIndex
from visualsearch.index.utils import normalize
from visualsearch.service import SearchService

COLORS = {"red": (210, 30, 30), "green": (30, 200, 40), "blue": (30, 40, 220)}
IMAGES_PER_CATEGORY = 8
DIM = 8


class ColorEncoder:
    """Stands in for CLIP: images embed by average colour, text by the colour words in it."""

    def encode_images(self, images) -> np.ndarray:
        means = [
            np.asarray(image.convert("RGB"), dtype=np.float32).mean(axis=(0, 1)) for image in images
        ]
        return normalize(np.array([self._embed(mean / 255) for mean in means]))

    def encode_text(self, texts) -> np.ndarray:
        words = ("red", "green", "blue")
        flags = [[float(word in text.lower()) for word in words] for text in texts]
        return normalize(np.array([self._embed(np.array(row)) for row in flags]))

    @staticmethod
    def _embed(rgb: np.ndarray) -> np.ndarray:
        vector = np.full(DIM, 1e-3, dtype=np.float32)  # keeps the vector non-zero
        vector[:3] = rgb
        return vector


@pytest.fixture(scope="session")
def catalog_dir(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("catalog")
    for name, color in COLORS.items():
        (root / name).mkdir()
        for i in range(IMAGES_PER_CATEGORY):
            shade = 0.8 + 0.05 * i  # small brightness variation within a category
            size = (40 + 4 * i, 30 + 2 * i)
            pixel = tuple(min(255, int(channel * shade)) for channel in color)
            Image.new("RGB", size, pixel).save(root / name / f"{i}.jpg", quality=100)
    return root


@pytest.fixture(scope="session")
def service(catalog_dir) -> SearchService:
    paths = sorted(catalog_dir.glob("*/*.jpg"))
    records = [
        Record(i, p.relative_to(catalog_dir).as_posix(), p.parent.name) for i, p in enumerate(paths)
    ]
    encoder = ColorEncoder()
    vectors = encoder.encode_images([Image.open(path) for path in paths])
    indexes = {
        "brute-force": BruteForceIndex(),
        "hnsw": HNSWIndex(M=4, ef_construction=50, ef_search=32),
        "lsh": LSHIndex(num_tables=8, bits_per_table=4, hamming_radius=1),
    }
    for index in indexes.values():
        index.build(vectors)
    return SearchService(
        records, vectors, encoder, indexes, catalog_dir, default_index="brute-force"
    )
