"""Search logic behind the API: turn a query into a vector, run an index, attach metadata."""

import threading
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from time import perf_counter
from typing import Protocol

import numpy as np
from PIL import Image

from visualsearch.config import EMBEDDINGS_PATH, IMAGES_DIR, METADATA_PATH
from visualsearch.data.caltech256 import Record, read_metadata
from visualsearch.embedding.store import load_embeddings
from visualsearch.index import BruteForceIndex, HNSWIndex, Index, LSHIndex


class Encoder(Protocol):
    """Anything that maps images and text into the catalog's embedding space."""

    def encode_images(self, images: Sequence[Image.Image]) -> np.ndarray: ...

    def encode_text(self, texts: Sequence[str]) -> np.ndarray: ...


class UnknownIndexError(ValueError):
    """The requested index name is not one this service offers."""


class ItemNotFoundError(LookupError):
    """No catalog item has the requested id."""


@dataclass(frozen=True)
class Hit:
    id: int
    score: float
    category: str


@dataclass(frozen=True)
class SearchResult:
    index: str
    hits: list[Hit]
    embed_ms: float  # time to embed the query (0 when searching from a catalog item)
    search_ms: float  # time spent inside the index


class SearchService:
    def __init__(
        self,
        records: list[Record],
        vectors: np.ndarray,
        encoder: Encoder,
        indexes: dict[str, Index],
        images_dir: Path,
        default_index: str,
    ) -> None:
        if len(records) != len(vectors):
            raise ValueError(f"{len(records)} records but {len(vectors)} vectors")
        if default_index not in indexes:
            raise ValueError(f"default index {default_index!r} is not among {list(indexes)}")
        self.records = records
        self.index_names = list(indexes)
        self.default_index = default_index
        self._vectors = vectors
        self._encoder = encoder
        self._indexes = indexes
        self._images_dir = Path(images_dir)
        # The encoder is a shared model and HNSW changes its search width per query,
        # so concurrent requests take turns instead of racing.
        self._encoder_lock = threading.Lock()
        self._index_locks = {name: threading.Lock() for name in indexes}

    def __len__(self) -> int:
        return len(self.records)

    def search_text(self, text: str, k: int, index: str | None = None) -> SearchResult:
        start = perf_counter()
        with self._encoder_lock:
            vector = self._encoder.encode_text([text])[0]
        return self._search(vector, k, index, embed_ms=(perf_counter() - start) * 1000)

    def search_image(self, image: Image.Image, k: int, index: str | None = None) -> SearchResult:
        start = perf_counter()
        with self._encoder_lock:
            vector = self._encoder.encode_images([image])[0]
        return self._search(vector, k, index, embed_ms=(perf_counter() - start) * 1000)

    def similar_to(self, item_id: int, k: int, index: str | None = None) -> SearchResult:
        """Catalog items most like the given one, not including the item itself."""
        self._check_id(item_id)
        return self._search(self._vectors[item_id], k, index, embed_ms=0.0, exclude=item_id)

    def image_path(self, item_id: int) -> Path:
        self._check_id(item_id)
        return self._images_dir / self.records[item_id].path

    def image_size(self, item_id: int) -> tuple[int, int]:
        return _read_image_size(str(self.image_path(item_id)))

    def _search(
        self,
        vector: np.ndarray,
        k: int,
        index_name: str | None,
        embed_ms: float,
        exclude: int | None = None,
    ) -> SearchResult:
        name = index_name or self.default_index
        if name not in self._indexes:
            raise UnknownIndexError(f"unknown index {name!r}; choose from {self.index_names}")
        wanted = k + 1 if exclude is not None else k
        with self._index_locks[name]:
            start = perf_counter()
            ids, scores = self._indexes[name].search(vector, wanted)
            search_ms = (perf_counter() - start) * 1000
        hits = [
            Hit(int(item_id), float(score), self.records[item_id].category)
            for item_id, score in zip(ids, scores, strict=True)
            if item_id != exclude
        ][:k]
        return SearchResult(name, hits, embed_ms, search_ms)

    def _check_id(self, item_id: int) -> None:
        if not 0 <= item_id < len(self.records):
            raise ItemNotFoundError(f"no image with id {item_id}")


@lru_cache(maxsize=8192)
def _read_image_size(path: str) -> tuple[int, int]:
    with Image.open(path) as image:  # reads only the file header
        return image.size


def load_service() -> SearchService:
    """Build the real service from the files produced by the scripts in scripts/."""
    # Imported here so that tests and tools that don't need CLIP don't pay for torch.
    from visualsearch.embedding.clip_encoder import ClipEncoder

    if not IMAGES_DIR.is_dir():
        raise FileNotFoundError(f"{IMAGES_DIR} not found; run scripts/download_data.py first")
    records = read_metadata(METADATA_PATH)
    vectors, info = load_embeddings(EMBEDDINGS_PATH)
    indexes: dict[str, Index] = {
        "hnsw": HNSWIndex(M=16, ef_construction=200, ef_search=64),
        "lsh": LSHIndex(num_tables=32, bits_per_table=14, hamming_radius=1),
        "brute-force": BruteForceIndex(),
    }
    for index in indexes.values():
        index.build(vectors)
    encoder = ClipEncoder(info["model"], info["pretrained"])
    return SearchService(records, vectors, encoder, indexes, IMAGES_DIR, default_index="hnsw")
