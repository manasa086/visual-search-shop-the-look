# Visual Search: "Shop the Look"

A visual search engine: give it an image (or a text description) and it returns the most
similar images from a catalog. Images are turned into CLIP embeddings, and the interesting
part of the project is the nearest-neighbor search over them. It compares an exact
brute-force baseline, an LSH index written from scratch, and an HNSW reference.

Dataset: [Caltech-256](https://data.caltech.edu/records/nyy15-4j048) (about 30k images, 256 object categories).

## Status

- [x] Project setup, brute-force index, CLIP encoder, data and embedding scripts, CLI
- [ ] LSH index and benchmark harness (recall@k, latency, memory, build time)
- [ ] FastAPI service
- [ ] React + TypeScript frontend
- [ ] HNSW comparison, Docker, CI, results write-up

## Quickstart

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync                                   # create the environment
uv run python scripts/download_data.py    # download and unpack Caltech-256 (about 1.2 GB)
uv run python scripts/build_metadata.py   # list images into data/metadata.csv
uv run python scripts/build_embeddings.py # embed every image with CLIP
uv run vsearch --image path/to/photo.jpg  # search by image
uv run vsearch --text "a bright red guitar" -k 5
```

For a quick trial run, use `build_metadata.py --limit 2000` before embedding.

## How it works

```
images -> CLIP image encoder -> unit vectors (n x 512) -> Index.build()
query (image or text) -> CLIP encoder -> unit vector -> Index.search(k) -> ranked ids
```

All indexes implement the same `Index` interface (`src/visualsearch/index/base.py`) and
normalize vectors themselves, so cosine similarity is a dot product and any index can be
swapped in. `BruteForceIndex` is exact and provides the ground truth for measuring the
approximate indexes.

## Development

```bash
uv run pytest        # tests
uv run ruff check .  # lint
```

Datasets, embeddings and benchmark outputs live in `data/`, which is git-ignored.
