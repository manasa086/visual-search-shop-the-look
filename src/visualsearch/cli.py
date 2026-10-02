"""Command-line search: `vsearch --image photo.jpg` or `vsearch --text "a red shoe"`."""

import argparse
import sys
from pathlib import Path
from time import perf_counter

from PIL import Image

from visualsearch.config import EMBEDDINGS_PATH, IMAGES_DIR, METADATA_PATH
from visualsearch.data.caltech256 import read_metadata
from visualsearch.embedding.store import load_embeddings
from visualsearch.index import BruteForceIndex


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="vsearch",
        description="Find catalog images similar to an image or a text description. "
        "Querying with a catalog image returns that image first, with score 1.0.",
    )
    query = parser.add_mutually_exclusive_group(required=True)
    query.add_argument("--image", type=Path, help="path to a query image")
    query.add_argument("--text", help="text description to search for")
    parser.add_argument("-k", type=int, default=10, help="number of results (default: 10)")
    args = parser.parse_args(argv)

    records = read_metadata(METADATA_PATH)
    vectors, info = load_embeddings(EMBEDDINGS_PATH)
    if len(records) != len(vectors):
        print(
            f"metadata has {len(records)} rows but embeddings has {len(vectors)}; "
            "rebuild the embeddings with scripts/build_embeddings.py",
            file=sys.stderr,
        )
        return 1

    # Imported late so --help and argument errors don't wait on torch.
    from visualsearch.embedding.clip_encoder import ClipEncoder

    index = BruteForceIndex()
    index.build(vectors)
    encoder = ClipEncoder(info["model"], info["pretrained"])

    if args.image:
        with Image.open(args.image) as image:
            vector = encoder.encode_images([image])[0]
    else:
        vector = encoder.encode_text([args.text])[0]

    start = perf_counter()
    ids, scores = index.search(vector, args.k)
    elapsed_ms = (perf_counter() - start) * 1000

    for rank, (item_id, score) in enumerate(zip(ids, scores, strict=True), start=1):
        record = records[item_id]
        print(f"{rank:>2}. {score:.3f}  {record.category:<20} {IMAGES_DIR / record.path}")
    print(f"\nsearched {len(index)} images in {elapsed_ms:.1f} ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
