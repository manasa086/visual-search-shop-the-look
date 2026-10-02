"""List the dataset's images into data/metadata.csv (optionally a random subset)."""

import argparse

from visualsearch.config import IMAGES_DIR, METADATA_PATH
from visualsearch.data.caltech256 import build_metadata, write_metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, help="keep a random sample of this many images")
    parser.add_argument("--seed", type=int, default=0, help="seed for --limit sampling")
    parser.add_argument("--include-clutter", action="store_true", help="keep the clutter folder")
    args = parser.parse_args()

    records = build_metadata(IMAGES_DIR, args.limit, args.seed, args.include_clutter)
    write_metadata(records, METADATA_PATH)
    categories = len({record.category for record in records})
    print(f"Wrote {len(records)} images across {categories} categories to {METADATA_PATH}")


if __name__ == "__main__":
    main()
