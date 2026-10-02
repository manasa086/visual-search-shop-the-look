"""Image catalog for the Caltech-256 dataset.

The dataset unpacks to one folder per category (`001.ak47/`, `002.american-flag/`,
...) plus a `257.clutter/` folder of background images that we skip by default.
"""

import csv
import random
from dataclasses import astuple, dataclass, fields
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
CLUTTER_FOLDER = "257.clutter"


@dataclass(frozen=True)
class Record:
    id: int
    path: str  # relative to the images directory
    category: str


def category_name(folder: str) -> str:
    """'001.ak47' -> 'ak47'."""
    return folder.split(".", 1)[-1]


def build_metadata(
    images_dir: Path,
    limit: int | None = None,
    seed: int = 0,
    include_clutter: bool = False,
) -> list[Record]:
    """List the images under `images_dir`, optionally keeping a random sample of `limit`.

    Ids are consecutive from 0 in path order, and match the row order of the
    embedding matrix built from the same list.
    """
    images_dir = Path(images_dir)
    if not images_dir.is_dir():
        raise FileNotFoundError(f"{images_dir} not found; run scripts/download_data.py first")

    paths = sorted(
        path
        for path in images_dir.glob("*/*")
        if path.suffix.lower() in IMAGE_SUFFIXES
        and (include_clutter or path.parent.name != CLUTTER_FOLDER)
    )
    if limit is not None and limit < len(paths):
        paths = sorted(random.Random(seed).sample(paths, limit))

    return [
        Record(i, path.relative_to(images_dir).as_posix(), category_name(path.parent.name))
        for i, path in enumerate(paths)
    ]


def write_metadata(records: list[Record], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(field.name for field in fields(Record))
        writer.writerows(astuple(record) for record in records)


def read_metadata(path: Path) -> list[Record]:
    with Path(path).open(newline="") as f:
        return [Record(int(row["id"]), row["path"], row["category"]) for row in csv.DictReader(f)]
