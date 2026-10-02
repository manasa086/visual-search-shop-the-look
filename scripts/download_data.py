"""Download and unpack the Caltech-256 dataset (about 1.18 GB) into data/raw/."""

import tarfile
import urllib.request

from tqdm import tqdm

from visualsearch.config import IMAGES_DIR, RAW_DIR

URL = "https://data.caltech.edu/records/nyy15-4j048/files/256_ObjectCategories.tar?download=1"
EXPECTED_BYTES = 1_183_006_720
ARCHIVE = RAW_DIR / "256_ObjectCategories.tar"


def download() -> None:
    partial = ARCHIVE.with_suffix(".part")
    request = urllib.request.Request(URL, headers={"User-Agent": "visual-search-project"})
    with urllib.request.urlopen(request) as response, partial.open("wb") as out:
        with tqdm(total=EXPECTED_BYTES, unit="B", unit_scale=True, desc=ARCHIVE.name) as bar:
            while chunk := response.read(1 << 20):
                out.write(chunk)
                bar.update(len(chunk))
    if partial.stat().st_size != EXPECTED_BYTES:
        raise RuntimeError(
            f"download is {partial.stat().st_size} bytes, expected {EXPECTED_BYTES}; "
            f"delete {partial} and retry"
        )
    partial.rename(ARCHIVE)


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if not ARCHIVE.exists():
        download()
    if not IMAGES_DIR.exists():
        print(f"Extracting to {RAW_DIR} ...")
        with tarfile.open(ARCHIVE) as tar:
            tar.extractall(RAW_DIR, filter="data")
    print(f"Images ready in {IMAGES_DIR}")


if __name__ == "__main__":
    main()
