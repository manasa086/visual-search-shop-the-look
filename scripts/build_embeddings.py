"""Embed every image listed in data/metadata.csv with CLIP and save data/embeddings.npy."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from visualsearch.config import (
    CLIP_MODEL,
    CLIP_PRETRAINED,
    EMBEDDINGS_PATH,
    IMAGES_DIR,
    METADATA_PATH,
)
from visualsearch.data.caltech256 import Record, read_metadata
from visualsearch.embedding.clip_encoder import ClipEncoder
from visualsearch.embedding.store import save_embeddings


class ImageDataset(Dataset):
    def __init__(self, root: Path, records: list[Record], preprocess) -> None:
        self.root = root
        self.records = records
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        path = self.root / self.records[index].path
        with Image.open(path) as image:
            return self.preprocess(image.convert("RGB"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4, help="image-decoding processes")
    parser.add_argument("--device", help="cuda, mps or cpu (default: best available)")
    args = parser.parse_args()

    records = read_metadata(METADATA_PATH)
    encoder = ClipEncoder(device=args.device)
    loader = DataLoader(
        ImageDataset(IMAGES_DIR, records, encoder.preprocess),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    batches = [encoder.encode_pixels(pixels) for pixels in tqdm(loader, desc="Embedding")]
    vectors = np.concatenate(batches)
    save_embeddings(
        EMBEDDINGS_PATH,
        vectors,
        {
            "model": CLIP_MODEL,
            "pretrained": CLIP_PRETRAINED,
            "count": len(vectors),
            "dim": vectors.shape[1],
        },
    )
    print(f"Saved {vectors.shape} embeddings to {EMBEDDINGS_PATH}")


if __name__ == "__main__":
    main()
