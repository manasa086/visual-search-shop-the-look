"""Paths and model settings shared by the scripts, CLI and (later) the API."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("VISUALSEARCH_DATA_DIR", ROOT / "data"))

RAW_DIR = DATA_DIR / "raw"
IMAGES_DIR = RAW_DIR / "256_ObjectCategories"
METADATA_PATH = DATA_DIR / "metadata.csv"
EMBEDDINGS_PATH = DATA_DIR / "embeddings.npy"

CLIP_MODEL = "ViT-B-32"
CLIP_PRETRAINED = "laion2b_s34b_b79k"
