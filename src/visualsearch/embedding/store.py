"""Saving and loading embedding matrices together with the model that made them."""

import json
from pathlib import Path

import numpy as np


def _info_path(path: Path) -> Path:
    return path.with_suffix(".json")


def save_embeddings(path: Path, vectors: np.ndarray, info: dict) -> None:
    """Write `vectors` to `path` (.npy) and `info` to a JSON file next to it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, vectors.astype(np.float32, copy=False))
    _info_path(path).write_text(json.dumps(info, indent=2) + "\n")


def load_embeddings(path: Path) -> tuple[np.ndarray, dict]:
    vectors = np.load(path)
    info = json.loads(_info_path(path).read_text())
    return vectors, info
