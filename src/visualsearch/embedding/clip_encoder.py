"""CLIP image and text encoders that return unit-length float32 vectors."""

from collections.abc import Sequence

import numpy as np
import open_clip
import torch
from PIL import Image

from visualsearch.config import CLIP_MODEL, CLIP_PRETRAINED


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class ClipEncoder:
    """Maps images and text into one shared embedding space.

    Pass `pretrained=None` for randomly initialised weights (no download), which
    is enough to test shapes and plumbing but gives meaningless embeddings.
    """

    def __init__(
        self,
        model_name: str = CLIP_MODEL,
        pretrained: str | None = CLIP_PRETRAINED,
        device: str | None = None,
    ) -> None:
        self.device = device or pick_device()
        model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=self.device
        )
        self.model = model.eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)

    @torch.inference_mode()
    def encode_pixels(self, pixels: torch.Tensor) -> np.ndarray:
        """Embed a batch of already-preprocessed (n, 3, H, W) image tensors."""
        return _to_unit_numpy(self.model.encode_image(pixels.to(self.device)))

    def encode_images(self, images: Sequence[Image.Image]) -> np.ndarray:
        pixels = torch.stack([self.preprocess(image.convert("RGB")) for image in images])
        return self.encode_pixels(pixels)

    @torch.inference_mode()
    def encode_text(self, texts: Sequence[str]) -> np.ndarray:
        tokens = self.tokenizer(list(texts)).to(self.device)
        return _to_unit_numpy(self.model.encode_text(tokens))


def _to_unit_numpy(features: torch.Tensor) -> np.ndarray:
    features = features.float()
    features = features / features.norm(dim=-1, keepdim=True)
    return features.cpu().numpy()
