"""Plumbing checks for the encoder using randomly initialised weights (nothing is downloaded)."""

import numpy as np
import pytest
from PIL import Image

from visualsearch.embedding.clip_encoder import ClipEncoder


@pytest.fixture(scope="module")
def encoder() -> ClipEncoder:
    return ClipEncoder(pretrained=None, device="cpu")


def test_images_and_text_share_a_unit_length_space(encoder):
    images = [Image.new("RGB", (300, 200), color) for color in ("red", "blue")]

    image_vectors = encoder.encode_images(images)
    text_vectors = encoder.encode_text(["a red square", "a blue square", "a cat"])

    assert image_vectors.shape == (2, 512)
    assert text_vectors.shape == (3, 512)
    assert image_vectors.dtype == text_vectors.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(image_vectors, axis=1), 1.0, rtol=1e-5)
    np.testing.assert_allclose(np.linalg.norm(text_vectors, axis=1), 1.0, rtol=1e-5)


def test_grayscale_images_are_accepted(encoder):
    assert encoder.encode_images([Image.new("L", (64, 64), 128)]).shape == (1, 512)
