"""Convert canonical robot data objects into model-ready arrays."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict

import numpy as np

from ewpl.data.schemas import Observation, Step


@dataclass
class TensorizedStep:
    image: np.ndarray
    proprio: np.ndarray
    language_ids: np.ndarray
    action: np.ndarray


def tensorize_observation(
    observation: Observation,
    *,
    image_size: int = 32,
    language_length: int = 16,
    vocab_size: int = 4096,
) -> Dict[str, np.ndarray]:
    """Turn one canonical observation into normalized arrays."""

    return {
        "image": normalize_image(observation.rgb, image_size=image_size),
        "proprio": np.asarray(observation.proprio, dtype=np.float32).reshape(-1),
        "language_ids": tokenize_language(
            observation.language,
            length=language_length,
            vocab_size=vocab_size,
        ),
    }


def tensorize_step(
    step: Step,
    *,
    image_size: int = 32,
    language_length: int = 16,
    vocab_size: int = 4096,
) -> TensorizedStep:
    obs = tensorize_observation(
        step.observation,
        image_size=image_size,
        language_length=language_length,
        vocab_size=vocab_size,
    )
    return TensorizedStep(
        image=obs["image"],
        proprio=obs["proprio"],
        language_ids=obs["language_ids"],
        action=np.asarray(step.action.vector, dtype=np.float32).reshape(-1),
    )


def normalize_image(rgb, *, image_size: int) -> np.ndarray:
    """Normalize RGB to float32 CHW in [0, 1] with nearest-neighbor resizing."""

    image = np.asarray(rgb)
    if image.ndim != 3:
        raise ValueError(f"expected RGB image with 3 dims, got shape {image.shape}")
    if image.shape[0] in {1, 3, 4} and image.shape[-1] not in {1, 3, 4}:
        image = np.moveaxis(image, 0, -1)
    image = image[..., :3]
    image = _resize_nearest(image, image_size, image_size)
    return (image.astype(np.float32) / 255.0).transpose(2, 0, 1)


def tokenize_language(text: str, *, length: int, vocab_size: int) -> np.ndarray:
    """Hash-tokenize text into deterministic integer ids with zero padding."""

    if length <= 0:
        raise ValueError("language token length must be positive")
    if vocab_size <= 1:
        raise ValueError("vocab_size must be greater than 1")
    tokens = []
    for word in text.lower().split():
        digest = hashlib.sha1(word.encode("utf-8")).hexdigest()
        tokens.append((int(digest[:8], 16) % (vocab_size - 1)) + 1)
    padded = np.zeros(length, dtype=np.int64)
    padded[: min(length, len(tokens))] = tokens[:length]
    return padded


def _resize_nearest(image: np.ndarray, height: int, width: int) -> np.ndarray:
    y_idx = np.linspace(0, image.shape[0] - 1, num=height).round().astype(np.int64)
    x_idx = np.linspace(0, image.shape[1] - 1, num=width).round().astype(np.int64)
    return image[y_idx][:, x_idx]

