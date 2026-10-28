"""The fitted camera response, applied to a raw MuJoCo render.

fit_render.py writes one smooth per-channel gain field per camera; this module
is the single place that applies it, so the policy, the comparison images and
the difference heatmap all see the same pixels.
"""
import json
from pathlib import Path

import numpy as np

MODEL_PATH = Path(__file__).resolve().parent / "render_model.json"


def load(path=None):
    path = Path(path) if path else MODEL_PATH
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as handle:
        return {name: np.asarray(value, float)
                for name, value in json.load(handle).items()}


def apply(model, name, image):
    """Scale a render by the field fitted for that camera; passthrough if none."""
    coefficients = model.get(name)
    if coefficients is None:
        return image
    height, width = image.shape[:2]
    u = (np.arange(width) - (width - 1) / 2) / ((width - 1) / 2)
    v = (np.arange(height) - (height - 1) / 2) / ((height - 1) / 2)
    u, v = np.meshgrid(u, v)
    basis = np.stack([np.ones_like(u), u, v, u * u, u * v, v * v], axis=-1)
    pixels = image.astype(float)
    out = np.stack([
        (basis * pixels[..., c][..., None]) @ coefficients[c] for c in range(3)],
        axis=-1)
    return np.clip(out, 0, 255).astype(np.uint8)
