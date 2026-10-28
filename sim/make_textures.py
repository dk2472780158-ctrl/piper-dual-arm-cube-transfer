"""Generate the small modulation textures the realism scene puts on the cell.

The albedo of every surface was calibrated against recorded frames
(calibrate_colors.py), so the maps here are centred on TEX_MEAN rather than
255 and carry a few percent of noise: they break up the flat plastic look and
leave the mean colour alone (build_scene.py scales the material rgba by
255 / TEX_MEAN to compensate). Each map is seeded, so a rebuild is identical.

    python make_textures.py            # writes textures/*.png next to this file
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "textures"
TEX_MEAN = 250


def smooth_noise(rng, shape, radius):
    """Low-frequency blotches: white noise blurred by a separable Gaussian."""
    field = rng.normal(0.0, 1.0, (shape, shape))
    kernel = np.exp(-0.5 * (np.arange(-3 * radius, 3 * radius + 1) / radius) ** 2)
    kernel /= kernel.sum()
    for axis in (0, 1):
        field = np.apply_along_axis(
            lambda row: np.convolve(row, kernel, mode="same"), axis, field)
    return field / (np.abs(field).max() + 1e-9)


def table(rng):
    """Matte grey laminate: fine grain over a few broad tone shifts."""
    n = 512
    fine = rng.normal(0.0, 1.0, (n, n))
    grain = fine / (np.abs(fine).max() + 1e-9)
    tone = smooth_noise(rng, n, max(1, n // 32))
    return TEX_MEAN + 5.0 * grain + 4.0 * tone


def save(name, array):
    import imageio.v2 as imageio
    OUT.mkdir(exist_ok=True)
    path = OUT / name
    image = np.repeat(np.clip(array, 200, 255).astype(np.uint8)[..., None], 3, axis=2)
    imageio.imwrite(path, image)
    print(f"{path}  mean {image.mean():.1f}  spread {image.std():.1f}")


def main():
    save("table.png", table(np.random.default_rng(7)))


if __name__ == "__main__":
    main()
