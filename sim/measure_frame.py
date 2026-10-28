"""Measure the mats and the cube in a recorded middle-camera frame.

Colour segmentation gives the centroid and the second-moment extents of each
mat and of the cube. The cube-to-mat size ratio is what pins the cell's real
dimensions: both sit at nearly the same distance from the camera, so their
projected sizes scale the same way and the ratio is geometry-independent.
"""
import sys

import numpy as np
import imageio.v2 as imageio

path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/real_middle.png"
image = np.asarray(imageio.imread(path))[:, :, :3].astype(float) / 255.0
h, w = image.shape[:2]
red, green, blue = image[..., 0], image[..., 1], image[..., 2]
print(f"{path}  {w}x{h}")

masks = {
    "yellow_mat": (red > 0.60) & (green > 0.55) & (blue < 0.62) & (red - blue > 0.12),
    "purple_mat": (blue > 0.55) & (red > 0.45) & (green < 0.75) & (blue - green > 0.10),
    "cube": (red > 0.55) & (red - green > 0.18) & (red - blue > 0.12) & (green < 0.62),
}
for name, mask in masks.items():
    ys, xs = np.nonzero(mask)
    if len(xs) < 50:
        print(f"{name:11s} not found ({len(xs)} px)")
        continue
    # Second moments give the semi-axes of the fitted ellipse.
    cx, cy = xs.mean(), ys.mean()
    cov = np.cov(np.stack([xs - cx, ys - cy]))
    eigenvalues = np.sort(np.linalg.eigvalsh(cov))[::-1]
    semi_major, semi_minor = 2 * np.sqrt(eigenvalues)
    print(f"{name:11s} px={len(xs):6d} centre=({cx:6.1f},{cy:6.1f}) "
          f"bbox x[{xs.min()}-{xs.max()}] y[{ys.min()}-{ys.max()}] "
          f"axis_major={semi_major:5.1f} axis_minor={semi_minor:5.1f} "
          f"area={np.pi * semi_major * semi_minor:8.1f}")

my = masks["yellow_mat"]
mp = masks["purple_mat"]
mk = masks["cube"]
if my.sum() and mp.sum() and mk.sum():
    def width(mask):
        xs = np.nonzero(mask)[1]
        return xs.max() - xs.min() + 1

    mat_px = 0.5 * (width(my) + width(mp))
    cube_px = width(mk)
    print(f"\nmat width {mat_px:.1f} px, cube width {cube_px:.1f} px, "
          f"ratio {cube_px / mat_px:.3f}")
    print("cube edge = mat diameter * ratio; so cube/radius = "
          f"{2 * cube_px / mat_px:.3f}")
