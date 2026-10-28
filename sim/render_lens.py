"""What the cell's cameras do to a MuJoCo render that the physics does not.

The render comes out geometrically exact and perfectly sharp; the recording it
is compared against does not. A 640x480 webcam sitting 0.47 m from the table
bends straight lines near the frame edge, falls off towards the corners, has no
resolving power at one pixel, and adds a couple of counts of sensor noise. Those
four effects are what is left of the sim-to-real gap once the lighting has been
ruled out (policy_probe_light.py), so they are applied here, in one place, after
render_sensor.py has done the colour response.

    CUBE_SIM_LENS=1   enabled (default off: the shipped checkpoint was trained
                      on the sharp renders, so this is an experiment)

Everything is deterministic per call except the noise, which is seeded off the
frame counter, so a replay of the same run gives the same images.
"""
import numpy as np

# Fitted by eye against the recorded middle frames: the straight table edge bows
# by about four pixels at the frame edge, the corners lose about a tenth, and a
# one-pixel checkerboard is gone by the time it reaches the policy.
K1 = -0.055          # barrel distortion
VIGNETTE = 0.085     # corner falloff at r = 1
BLUR = (0.25, 0.5, 0.25)
NOISE = 1.6          # counts, standard deviation

_grids = {}


def _grid(height, width):
    """Sampling coordinates and the vignette mask, computed once per size."""
    key = (height, width)
    if key in _grids:
        return _grids[key]
    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    cx, cy = (width - 1) / 2.0, (height - 1) / 2.0
    nx, ny = (x - cx) / cx, (y - cy) / cy
    r2 = nx * nx + ny * ny
    scale = 1.0 + K1 * r2
    src_x = np.clip(nx * scale * cx + cx, 0, width - 1)
    src_y = np.clip(ny * scale * cy + cy, 0, height - 1)
    x0 = np.floor(src_x).astype(np.int32)
    y0 = np.floor(src_y).astype(np.int32)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    fx = (src_x - x0)[..., None].astype(np.float32)
    fy = (src_y - y0)[..., None].astype(np.float32)
    mask = (1.0 - VIGNETTE * np.clip(r2, 0, 1.2))[..., None].astype(np.float32)
    _grids[key] = (y0 * width + x0, y0 * width + x1, y1 * width + x0,
                   y1 * width + x1, fx, fy, mask)
    return _grids[key]


def _blur(image):
    """One separable 3-tap pass: enough to take the razor edge off."""
    k0, k1, k2 = BLUR
    for axis in (0, 1):
        before = [(1, 0) if i == axis else (0, 0) for i in range(3)]
        after = [(0, 1) if i == axis else (0, 0) for i in range(3)]
        head = [slice(1, None) if i == axis else slice(None) for i in range(3)]
        tail = [slice(None, -1) if i == axis else slice(None) for i in range(3)]
        image = (k0 * np.pad(image, before, mode="edge")[tuple(head)]
                 + k1 * image
                 + k2 * np.pad(image, after, mode="edge")[tuple(tail)])
    return image


def apply(image, frame=0):
    """Distort, vignette, soften and add noise, in that order."""
    height, width = image.shape[:2]
    i00, i01, i10, i11, fx, fy, mask = _grid(height, width)
    flat = image.reshape(-1, 3).astype(np.float32)
    top = flat[i00] * (1 - fx) + flat[i01] * fx
    bottom = flat[i10] * (1 - fx) + flat[i11] * fx
    warped = (top * (1 - fy) + bottom * fy) * mask
    warped = _blur(warped)
    rng = np.random.default_rng(1000 + frame)
    warped += rng.normal(0.0, NOISE, warped.shape).astype(np.float32)
    return np.clip(warped, 0, 255).astype(np.uint8)
