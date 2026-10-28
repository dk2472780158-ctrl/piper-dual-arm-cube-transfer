#!/usr/bin/env python3
"""Contact sheet of candidate wrist-camera mounts, rendered at the start pose.

The real rig has no calibration file on the machine, so the wrist camera angle is
recovered by eye against the recorded dataset frame. Each candidate is baked into
a temporary copy of scene.xml (same directory, so meshdir still resolves).
"""
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TRAINING_START_POSE = np.asarray([
    0.0492107794, 0.0, 0.0, -0.2287141085, 0.2685223222, 0.2780382633, 0.0,
    -0.0443961099, 0.0, 0.0, 0.0536503866, 0.3397218585, -0.0396250561, 0.0,
])

# Local +x is world-up and local +z runs along the fingers, so the bracket sits above
# the gripper base and aims forward along the approach axis.
POSITIONS = [(0.10, 0.0, 0.00), (0.10, 0.0, 0.04), (0.12, 0.0, -0.02),
             (0.10, 0.05, 0.00), (0.10, -0.05, 0.00), (0.13, 0.0, 0.05)]
UP_TILTS = [-15.0, -25.0, -35.0]
SIDE_TILTS = [0.0]
FOVY = 50.0


def look_at_xyaxes(direction, up):
    d = np.asarray(direction, float)
    d /= np.linalg.norm(d)
    z = -d
    x = np.cross(np.asarray(up, float), z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return " ".join(f"{v:.6g}" for v in [*x, *y])


def bake(scene_text: str, pos, axes) -> str:
    pattern = re.compile(r'<camera name="left"[^/]*/>')
    replacement = (f'<camera name="left" pos="{" ".join(f"{v:.6g}" for v in pos)}" '
                   f'xyaxes="{axes}" fovy="{FOVY}"/>')
    out, count = pattern.subn(replacement, scene_text)
    if count != 1:
        raise RuntimeError(f"expected exactly one left camera, found {count}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default=str(HERE / "scene.xml"))
    ap.add_argument("--out", default=str(HERE.parent / "out" / "camera_search.png"))
    ap.add_argument("--size", type=int, default=260)
    args = ap.parse_args()

    import mujoco
    from PIL import Image, ImageDraw

    scene_text = Path(args.scene).read_text()
    tmp = Path(args.scene).with_name("_search_scene.xml")
    renderer = None
    tiles, labels = [], []
    import math as _m
    for offset in POSITIONS:
        for up_tilt in UP_TILTS:
            for side_tilt in SIDE_TILTS:
                a, b = _m.radians(up_tilt), _m.radians(side_tilt)
                direction = [_m.sin(a), _m.sin(b), _m.cos(a) * _m.cos(b)]
                axes = look_at_xyaxes(direction, [1.0, 0.0, 0.0])
                tmp.write_text(bake(scene_text, offset, axes))
                model = mujoco.MjModel.from_xml_path(str(tmp))
                data = mujoco.MjData(model)
                for name in ("left", "right"):
                    for index in range(1, 7):
                        joint = model.joint(f"{name}_joint{index}")
                        data.qpos[joint.qposadr] = TRAINING_START_POSE[
                            (0 if name == "left" else 7) + index - 1]
                mujoco.mj_forward(model, data)
                renderer = mujoco.Renderer(model, args.size, args.size)
                renderer.update_scene(data, camera="left")
                tiles.append(Image.fromarray(renderer.render()))
                renderer.close()
                labels.append(f"p{offset} up{up_tilt:.0f} sd{side_tilt:.0f}")
                del data, model
    tmp.unlink(missing_ok=True)
    if renderer is not None:
        renderer.close()

    cols = 3
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * args.size, rows * (args.size + 14)), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (tile, label) in enumerate(zip(tiles, labels)):
        row, col = divmod(index, cols)
        sheet.paste(tile, (col * args.size, row * (args.size + 14)))
        draw.text((col * args.size + 2, row * (args.size + 14) + args.size + 1), label, fill="black")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.out)
    print("wrote", args.out, sheet.size, len(tiles), "candidates")


if __name__ == "__main__":
    main()
