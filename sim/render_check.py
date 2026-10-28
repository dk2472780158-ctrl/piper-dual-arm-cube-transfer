#!/usr/bin/env python3
"""Load scene.xml, put both arms at the training start pose, render every camera.

Writes PNGs plus a short report of where the fingertips end up, which is what
tells us whether the arm mounts and the table height are plausible.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

TRAINING_START_POSE = np.asarray([
    0.0492107794, 0.0, 0.0, -0.2287141085, 0.2685223222, 0.2780382633, 0.0,
    -0.0443961099, 0.0, 0.0, 0.0536503866, 0.3397218585, -0.0396250561, 0.0,
])


def sensor_sites(model, data):
    report = {}
    for side in ("left", "right"):
        for link in ("gripper_base", "link7", "link8", "link6"):
            body_id = model.body(f"{side}_{link}").id
            report[f"{side}_{link}"] = np.round(data.xpos[body_id], 4).tolist()
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default=str(HERE / "scene.xml"))
    ap.add_argument("--out", default=str(HERE.parent / "out"))
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    args = ap.parse_args()

    import mujoco

    model = mujoco.MjModel.from_xml_path(args.scene)
    data = mujoco.MjData(model)
    print("nq", model.nq, "nu", model.nu, "ngeom", model.ngeom, "ncam", model.ncam)

    for name in ("left", "right"):
        for index in range(1, 7):
            joint = model.joint(f"{name}_joint{index}")
            value = TRAINING_START_POSE[(0 if name == "left" else 7) + index - 1]
            data.qpos[joint.qposadr] = value
    mujoco.mj_forward(model, data)

    for key, value in sensor_sites(model, data).items():
        print(f"{key:20s} {value}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    renderer = mujoco.Renderer(model, args.height, args.width)
    for cam in ("middle", "left", "right"):
        renderer.update_scene(data, camera=cam)
        rgb = renderer.render()
        from PIL import Image
        Image.fromarray(rgb).save(out / f"sim_{cam}.png")
        print("wrote", out / f"sim_{cam}.png", rgb.mean().round(1))
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0.0, 0.0, 0.75]
    cam.distance = 1.35
    cam.azimuth = 90.0
    cam.elevation = -55.0
    renderer.update_scene(data, camera=cam)
    from PIL import Image
    Image.fromarray(renderer.render()).save(out / "sim_free.png")
    print("wrote", out / "sim_free.png")
    renderer.close()


if __name__ == "__main__":
    main()
