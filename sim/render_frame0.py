"""Render the three simulated cameras at the start pose, for side-by-side checks.

Writes /tmp/sim_frame0_middle.png and /tmp/sim_frame0_cams.png (the three views
stacked) so they can be compared with the recorded frame 0.
"""
import numpy as np
import mujoco
import imageio.v2 as imageio

from piper_dual_sim import START_POSE, JOINT_SIGN

SCENE = "/home/databall/VLA/cube_mujoco/sim/scene.xml"
m = mujoco.MjModel.from_xml_path(SCENE)
d = mujoco.MjData(m)
for base, side in ((0, "left"), (7, "right")):
    for i in range(1, 7):
        d.qpos[int(np.ravel(m.joint(f"{side}_joint{i}").qposadr)[0])] = (
            JOINT_SIGN[i] * START_POSE[base + i - 1])
mujoco.mj_forward(m, d)

renderer = mujoco.Renderer(m, 480, 640)
tiles = []
for name in ("left", "middle", "right"):
    renderer.update_scene(d, camera=name)
    frame = renderer.render().copy()
    imageio.imwrite(f"/tmp/sim_frame0_{name}.png", frame)
    tiles.append(frame)
imageio.imwrite("/tmp/sim_frame0_cams.png", np.concatenate(tiles, axis=0))
print("wrote /tmp/sim_frame0_middle.png and /tmp/sim_frame0_cams.png")
