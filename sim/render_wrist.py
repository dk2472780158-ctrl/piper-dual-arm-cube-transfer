"""Render the solved wrist-camera pose so it can be compared with the data."""
import sys

import numpy as np
import mujoco
import imageio.v2 as imageio

SCENE = "/home/databall/VLA/cube_mujoco/sim/scene.xml"
SIGN = {1: 1.0, 2: -1.0, 3: -1.0, 4: -1.0, 5: -1.0, 6: 1.0}
START = {"left": [0.0492107794, 0.0, 0.0, -0.2287141085, 0.2685223222, 0.2780382633],
         "right": [-0.0443961099, 0.0, 0.0, 0.0536503866, 0.3397218585, -0.0396250561]}
POS = np.array([0.02, 0.0, 0.02])
AZ, EL, ROLL = np.radians(144.0), np.radians(64.0), np.radians(160.0)


def basis(az, el, roll):
    dvec = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    ref = np.array([0.0, 0.0, 1.0])
    if abs(dvec @ ref) > 0.9:
        ref = np.array([1.0, 0.0, 0.0])
    u0 = ref - (ref @ dvec) * dvec
    u0 /= np.linalg.norm(u0)
    w0 = np.cross(dvec, u0)
    return dvec, np.cos(roll) * u0 + np.sin(roll) * w0


m = mujoco.MjModel.from_xml_path(SCENE)
d = mujoco.MjData(m)
for side, pose in START.items():
    for i in range(1, 7):
        d.qpos[int(np.ravel(m.joint(f"{side}_joint{i}").qposadr)[0])] = SIGN[i] * pose[i - 1]
mujoco.mj_forward(m, d)

q = np.zeros(4)
renderer = mujoco.Renderer(m, 480, 640)
for side, mirror in (("right", 1.0), ("left", -1.0)):
    dvec, up = basis(AZ, EL, ROLL)
    if mirror < 0:
        dvec = dvec * np.array([1, -1, 1])
        up = up * np.array([1, -1, 1])
    z = -dvec / np.linalg.norm(dvec)
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    mujoco.mju_mat2Quat(q, np.column_stack([x, y, z]).ravel())
    cid = m.camera(side).id
    m.cam_pos[cid] = POS * np.array([1, mirror, 1])
    m.cam_quat[cid] = q
    mujoco.mj_forward(m, d)          # refresh d.cam_xpos for the new camera pose
    renderer.update_scene(d, camera=side)
    imageio.imwrite(f"/tmp/wrist_{side}.png", renderer.render())
    print(side, "dir", np.round(dvec, 4), "up", np.round(up, 4))
print("wrote /tmp/wrist_right.png /tmp/wrist_left.png")
