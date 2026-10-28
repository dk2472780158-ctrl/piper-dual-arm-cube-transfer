"""Which joint signs turn the right arm's pose into the left arm's mirror?

There is a recording for the right-to-left pass but none for the left-to-right
one. The two arms are mounted as mirror images, so the same joint vector
reflected in y is a legitimate left-arm pose provided the sign of each joint
is right: a mirror reverses the joints whose axis is not perpendicular to it.

The arm is put in the recorded start pose, and every sign combination is tried
on the left arm. The one whose fingertip lands on the mirror of the right
fingertip is the answer.

    python diag_mirror.py
"""
import glob
import itertools

import numpy as np
import pandas as pd
import mujoco

SCENE = "/home/databall/VLA/cube_mujoco/sim/scene.xml"
REAL = ("/home/databall/.cache/huggingface/lerobot/local/"
        "cube_r2l_dataset_v2_robust/data/chunk-*/file-*.parquet")
SIGN = {1: 1.0, 2: -1.0, 3: -1.0, 4: -1.0, 5: -1.0, 6: 1.0}
np.set_printoptions(precision=4, suppress=True)

df = pd.read_parquet(sorted(glob.glob(REAL))[0])
df = df[df.episode_index == 0].reset_index(drop=True)
actions = np.stack(df["action"].values)

m = mujoco.MjModel.from_xml_path(SCENE)
d = mujoco.MjData(m)


def joint(side, i):
    return int(np.ravel(m.joint(f"{side}_joint{i}").qposadr)[0])


def tip(side):
    return 0.5 * (d.xpos[m.body(f"{side}_link7").id] +
                  d.xpos[m.body(f"{side}_link8").id])


def rot(side):
    return d.xmat[m.body(f"{side}_gripper_base").id].reshape(3, 3).copy()


MIRROR = np.diag([1.0, -1.0, 1.0])


def angle_between(a, b):
    """Rotation angle of a⁻¹b, in radians."""
    return float(np.arccos(np.clip((np.trace(a.T @ b) - 1.0) / 2.0, -1.0, 1.0)))


for frame in (0, 60, 150, 200, 255):
    pose = np.asarray(actions[frame][7:13], float)
    q = np.array([SIGN[i] * pose[i - 1] for i in range(1, 7)])
    for i in range(1, 7):
        d.qpos[joint("right", i)] = q[i - 1]
    mujoco.mj_forward(m, d)
    right_tip = tip("right").copy()
    target_tip = MIRROR @ right_tip
    target_rot = MIRROR @ rot("right") @ MIRROR

    best = []
    for signs in itertools.product((1.0, -1.0), repeat=6):
        for i in range(1, 7):
            d.qpos[joint("left", i)] = signs[i - 1] * q[i - 1]
        mujoco.mj_forward(m, d)
        pos_err = float(np.linalg.norm(tip("left") - target_tip))
        rot_err = angle_between(target_rot, rot("left"))
        best.append((pos_err + 0.10 * rot_err, pos_err, rot_err, signs))
    best.sort(key=lambda item: item[0])
    print(f"f{frame:4d} q6 {q[5]:+.3f}  right tip {np.round(right_tip, 4)}")
    for score, pos_err, rot_err, signs in best[:3]:
        print(f"      pos {1000 * pos_err:7.2f} mm  rot {np.degrees(rot_err):6.2f}°"
              "  signs " + " ".join(f"{s:+.0f}" for s in signs))
