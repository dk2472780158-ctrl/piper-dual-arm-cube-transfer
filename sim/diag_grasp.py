"""Why does the cube stay put when the fingers close on it?

Replays the recorded demonstration through the physics and, around the grasp
frames, prints the finger opening the actuators actually achieve, the contact
list between the right fingers and the cube, and the vertical span of the
finger pad meshes next to the cube's. A grasp that never happens is one of:
the pads do not bracket the cube, the pads never meet it, or the pads meet it
but the friction cannot hold 30 g.

    MUJOCO_GL=egl python diag_grasp.py
"""
import glob

import numpy as np
import pandas as pd
import mujoco

SCENE = "/home/databall/VLA/cube_mujoco/sim/scene.xml"
REAL = ("/home/databall/.cache/huggingface/lerobot/local/"
        "cube_r2l_dataset_v2_robust/data/chunk-*/file-*.parquet")
SIGN = {1: 1.0, 2: -1.0, 3: -1.0, 4: -1.0, 5: -1.0, 6: 1.0}
LEFT_START = np.asarray([0.0492107794, 0.0, 0.0, -0.2287141085, 0.2685223222,
                         0.2780382633, 0.0])
WATCH = (180, 195, 210, 225, 240, 255, 270, 285, 300)
np.set_printoptions(precision=4, suppress=True)

df = pd.read_parquet(sorted(glob.glob(REAL))[0])
df = df[df.episode_index == 0].reset_index(drop=True)
actions = np.stack(df["action"].values)

m = mujoco.MjModel.from_xml_path(SCENE)
d = mujoco.MjData(m)
cube_body = m.body("cube").id


def pad_points(body):
    """Every vertex of the body's meshes, in world coordinates."""
    bid = m.body(body).id
    out = []
    for g in range(m.body_geomadr[bid], m.body_geomadr[bid] + m.body_geomnum[bid]):
        if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        h = m.geom_dataid[g]
        a, n = m.mesh_vertadr[h], m.mesh_vertnum[h]
        R = d.xmat[bid].reshape(3, 3)
        out.append(m.mesh_vert[a:a + n] @ R.T + d.xpos[bid])
    return np.concatenate(out)


def finger_contacts():
    hits = []
    for k in range(d.ncon):
        c = d.contact[k]
        b1 = m.geom_bodyid[c.geom1]
        b2 = m.geom_bodyid[c.geom2]
        n1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b1) or "world"
        n2 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b2) or "world"
        if "cube" in (n1, n2) and (n1, n2) != ("cube", "cube"):
            other = n2 if n1 == "cube" else n1
            hits.append(f"{other} @ {np.round(c.pos, 4)}")
    return hits


def drive(pose7):
    for i in range(1, 7):
        d.ctrl[m.actuator(f"right_act_{i}").id] = SIGN[i] * float(pose7[i - 1])
    g = float(np.clip(pose7[6], 0.0, 0.07))
    d.ctrl[m.actuator("right_act_7").id] = g / 2
    d.ctrl[m.actuator("right_act_8").id] = -g / 2


mujoco.mj_resetData(m, d)
for i in range(1, 7):
    d.qpos[int(np.ravel(m.joint(f"left_joint{i}").qposadr)[0])] = (
        SIGN[i] * LEFT_START[i - 1])
    d.ctrl[m.actuator(f"left_act_{i}").id] = SIGN[i] * LEFT_START[i - 1]
for k in (7, 8):
    d.qpos[int(np.ravel(m.joint(f"left_joint{k}").qposadr)[0])] = 0.0
mujoco.mj_forward(m, d)

for f in range(len(actions)):
    drive(actions[f][7:14])
    for _ in range(17):
        mujoco.mj_step(m, d)
    if f not in WATCH:
        continue
    o7 = float(d.qpos[int(np.ravel(m.joint("right_joint7").qposadr)[0])])
    o8 = float(d.qpos[int(np.ravel(m.joint("right_joint8").qposadr)[0])])
    print(f"f{f:4d} cmd grip {actions[f][13]:.4f}  actual {o7 - o8:.4f}  "
          f"cube {np.round(d.xpos[cube_body], 3)}  z {d.xpos[cube_body][2]:.4f}")
    for body in ("right_link7", "right_link8"):
        p = pad_points(body)
        print(f"      {body:11s} x [{p[:,0].min():.4f},{p[:,0].max():.4f}] "
              f"y [{p[:,1].min():.4f},{p[:,1].max():.4f}] "
              f"z [{p[:,2].min():.4f},{p[:,2].max():.4f}]")
    for hit in finger_contacts():
        print(f"      contact {hit}")

print("\ncube geoms / friction:")
for g in range(m.ngeom):
    b = m.geom_bodyid[g]
    name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or "world"
    if name in ("cube", "right_link7", "right_link8"):
        print(f"  {name:11s} geom {g:3d} type {int(m.geom_type[g])} "
              f"friction {np.round(m.geom_friction[g], 3)} "
              f"contype {int(m.geom_contype[g])} conaffinity {int(m.geom_conaffinity[g])}")
