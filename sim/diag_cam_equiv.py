"""Does the XML camera equal the one fit_wrist sets at runtime?

fit_wrist writes the mount into the model at run time; build_scene writes the
same numbers into scene.xml as `pos` + `xyaxes`. If those two disagree the fit
is not what gets rendered, so they are rendered back to back here and the
difference is reported along with both poses.
"""
import glob

import numpy as np
import pandas as pd
import mujoco

SCENE = "/home/databall/VLA/cube_mujoco/sim/scene.xml"
DATASET = ("/home/databall/.cache/huggingface/lerobot/local/"
           "cube_r2l_dataset_v2_robust")
SIGN = {1: 1.0, 2: -1.0, 3: -1.0, 4: -1.0, 5: -1.0, 6: 1.0}
np.set_printoptions(precision=4, suppress=True)


def view_axes(params):
    tilt, azim, roll = (np.radians(v) for v in params[3:6])
    direction = np.array([np.sin(tilt) * np.cos(azim),
                          np.sin(tilt) * np.sin(azim),
                          np.cos(tilt)])
    x = np.cross(direction, np.array([0.0, 0.0, 1.0]))
    if np.linalg.norm(x) < 1e-9:
        x = np.array([1.0, 0.0, 0.0])
    x /= np.linalg.norm(x)
    y = np.cross(x, direction)
    cr, sr = np.cos(roll), np.sin(roll)
    x, y = cr * x + sr * y, -sr * x + cr * y
    return np.stack([x, y, -direction])


model = mujoco.MjModel.from_xml_path(SCENE)
data = mujoco.MjData(model)
df = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(
    f"{DATASET}/data/chunk-*/file-*.parquet"))])
df = df[df.episode_index == 0].reset_index(drop=True)
state = np.stack(df["observation.state"].values)[0][0::2]
for side, base in (("left", 0), ("right", 7)):
    for i in range(1, 7):
        data.qpos[int(np.ravel(model.joint(f"{side}_joint{i}").qposadr)[0])] = (
            SIGN[i] * state[base + i - 1])
mujoco.mj_forward(model, data)

renderer = mujoco.Renderer(model, 480, 640)
quat = np.zeros(4)
for side in ("left", "right"):
    cam = model.camera(side).id
    renderer.update_scene(data, camera=side)
    xml_image = renderer.render().copy()
    xml_pos = model.cam_pos[cam].copy()
    xml_mat = data.cam_xmat[cam].reshape(3, 3).copy()

    params = np.load(f"/tmp/fit_wrist_{side}_0.npy")
    model.cam_pos[cam] = params[:3]
    mujoco.mju_mat2Quat(quat, view_axes(params).reshape(-1))
    model.cam_quat[cam] = quat
    model.cam_fovy[cam] = params[6]
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera=side)
    fit_image = renderer.render().copy()
    fit_mat = data.cam_xmat[cam].reshape(3, 3).copy()

    print(f"{side}: mean |xml - fit| = "
          f"{np.abs(xml_image.astype(float) - fit_image).mean():.2f}")
    print("  xml axes (rows x, y, z)\n", np.round(xml_mat, 3))
    print("  fit axes (rows x, y, z)\n", np.round(fit_mat, 3))
    print("  xml look direction", np.round(-xml_mat[2], 4),
          " fit look direction", np.round(-fit_mat[2], 4))

    # put the XML camera back for the next side
    model.cam_pos[cam] = xml_pos
    mujoco.mju_mat2Quat(quat, xml_mat.reshape(-1))
    model.cam_quat[cam] = quat
    mujoco.mj_forward(model, data)
