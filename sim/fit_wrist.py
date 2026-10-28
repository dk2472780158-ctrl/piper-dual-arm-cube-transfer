"""Fit a wrist camera mount to the recorded video, by silhouette overlap.

The camera is rigidly attached to the gripper, so its pose is expressed in the
gripper_base frame: position, a view direction given by a tilt off the flange
axis and an azimuth around it, a roll about the view direction, and the field
of view.

The earlier version of this file compared the *moments* of three colour masks
and settled on a mount whose picture was wrong: a wide field of view with the
mat pushed to the frame edge reproduces the same centroid and area while the
camera tips far enough to catch the wall behind the table. So the comparison is
now the silhouettes themselves -- each mask is box-downsampled and blurred, and
the cost is the mean squared difference of the three soft masks. That is
sensitive to position, size, shape and orientation at once, and it also
punishes things the recorded image does not show, such as wall where the table
should be.

Usage:  MUJOCO_GL=egl python fit_wrist.py [side] [frame] [--check f1,f2,...]
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
import imageio.v2 as imageio
import mujoco

SCENE = "/home/databall/VLA/cube_mujoco/sim/scene.xml"
DATASET = ("/home/databall/.cache/huggingface/lerobot/local/"
           "cube_r2l_dataset_v2_robust")
SIGN = {1: 1.0, 2: -1.0, 3: -1.0, 4: -1.0, 5: -1.0, 6: 1.0}
W, H = 640, 480
DOWN = 8              # masks are compared at 80x60
DARK = 0.32           # gripper ~0.20, table ~0.46 in both images now

SIDE = sys.argv[1] if len(sys.argv) > 1 else "right"
FRAME = int(sys.argv[2]) if len(sys.argv) > 2 else 0
CHECK = [int(v) for v in sys.argv[4].split(",")] if len(sys.argv) > 4 else [0]
np.set_printoptions(precision=4, suppress=True)


def box3(m):
    p = np.pad(m, 1, mode="edge")
    return (p[:-2, 1:-1] + p[1:-1, :-2] + p[1:-1, 1:-1]
            + p[1:-1, 2:] + p[2:, 1:-1]) / 5.0


def soft(mask):
    m = mask.astype(np.float32)
    m = m.reshape(H // DOWN, DOWN, W // DOWN, DOWN).mean(axis=(1, 3))
    return box3(box3(m))


def masks(image):
    """Filled mat (either colour), cube, and the dark gripper below the mats.

    The dark mask is limited to the middle columns: in the recorded left take
    the operator's hand sits at the bottom left corner, and in a render a link
    can poke into the corner once the camera is close to the wrist. Neither is
    part of the gripper silhouette the fit is trying to line up.
    """
    rgb = np.asarray(image)[:, :, :3].astype(float) / 255.0
    red, green, blue = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    columns = np.arange(W)[None, :]
    cube = (red > 0.45) & (red - green > 0.15) & (red - blue > 0.08) & (green < 0.55)
    mat = (((blue > 0.45) & (red > 0.40) & (blue - green > 0.06))
           | ((red > 0.55) & (green > 0.45) & (red - blue > 0.06)
              & (green - blue > 0.04))) & ~cube
    arm = ((rgb.max(axis=2) < DARK) & (np.arange(H)[:, None] > 240)
           & (columns > 120) & (columns < 520) & ~mat & ~cube)
    return [mat, cube, arm]


reader = imageio.get_reader(glob.glob(
    f"{DATASET}/videos/observation.images.{SIDE}/*/*.mp4")[0])
demo = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(
    f"{DATASET}/data/chunk-*/file-*.parquet"))])
demo = demo[demo.episode_index == 0].reset_index(drop=True)
states = np.stack(demo["observation.state"].values)[:, 0::2]

real_image = np.asarray(reader.get_data(FRAME))[:, :, :3]
target = [soft(m) for m in masks(real_image)]
print(f"reference frame {FRAME}: mask areas",
      [round(float(m.sum()), 1) for m in masks(real_image)])

model = mujoco.MjModel.from_xml_path(SCENE)
data = mujoco.MjData(model)
cam_id = model.camera(SIDE).id
renderer = mujoco.Renderer(model, H, W)
_quat = np.zeros(4)


def pose_arm(state14):
    for side, base in (("left", 0), ("right", 7)):
        for i in range(1, 7):
            value = SIGN[i] * state14[base + i - 1]
            data.qpos[int(np.ravel(model.joint(f"{side}_joint{i}").qposadr)[0])] = value
    for j in (7, 8):
        for side in ("left", "right"):
            data.qpos[int(np.ravel(model.joint(f"{side}_joint{j}").qposadr)[0])] = 0.0
    mujoco.mj_forward(model, data)


def view_axes(params):
    """Rows: camera x (right), y (up), z (back). The camera looks along -z."""
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


def render(params):
    model.cam_pos[cam_id] = params[:3]
    rotation = view_axes(params)
    # mju_mat2Quat reads the matrix column-major; its columns are the camera
    # axes, so the row-major flatten of the row version is the same array.
    mujoco.mju_mat2Quat(_quat, rotation.reshape(-1))
    model.cam_quat[cam_id] = _quat
    model.cam_fovy[cam_id] = params[6]
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera=cam_id)
    return renderer.render()


BOUNDS = np.array([[-0.16, 0.04], [-0.12, 0.12], [-0.10, 0.16],
                   [0.0, 80.0], [-180.0, 180.0], [-180.0, 180.0],
                   [32.0, 74.0]])
WALL = 30.0
# Each mask is normalised by its own mass. Without that, a pose that sees no
# mat and no cube at all scores as well as one that matches them loosely --
# the residual is then just the target, and a view of bare floor wins.


GRID_Y, GRID_X = np.mgrid[0:H // DOWN, 0:W // DOWN]


def moments(mask):
    """Centre, spread and tilt of a soft mask, in the downsampled grid."""
    total = float(mask.sum()) + 1e-6
    dx = GRID_X - (mask * GRID_X).sum() / total
    dy = GRID_Y - (mask * GRID_Y).sum() / total
    return np.array([(mask * dx).sum() / total, (mask * dy).sum() / total,
                     (mask * dx * dx).sum() / total,
                     (mask * dy * dy).sum() / total,
                     (mask * dx * dy).sum() / total])


TARGET_MOMENTS = [moments(b) for b in target]


def cost(params):
    """Shape error plus moment error, both relative, so a pose that matches
    neither cannot look as good as one that matches loosely.

    The pixel term alone is orientation-blind: a mat can be rotated about the
    view direction and still cover a similar area in a similar place. The
    moment term adds the covariance, which is what fixes the roll and the
    aspect, and with them the distance.
    """
    if np.any(params < BOUNDS[:, 0]) or np.any(params > BOUNDS[:, 1]):
        return WALL * WALL
    got = [soft(m) for m in masks(render(params))]
    pixel, moment = [], []
    for a, b, mb in zip(got, target, TARGET_MOMENTS):
        pixel.append(((a - b) ** 2).sum() / max(float((b * b).sum()), 1e-6))
        ma = moments(a)
        scale = np.array([GRID_X.max(), GRID_Y.max(),
                          max(mb[2], 1.0), max(mb[3], 1.0),
                          max(np.sqrt(mb[2] * mb[3]), 1.0)])
        moment.append(((ma - mb) / scale) ** 2)
    return float(np.mean(pixel) + np.mean(np.concatenate(moment)))


def nelder_mead(fun, x0, scale, max_iter=1200, tol=1e-13):
    """Simplex search in scaled coordinates; scipy is not installed here."""
    n = len(x0)
    x0 = np.asarray(x0, float)
    scale = np.asarray(scale, float)

    def evaluate(z):
        return float(fun(x0 + z * scale))

    simplex = [np.zeros(n)] + [np.eye(n)[i] * 0.5 for i in range(n)]
    values = [evaluate(z) for z in simplex]
    for _ in range(max_iter):
        order = np.argsort(values)
        simplex = [simplex[i] for i in order]
        values = [values[i] for i in order]
        if abs(values[-1] - values[0]) < tol * max(1.0, abs(values[0])):
            break
        centroid = np.mean(simplex[:-1], axis=0)
        reflected = centroid + (centroid - simplex[-1])
        f_reflected = evaluate(reflected)
        if f_reflected < values[0]:
            expanded = centroid + 2 * (centroid - simplex[-1])
            f_expanded = evaluate(expanded)
            simplex[-1], values[-1] = ((expanded, f_expanded) if f_expanded < f_reflected
                                       else (reflected, f_reflected))
        elif f_reflected < values[-2]:
            simplex[-1], values[-1] = reflected, f_reflected
        else:
            contracted = centroid + 0.5 * (simplex[-1] - centroid)
            f_contracted = evaluate(contracted)
            if f_contracted < values[-1]:
                simplex[-1], values[-1] = contracted, f_contracted
            else:
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                    values[i] = evaluate(simplex[i])
    best = int(np.argmin(values))
    result = type("Result", (), {})()
    result.x = x0 + simplex[best] * scale
    result.cost = values[best]
    return result


def from_axes(pos, direction, up, fovy):
    """Inverse of view_axes: a mount given as axes back to the search vector."""
    direction = np.asarray(direction, float)
    direction /= np.linalg.norm(direction)
    tilt = np.degrees(np.arccos(np.clip(direction[2], -1.0, 1.0)))
    azim = np.degrees(np.arctan2(direction[1], direction[0]))
    x0 = np.cross(direction, np.array([0.0, 0.0, 1.0]))
    x0 /= np.linalg.norm(x0)
    y0 = np.cross(x0, direction)
    roll = np.degrees(np.arctan2(-np.dot(up, x0), np.dot(up, y0)))
    return np.array([*pos, tilt, azim, roll, fovy])


# Each side is fitted separately: the brackets are not exact mirrors. These
# are the last accepted mounts and they seed the search, so a refit after a
# change elsewhere in the cell starts near the answer instead of rediscovering
# it from a uniform scan.
RIGHT_MOUNT = np.array([-0.0913, -0.0139, 0.0027,
                        25.264, -102.904, -108.641, 32.745])
LEFT_MOUNT = np.array([-0.1072, 0.0220, -0.0143,
                       26.257, -81.684, -77.545, 33.073])
SEED = RIGHT_MOUNT if SIDE == "right" else LEFT_MOUNT
GUESS = SEED.copy()
SCALE = np.array([0.01, 0.01, 0.01, 4.0, 15.0, 10.0, 4.0])

pose_arm(states[FRAME])
print("cost at the current mount %.4f" % cost(GUESS))
print("cost at the reference mount %.4f" % cost(SEED))

rng = np.random.default_rng(1)
scan = [(cost(SEED), SEED)]
# Half the samples are local (a few search steps around the seed) and half are
# global, so a cell change that only nudges the answer is found cheaply while a
# wrong seed can still be escaped.
count = int(os.environ.get("FIT_SCAN", "500"))
for index in range(count):
    if index % 2:
        sample = np.clip(SEED + rng.normal(size=7) * SCALE * 1.5,
                         BOUNDS[:, 0], BOUNDS[:, 1])
    else:
        sample = rng.uniform(BOUNDS[:, 0], BOUNDS[:, 1])
    scan.append((cost(sample), sample))
scan.sort(key=lambda item: item[0])
print("scan best costs", [round(c, 4) for c, _ in scan[:6]])

best = None
for value, sample in scan[:6]:
    trial = nelder_mead(cost, sample, SCALE)
    trial.cost = cost(trial.x)
    if best is None or trial.cost < best.cost:
        best = trial
    print(f"  refine from {value:8.4f} -> {trial.cost:8.4f}")

for offset in (np.zeros(7),
               np.array([0.4, -0.4, 0.3, 0.4, 0.4, 0.4, -0.3]),
               np.array([-0.4, 0.4, -0.3, -0.4, -0.4, -0.4, 0.3])):
    trial = nelder_mead(cost, best.x + offset * SCALE * 1.5, SCALE)
    trial.cost = cost(trial.x)
    if trial.cost < best.cost:
        best = trial
    print(f"  polish {np.round(offset, 1)} -> {trial.cost:8.4f}")

p = best.x
print(f"\nbest cost {best.cost:.4f}")
print("  rendered mask areas",
      [round(float(m.sum()), 1) for m in masks(render(p))])
print(f"  pos   {np.round(p[:3], 4)}")
print(f"  tilt {p[3]:7.3f} deg  azim {p[4]:8.3f} deg  roll {p[5]:8.3f} deg")
print(f"  fovy  {p[6]:7.3f} deg")
axes = view_axes(p)
print("  direction", np.round(-axes[2], 4), " up", np.round(axes[1], 4))
np.save(f"/tmp/fit_wrist_{SIDE}_{FRAME}.npy", p)

# One rigid mount has to explain every frame, so the same parameters are
# scored on other frames of the same take without re-fitting.
for frame in CHECK:
    pose_arm(states[frame])
    got = [soft(m) for m in masks(render(p))]
    target_f = [soft(m) for m in masks(np.asarray(reader.get_data(frame)))]
    err = np.mean([np.sqrt(np.mean((a - b) ** 2)) for a, b in zip(got, target_f)])
    print(f"  frame {frame:3d}  rms mask error {err:.4f}")
    imageio.imwrite(f"/tmp/fit_wrist_{SIDE}_{frame}.png",
                    np.concatenate([np.asarray(reader.get_data(frame))[:, :, :3],
                                    render(p)], axis=1))

pose_arm(states[FRAME])
print(f"wrote /tmp/fit_wrist_{SIDE}_*.png (recorded | fitted render)")
