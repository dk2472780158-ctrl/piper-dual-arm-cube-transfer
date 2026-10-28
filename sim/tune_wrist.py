"""Pick a wrist-camera mount: solve orientation per offset, then render a montage.

Each candidate offset in the gripper_base frame gets the orientation that best
puts the fingertips at the bottom centre and the cube at the left edge, as the
recorded start frame does. The montage shows whether the arm's own geometry
blocks the view.
"""
import numpy as np
import mujoco
import imageio.v2 as imageio

SCENE = "/home/databall/VLA/cube_mujoco/sim/scene.xml"
SIGN = {1: 1.0, 2: -1.0, 3: -1.0, 4: -1.0, 5: -1.0, 6: 1.0}
START = {"left": [0.0492107794, 0.0, 0.0, -0.2287141085, 0.2685223222, 0.2780382633],
         "right": [-0.0443961099, 0.0, 0.0, 0.0536503866, 0.3397218585, -0.0396250561]}
FOVY, W, H = 60.0, 640, 480
F_TGT = np.array([0.0, -0.55])
C_TGT = np.array([-0.72, -0.05])

m = mujoco.MjModel.from_xml_path(SCENE)
d = mujoco.MjData(m)
for side, pose in START.items():
    for i in range(1, 7):
        d.qpos[int(np.ravel(m.joint(f"{side}_joint{i}").qposadr)[0])] = SIGN[i] * pose[i - 1]
mujoco.mj_forward(m, d)

base = m.body("right_gripper_base").id
o = d.xpos[base]
R = d.xmat[base].reshape(3, 3)
FINGER = R.T @ (d.xpos[m.body("right_link7").id] - o)
CUBE = R.T @ (d.xpos[m.body("cube").id] - o)


def basis(az, el, roll):
    dvec = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    ref = np.array([0.0, 0.0, 1.0])
    if abs(dvec @ ref) > 0.9:
        ref = np.array([1.0, 0.0, 0.0])
    u0 = ref - (ref @ dvec) * dvec
    u0 /= np.linalg.norm(u0)
    w0 = np.cross(dvec, u0)
    return dvec, np.cos(roll) * u0 + np.sin(roll) * w0


def project(p, dvec, up, target):
    z = -np.asarray(dvec, float)
    z /= np.linalg.norm(z)
    x = np.cross(np.asarray(up, float), z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    v = np.asarray(target, float) - np.asarray(p, float)
    depth = -(z @ v)
    if depth <= 0.01:
        return None
    return np.array([(x @ v) / depth / np.tan(np.radians(FOVY) / 2 * W / H),
                     (y @ v) / depth / np.tan(np.radians(FOVY) / 2)])


def solve(p):
    best = None
    for az in np.radians(np.arange(-180, 180, 5)):
        for el in np.radians(np.arange(-80, 81, 5)):
            for roll in np.radians(np.arange(-180, 180, 8)):
                dvec, up = basis(az, el, roll)
                pf = project(p, dvec, up, FINGER)
                pc = project(p, dvec, up, CUBE)
                if pf is None or pc is None:
                    continue
                err = ((pf - F_TGT) @ (pf - F_TGT) + (pc - C_TGT) @ (pc - C_TGT))
                if best is None or err < best[0]:
                    best = (err, dvec, up)
    return best


OFFSETS = [(-0.06, 0.0, 0.02), (-0.09, 0.0, 0.0), (-0.06, 0.05, 0.02),
           (-0.06, -0.05, 0.02), (-0.12, 0.0, 0.04), (-0.05, 0.0, 0.08),
           (-0.08, 0.0, -0.04), (-0.10, 0.04, 0.06), (-0.04, 0.06, -0.02),
           (-0.14, -0.02, 0.02)]

cid = m.camera("right").id
renderer = mujoco.Renderer(m, 360, 480)
q = np.zeros(4)
tiles = []
for off in OFFSETS:
    p = np.array(off)
    err, dvec, up = solve(p)
    z = -dvec / np.linalg.norm(dvec)
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    mujoco.mju_mat2Quat(q, np.column_stack([x, y, z]).ravel())
    m.cam_pos[cid] = p
    m.cam_quat[cid] = q
    mujoco.mj_forward(m, d)
    renderer.update_scene(d, camera="right")
    tiles.append(renderer.render().copy())
    print(f"pos {off} err {err:.3f} dir {np.round(dvec, 3)} up {np.round(up, 3)}")

cols = 5
rows = (len(tiles) + cols - 1) // cols
h, w = tiles[0].shape[:2]
canvas = np.zeros((rows * h, cols * w, 3), np.uint8)
for i, t in enumerate(tiles):
    r, c = divmod(i, cols)
    canvas[r * h:(r + 1) * h, c * w:(c + 1) * w] = t
imageio.imwrite("/tmp/wrist_cands.png", canvas)
print("wrote /tmp/wrist_cands.png", canvas.shape)
