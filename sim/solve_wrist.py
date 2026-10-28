"""Solve the wrist camera pose from the two things the real image shows.

At the training start pose the recorded right-wrist frame puts the two fingers
at the bottom centre and the purple mat with the cube at the left edge. The
camera mount is described in the gripper_base frame; the orientation that
projects those two targets where the recording shows them is found by search.
"""
import numpy as np
import mujoco

SCENE = "/home/databall/VLA/cube_mujoco/sim/scene.xml"
SIGN = {1: 1.0, 2: -1.0, 3: -1.0, 4: -1.0, 5: -1.0, 6: 1.0}
START = {"left": [0.0492107794, 0.0, 0.0, -0.2287141085, 0.2685223222, 0.2780382633],
         "right": [-0.0443961099, 0.0, 0.0, 0.0536503866, 0.3397218585, -0.0396250561]}
FOVY = 60.0
W, H = 640, 480
np.set_printoptions(precision=4, suppress=True)

m = mujoco.MjModel.from_xml_path(SCENE)
d = mujoco.MjData(m)
for side, pose in START.items():
    for i in range(1, 7):
        d.qpos[int(np.ravel(m.joint(f"{side}_joint{i}").qposadr)[0])] = SIGN[i] * pose[i - 1]
mujoco.mj_forward(m, d)


def targets(side):
    base = m.body(f"{side}_gripper_base").id
    o = d.xpos[base]
    R = d.xmat[base].reshape(3, 3)          # world = R @ local + o
    finger = d.xpos[m.body(f"{side}_link7").id]
    cube = d.xpos[m.body("cube").id]
    sign = -1.0 if side == "right" else 1.0
    return R.T @ (finger - o), R.T @ (cube - o), sign


def project(p, dvec, up, target):
    z = -np.asarray(dvec, float)
    z /= np.linalg.norm(z)
    x = np.cross(np.asarray(up, float), z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    v = np.asarray(target, float) - np.asarray(p, float)
    # the camera looks along -z, so depth is -(z . v)
    depth = -(z @ v)
    if depth <= 0.01:
        return None
    u = (x @ v) / depth / np.tan(np.radians(FOVY) / 2 * W / H)
    vv = (y @ v) / depth / np.tan(np.radians(FOVY) / 2)
    return np.array([u, vv])          # NDC, +1 = right / up edge


def basis(az, el, roll):
    dvec = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    ref = np.array([0.0, 0.0, 1.0])
    if abs(dvec @ ref) > 0.9:
        ref = np.array([1.0, 0.0, 0.0])
    u0 = ref - (ref @ dvec) * dvec
    u0 /= np.linalg.norm(u0)
    w0 = np.cross(dvec, u0)
    up = np.cos(roll) * u0 + np.sin(roll) * w0
    return dvec, up


def solve(p, finger, cube, side):
    # recorded framing, in NDC: fingers bottom centre, cube at the left edge
    f_tgt = np.array([0.0, -0.55])
    c_tgt = np.array([-0.72, -0.05]) * (1 if side == "right" else -1)
    best = None
    for az in np.radians(np.arange(-180, 180, 4)):
        for el in np.radians(np.arange(-80, 81, 4)):
            for roll in np.radians(np.arange(-180, 180, 10)):
                dvec, up = basis(az, el, roll)
                pf = project(p, dvec, up, finger)
                pc = project(p, dvec, up, cube)
                if pf is None or pc is None:
                    continue
                err = ((pf - f_tgt) @ (pf - f_tgt) + (pc - c_tgt) @ (pc - c_tgt))
                if best is None or err < best[0]:
                    best = (err, az, el, roll, pf, pc)
    return best


for side in ("right",):
    finger, cube, sign = targets(side)
    print(side, "finger local", np.round(finger, 4), "cube local", np.round(cube, 4))
    for p in (np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.03]),
              np.array([0.02, 0.0, 0.02]), np.array([-0.02, 0.0, 0.02])):
        err, az, el, roll, pf, pc = solve(p, finger, cube, side)
        print(f"  pos {np.round(p, 3)} err {err:.3f} az {np.degrees(az):+.0f} "
              f"el {np.degrees(el):+.0f} roll {np.degrees(roll):+.0f} "
              f"finger {np.round(pf, 2)} cube {np.round(pc, 2)}")
