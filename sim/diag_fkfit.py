"""Match the URDF chain to the SDK DH chain.

For each of the 64 joint sign combinations, fit small per-joint offsets by
Gauss-Newton so the URDF flange positions agree with the SDK flange positions
under a constant rigid transform (Kabsch). The right combination and offsets
leave a residual near zero.
"""
import itertools
import sys

import numpy as np
import mujoco
from piper_sdk import C_PiperForwardKinematics

SCENE = "/home/databall/VLA/cube_mujoco/sim/scene.xml"

m = mujoco.MjModel.from_xml_path(SCENE)
d = mujoco.MjData(m)
fk = C_PiperForwardKinematics()
rng = np.random.default_rng(0)

LOW = np.array([-2.6, 0.0, -2.6, -1.7, -1.2, -2.0])
HIGH = np.array([2.6, 3.0, 0.0, 1.7, 1.2, 2.0])
SAMPLES = 24
qs = [rng.uniform(LOW, HIGH) for _ in range(SAMPLES)]
sdk = np.asarray([np.asarray(fk.CalFK(list(q))[-1][:3]) / 1000.0 for q in qs])

adr = [int(np.ravel(m.joint(f"right_joint{i}").qposadr)[0]) for i in range(1, 7)]
bid = m.body("right_link6").id
mount = m.body("right_mount").id


def urdf_positions(signs, deltas):
    out = np.empty((SAMPLES, 3))
    for k, q in enumerate(qs):
        for i in range(6):
            d.qpos[adr[i]] = signs[i] * q[i] + deltas[i]
        mujoco.mj_forward(m, d)
        Rm = d.xmat[mount].reshape(3, 3)
        out[k] = Rm.T @ (d.xpos[bid] - d.xpos[mount])
    return out


def kabsch(src, dst):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    H = (src - mu_s).T @ (dst - mu_d)
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    return R, mu_d - R @ mu_s


def residual(signs, deltas):
    mj = urdf_positions(signs, deltas)
    R, t = kabsch(mj, sdk)
    return ((mj @ R.T + t) - sdk).ravel()


def refine(signs, iters=25):
    x = np.zeros(6)
    step = 1e-4
    best = float(np.sqrt(np.mean(residual(signs, x) ** 2)))
    for _ in range(iters):
        f0 = residual(signs, x)
        J = np.empty((f0.size, 6))
        for j in range(6):
            dx = np.zeros(6)
            dx[j] = step
            J[:, j] = (residual(signs, x + dx) - f0) / step
        delta = -np.linalg.solve(J.T @ J + 1e-6 * np.eye(6), J.T @ f0)
        delta = np.clip(delta, -0.6, 0.6)
        trial = x + delta
        err = float(np.sqrt(np.mean(residual(signs, trial) ** 2)))
        if err < best:
            best, x = err, trial
        else:
            break
    return best, x


results = []
for signs in itertools.product((1.0, -1.0), repeat=6):
    rms, deltas = refine(signs)
    results.append((rms, signs, deltas))
    if rms < 0.002:
        print(f"  {rms:.6f} signs {signs} deltas {np.round(deltas, 5).tolist()}")

results.sort(key=lambda r: r[0])
print()
for rms, signs, deltas in results[:4]:
    print(f"rms {rms:.6f} signs {signs}")
    print(f"   deltas rad {np.round(deltas, 5).tolist()}")
    print(f"   deltas deg {np.round(np.degrees(deltas), 3).tolist()}")

rms, signs, deltas = results[0]
mj = urdf_positions(signs, deltas)
R, t = kabsch(mj, sdk)
err = np.linalg.norm((mj @ R.T + t) - sdk, axis=1)
print("best max err %.5f m" % err.max())
print("R =\n", np.round(R, 5))
print("t =", np.round(t, 5))
