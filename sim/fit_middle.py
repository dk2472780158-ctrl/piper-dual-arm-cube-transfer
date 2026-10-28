"""Fit the middle camera and the cell's true sizes to a recorded frame.

Inputs are the measurements `measure_frame.py` takes from the real frame 0:
the projected centres and bounding boxes of both mats and of the cube. The
unknowns are the camera pose, its vertical field of view, the mat radius and
the cube edge. The table height and the mat separation are fixed because they
were solved from the recorded fingertip poses, not from the image.

Usage:  python fit_middle.py <measured.json>
"""
import json
import sys

import numpy as np


def nelder_mead(fun, x0, scale, max_iter=4000, tol=1e-12):
    """Simplex search in units of `scale`; scipy is not installed here.

    The parameters mix metres and radians and differ by two orders of
    magnitude, so the search runs in scaled coordinates where every step is
    O(1).
    """
    n = len(x0)
    x0 = np.asarray(x0, float)
    scale = np.asarray(scale, float)

    def evaluate(z):
        return fun(x0 + z * scale)

    simplex = [np.zeros(n)]
    for i in range(n):
        point = np.zeros(n)
        point[i] = 0.6
        simplex.append(point)
    values = [float(evaluate(z) @ evaluate(z)) for z in simplex]
    for _ in range(max_iter):
        order = np.argsort(values)
        simplex = [simplex[i] for i in order]
        values = [values[i] for i in order]
        if abs(values[-1] - values[0]) < tol * max(1.0, abs(values[0])):
            break
        centroid = np.mean(simplex[:-1], axis=0)
        reflected = centroid + (centroid - simplex[-1])
        f_reflected = float(evaluate(reflected) @ evaluate(reflected))
        if f_reflected < values[0]:
            expanded = centroid + 2 * (centroid - simplex[-1])
            f_expanded = float(evaluate(expanded) @ evaluate(expanded))
            simplex[-1], values[-1] = ((expanded, f_expanded) if f_expanded < f_reflected
                                       else (reflected, f_reflected))
        elif f_reflected < values[-2]:
            simplex[-1], values[-1] = reflected, f_reflected
        else:
            contracted = centroid + 0.5 * (simplex[-1] - centroid)
            f_contracted = float(evaluate(contracted) @ evaluate(contracted))
            if f_contracted < values[-1]:
                simplex[-1], values[-1] = contracted, f_contracted
            else:
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                    values[i] = float(evaluate(simplex[i]) @ evaluate(simplex[i]))
    best = int(np.argmin(values))
    result = type("Result", (), {})()
    result.z = simplex[best]
    result.x = x0 + simplex[best] * scale
    result.cost = values[best]
    return result

TABLE_H = 0.712
MAT_Y = 0.111          # each mat centre, metres from the table centre line
MAT_T = 0.024          # mat thickness
W, H = 640, 480

# Measured on the real frame: centre x, centre y, bbox x0, x1, y0, y1.
MEAS = {
    "yellow_mat": (243.2, 300.7, 164, 322, 240, 361),
    "purple_mat": (500.4, 304.5, 414, 587, 238, 365),
    "cube": (501.2, 279.6, 476, 527, 254, 307),
}


def rot(yaw, pitch, roll):
    """Camera axes as rows: right, up, forward.

    `yaw` turns away from +x about world z, `pitch` tilts the view down from
    horizontal, and `roll` spins the image about the viewing direction. The
    image's right-hand axis is the viewer's right, so it is view x up, not the
    other way round -- getting that backwards mirrors the whole picture.
    """
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    forward = np.array([cp * cy, cp * sy, -sp])
    right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    basis = np.stack([right, up, forward])
    cr, sr = np.cos(roll), np.sin(roll)
    return np.array([[cr, -sr, 0.0], [sr, cr, 0.0], [0.0, 0.0, 1.0]]) @ basis


def project(points, cam, fovy):
    """Camera looks along +z of its own frame; image x right, y down."""
    rel = points - cam["pos"]
    local = rel @ cam["R"].T        # rows of R are the camera x, y, z axes
    depth = local[:, 2]
    if np.any(depth <= 0.02):
        return None
    scale = (H / 2) / np.tan(fovy / 2)      # fovy is already in radians
    u = W / 2 + scale * local[:, 0] / depth
    v = H / 2 - scale * local[:, 1] / depth
    return np.stack([u, v], axis=1)


def circle(centre, radius, z, n=48):
    angle = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([centre[0] + radius * np.cos(angle),
                     centre[1] + radius * np.sin(angle),
                     np.full(n, z)], axis=1)


def cube_points(centre, edge):
    half = edge / 2
    corners = np.array([[sx, sy, sz] for sx in (-half, half)
                        for sy in (-half, half) for sz in (-half, half)])
    return corners + centre


def predict(params):
    (cx, cy, cz, yaw, pitch, roll, fovy, mat_r, cube_a) = params
    cam = {"pos": np.array([cx, cy, cz]), "R": rot(yaw, pitch, roll)}
    out = {}
    for name, y in (("yellow_mat", MAT_Y), ("purple_mat", -MAT_Y)):
        top = circle((0.0, y), mat_r, TABLE_H + MAT_T)
        bottom = circle((0.0, y), mat_r, TABLE_H)
        pts = project(np.vstack([top, bottom]), cam, fovy)
        if pts is None:
            return None
        out[name] = (pts[:, 0].mean(), pts[:, 1].mean(),
                     pts[:, 0].min(), pts[:, 0].max(),
                     pts[:, 1].min(), pts[:, 1].max())
    cube = cube_points(np.array([0.0, -MAT_Y, TABLE_H + MAT_T + cube_a / 2]), cube_a)
    pts = project(cube, cam, fovy)
    if pts is None:
        return None
    out["cube"] = (pts[:, 0].mean(), pts[:, 1].mean(),
                   pts[:, 0].min(), pts[:, 0].max(),
                   pts[:, 1].min(), pts[:, 1].max())
    return out


def residuals(params):
    pred = predict(params)
    if pred is None:
        return np.full(24, 1e3)
    res = []
    for name, meas in MEAS.items():
        if name not in pred:
            res += [1e3] * 6
            continue
        res += [p - q for p, q in zip(pred[name], meas)]
    return np.asarray(res)


# Initial guess: the old hand-set camera. It sat at x=-0.55, z=1.05 and looked
# 27 degrees below horizontal (63 degrees off vertical) toward +x.
guess = np.array([-0.55, 0.0, 1.05, 0.0, np.radians(27.0), 0.0,
                  np.radians(45.0), 0.085, 0.029])
lower = np.array([-1.2, -0.8, 0.30, -np.pi, np.radians(2), -0.6,
                  np.radians(15), 0.02, 0.010])
upper = np.array([1.2, 0.8, 2.20, np.pi, np.radians(88), 0.6,
                  np.radians(120), 0.20, 0.08])
scale = np.array([0.15, 0.15, 0.15, 0.2, np.radians(10), 0.1,
                  np.radians(12), 0.02, 0.006])
print("predict at guess", {k: np.round(v, 1) for k, v in predict(guess).items()})
print("cost at guess", round(float(residuals(guess) @ residuals(guess)), 1))
print("residuals  ", np.round(residuals(guess), 1))
best = None
for offset in (np.zeros(9), np.array([0.3, -0.3, 0.4, 0.3, 0.3, 0.2, 0.3, -0.4, 0.4]),
               np.array([-0.4, 0.2, -0.3, -0.3, -0.3, -0.2, -0.3, 0.5, -0.5])):
    start = np.clip(guess + offset * scale * 3, lower + 1e-6, upper - 1e-6)
    trial = nelder_mead(residuals, start, scale)
    trial.x = np.clip(trial.x, lower, upper)
    trial.cost = float(residuals(trial.x) @ residuals(trial.x))
    if best is None or trial.cost < best.cost:
        best = trial
result = best

p = result.x
names = ["cam_x", "cam_y", "cam_z", "yaw", "pitch", "roll", "fovy",
         "mat_radius", "cube_edge"]
print("fit cost", round(result.cost, 1))
for name, value in zip(names, p):
    degrees = name in {"yaw", "pitch", "roll", "fovy"}
    print(f"  {name:11s} {np.degrees(value):9.3f} deg" if degrees
          else f"  {name:11s} {value:9.4f} m")

pred = predict(p)
print("\n          measured                          predicted")
for name, meas in MEAS.items():
    print(f"{name:11s} {np.round(meas, 1)}")
    print(f"            {np.round(pred[name], 1)}  "
          f"err {np.round(np.asarray(pred[name]) - np.asarray(meas), 1)}")

d = np.asarray(p)
guess_vec = dict(zip(names, d))
R = rot(d[3], d[4], d[5])
forward = R[2]
up = R[1]
print("\nlook-at direction", np.round(forward, 4), " up", np.round(up, 4))
json.dump({"cam_pos": d[:3].tolist(), "cam_R": R.tolist(),
           "fovy_deg": float(np.degrees(d[6])),
           "mat_radius": float(d[7]), "cube_edge": float(d[8])},
          open("/tmp/middle_fit.json", "w"), indent=2)
print("wrote /tmp/middle_fit.json")
