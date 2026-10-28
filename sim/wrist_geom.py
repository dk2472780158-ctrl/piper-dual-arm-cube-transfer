"""Print the right-wrist camera geometry at the training start pose.

Answers a single question: how far is the camera from the cube and from the
fingertips, and what size does the cube project to at the nominal field of
view? Comparing that with the recorded frame shows whether the mount or the
field of view is off.
"""
import numpy as np
import mujoco

SCENE = "/home/databall/VLA/cube_mujoco/sim/scene.xml"
SIGN = {1: 1.0, 2: -1.0, 3: -1.0, 4: -1.0, 5: -1.0, 6: 1.0}
START = np.asarray([0.0093, 0.0049, -0.0024, -0.0582, 0.3185, 0.0973, 0.0003,
                    0.0631, 0.0048, -0.0027, -0.0766, 0.3512, 0.0202, 0.0])
W, H = 640, 480
np.set_printoptions(precision=4, suppress=True)

m = mujoco.MjModel.from_xml_path(SCENE)
d = mujoco.MjData(m)
for side, base in (("left", 0), ("right", 7)):
    for i in range(1, 7):
        d.qpos[int(np.ravel(m.joint(f"{side}_joint{i}").qposadr)[0])] = (
            SIGN[i] * START[base + i - 1])
for i in (7, 8):
    for side in ("left", "right"):
        d.qpos[int(np.ravel(m.joint(f"{side}_joint{i}").qposadr)[0])] = 0.0
mujoco.mj_forward(m, d)

for side in ("right", "left"):
    body = m.body(f"{side}_gripper_base").id
    o = d.xpos[body].copy()
    R = d.xmat[body].reshape(3, 3)      # rows: body x, y, z in world
    print(f"\n{side}_gripper_base world pos {np.round(o, 4)}")
    print("  x axis", np.round(R[0], 3), " y axis", np.round(R[1], 3),
          " z axis", np.round(R[2], 3))
    points = {}
    for name in (7, 8):
        points[f"finger{name}"] = d.xpos[m.body(f"{side}_link{name}").id].copy()
    points["cube"] = d.xpos[m.body("cube").id].copy()
    for name, y in (("yellow_mat", 0.111), ("purple_mat", -0.111)):
        points[name] = np.array([0.0, y, 0.712 + 0.024])
    for name, p in points.items():
        print(f"  {name:11s} world {np.round(p, 4)} "
              f"local {np.round(R @ (p - o), 4)}")

    cam = m.camera(side)
    cam_world = o + R.T @ cam.pos
    print(f"  camera '{side}' local pos {np.round(cam.pos, 4)} "
          f"world {np.round(cam_world, 4)}")
    for name in ("cube", "purple_mat", "finger7"):
        print(f"  camera->{name:10s} {np.linalg.norm(points[name] - cam_world):.4f} m")

    fovy = np.radians(m.cam_fovy[cam.id])
    scale = (H / 2) / np.tan(fovy / 2)
    cube = points["cube"]
    depth = np.linalg.norm(cube - cam_world)
    edge = 0.0308
    print(f"  fovy {np.degrees(fovy):.1f} deg, scale {scale:.1f} px, "
          f"cube at {depth:.3f} m -> {scale * edge / depth:.1f} px wide")
