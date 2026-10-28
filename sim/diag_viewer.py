"""Why does the viewer window close before the loop is done?

Opens the passive viewer on the bare scene, steps nothing and renders nothing,
and prints once a second whether the window is still there. If this one also
closes, the window is being shut from outside the process; if it survives, the
offscreen camera renders in viewer_run.py are what ends it.
"""
import time

import mujoco
import mujoco.viewer

model = mujoco.MjModel.from_xml_path("scene.xml")
data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.lookat[:] = (0.0, 0.0, 0.78)
    viewer.cam.distance = 1.9
    viewer.cam.azimuth = 145.0
    viewer.cam.elevation = -22.0
    started = time.monotonic()
    last = -1
    while time.monotonic() - started < 25:
        mujoco.mj_step(model, data)
        viewer.sync()
        second = int(time.monotonic() - started)
        if second != last:
            print(f"t={second:2d}s  window open: {viewer.is_running()}", flush=True)
            last = second
        if not viewer.is_running():
            break
        time.sleep(1 / 60)
    print(f"left the loop at {time.monotonic() - started:.1f} s, "
          f"window open: {viewer.is_running()}")
