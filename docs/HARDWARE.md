# Hardware

<img src="../assets/hardware_setup.jpg" width="620" alt="Two PiPER arms below a desk-height work surface; a yellow pad with a pink cube on the left, a purple pad on the right">

*The cell. Start pad (yellow) with the cube, target pad (purple), both arms mounted below the edge of the surface. The right arm's wrist camera is visible in the foreground.*

---

## Bill of materials

| Item | Notes |
|---|---|
| 2 × AgileX **PiPER** 6-DoF arm | One leads each direction. |
| 2 × PiPER parallel gripper | Part of the arm's end effector. |
| 2 × USB-CAN adapter | One per arm — do **not** share one adapter across both. |
| 3 × USB RGB camera | 640×480 @ 30 fps, UVC. |
| 1 × x86 host with an NVIDIA GPU | ACT inference runs on `cuda`. |
| 1 × rigid mount | Both arm bases must be fixed relative to each other and to the surface. |
| 2 × foam pads + 1 cube | Yellow = start, purple = goal. The cube is ~4 cm. |

Optional, for recording demonstrations by hand: a second pair of PiPER arms as leader arms, driven through `piper_dual_leader`.

---

## CAN bus

| Interface | Arm |
|---|---|
| `can0` | **right** |
| `can1` | **left** |

Both at 1 Mbit/s:

```bash
sudo ip link set can0 up type can bitrate 1000000
sudo ip link set can1 up type can bitrate 1000000

# confirm
ip -details link show can0
```

These port names are passed explicitly on every entry point (`--robot.left_port=can1 --robot.right_port=can0`). If you swap them, the arms will mirror each other's motion, which looks like a policy failure but is a wiring problem.

One consequence worth internalising: **the CAN interfaces are single-owner resources.** Two processes bound to the same interface produce garbled frames and unpredictable motion, which is why the loop runner takes a lock file before it does anything else.

---

## Cameras

Three USB cameras, all at 640×480 @ 30 fps, named by their role in the task:

| Symlink | Role |
|---|---|
| `/dev/camera_left` | Looks along the left arm's approach |
| `/dev/camera_middle` | Overhead / centre view of both pads |
| `/dev/camera_right` | Looks along the right arm's approach |

### Why symlinks and not `/dev/videoN`

`/dev/videoN` numbers come from USB enumeration order. Reboot, or plug anything else in, and "left" and "middle" can swap. Because ACT consumes the raw image, a swap does not raise an error — it silently feeds the policy a view it was never trained on, and the arm goes somewhere wrong.

Everything in this repository therefore resolves the symlink at start-up and converts it to the numeric index OpenCV wants:

```
/dev/camera_left  →  /dev/video12  →  index 12
```

`loop_runner/resident_new_env.py:camera_index()` does this by resolving the symlink, and falls back to scanning `/dev/video*` for a node with the same device number when the symlink does not point at a plain `videoN`.

Use [`ops/fanout/scripts/99-piper-cameras.rules`](../ops/fanout/scripts/99-piper-cameras.rules) as a starting point for the udev rules. Find each camera's serial with:

```bash
udevadm info --attribute-walk --name=/dev/video0 | grep -E 'serial|KERNELS'
```

### One owner at a time

A V4L2 device cannot be opened by two processes. On an appliance-style deployment there is usually an *idle preview* holding the cameras for a remote viewer, so a task has to take them back before it starts and hand them over when it exits. That handover is what `ops/fanout/scripts/piper_lerobot_fanout_env.sh` does: it stops the idle publisher before the run and restarts it via an `EXIT` trap. If you run without an idle preview, set `PIPER_IDLE_FANOUT_SH` to empty and the handover is skipped.

Symptoms of getting this wrong are always the same: `refused`, `EOF`, or a `VIDIOC_STREAMON` error from OpenCV at the moment a run starts.

---

## Mounting and geometry

The policies are position-based and the arms are not told where anything is; they learn the trajectory from wherever the base sits. Practical consequences:

- **Fix the bases.** Any shift between recording and inference changes every trajectory. If you move an arm, re-record.
- **Keep the pads in the same place.** The pads are the only fixed reference the cameras have. Tape them down.
- **Check for cable drag.** The arms' own cabling and any camera cable in the workspace will appear in the observations and will change how the arm moves.
- **Mind the gripper.** The reset sequence opens the gripper before the arm is fully raised, and the gripper's tracking check (`--reset_gripper_tracking_limit=0.01`) is tight enough to trip a safety stop if the jaws are obstructed. Keep the start pose clear.

---

## Camera placement pre-flight

Before a recording session, confirm each camera's view actually contains what you think. There are two preview scripts:

```bash
python loop_runner/view_three_cameras.py    # ⚠️ holds all three cameras
python loop_runner/view_right_camera.py     # ⚠️ holds one
```

They hold the devices while running, so nothing else can use the cameras at the same time. Close them before starting a task.

What to check in each view:

1. Both pads are visible and are *not* clipped by the image border.
2. The cube at the start position is clearly distinguishable.
3. The arm does not occlude the pad it is not using.
4. Lighting is the same as it was during recording. A lamp moved between sessions is enough to shift the trajectories.

---

## Interface summary

```
host
 ├── can0 ──────────► right PiPER arm
 ├── can1 ──────────► left  PiPER arm
 ├── /dev/camera_left   ─┐
 ├── /dev/camera_middle ─┼─► 640×480 @ 30 fps ─► policy observation
 └── /dev/camera_right  ─┘
```
