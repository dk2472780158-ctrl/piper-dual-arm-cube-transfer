# Troubleshooting

Ordered roughly by how much time each one cost on the real machine.

The two commands that resolve most questions:

```bash
bash launch/cube_loop_rerun.sh --offline-check   # is the software stack healthy?
ls -l /dev/camera_* /dev/video*                  # are the cameras where you think?
```

---

## 1. `ModuleNotFoundError: No module named 'matplotlib'` on a machine where it is obviously installed

**Symptom.** The environment is definitely installed and definitely has matplotlib, yet launching a script dies immediately with the import error.

**What it means.** The same conda environment is reachable at **two different paths**, and you picked the wrong one. This is normal in a container that mounts a host directory: the environment directory may be visible as `/opt/.../envs/lerobot_v30` from inside the container and as `/home/.../miniconda3/envs/lerobot_v30` from the host, or vice versa. One of those paths resolves through a network-mounted or partially-populated view of the tree, and the packages that were installed into the other copy simply are not there.

It is not a small difference. A half-populated environment can import successfully and then fail later in an entirely unrelated-looking place, which is what makes this expensive to diagnose.

**Fix.** Determine which side you are on *first*, then set `LEROBOT_ENV` to the prefix that actually exists where you are:

```bash
whoami                        # distinguishes host from container
echo "$LEROBOT_ENV"
"$LEROBOT_ENV/bin/python" -c "import matplotlib; print(matplotlib.__version__)"
```

If that last line fails, you are pointed at the wrong copy. `config/env.sh` exists precisely so this is a one-line change rather than an edit to every script.

The general lesson: **a cluster-wide hostname in a shell prompt is not evidence of where you are.** In a container with a fixed `PS1`, the prompt keeps saying the host's name. Check the user, check the path, check the interpreter.

---

## 2. `conda activate` makes things worse

**Symptom.** You activate the environment "properly" and the scripts now fail, or silently pick up different packages.

**What it means.** An interactive login shell may already have put the correct environment's `bin` at the front of `PATH`. Activating the environment *by name* can resolve to a different, same-named environment — the other copy from problem 1 — and quietly replace it.

**Fix.** Don't activate. Every launcher here does:

```bash
ENV_ROOT="${LEROBOT_ENV}"
export PATH="$ENV_ROOT/bin:$PATH"
```

and then invokes `"$ENV_ROOT/bin/python"` by absolute path. It works identically from a login shell, a systemd unit and a container entrypoint, and it cannot be redirected by an activation script. If you need to switch environments, change the prefix, don't activate.

---

## 3. `RuntimeError: No CUDA GPUs are available` — but `nvidia-smi` worked an hour ago

**Symptom.** A run that worked earlier now cannot see the GPU. The host is fine.

**What it means.** GPU passthrough to a long-running container can drop out. The container's NVML stops reporting devices while the host's still does. Nothing in this repository causes it, and nothing in it can fix it — it is a container runtime problem.

**Fix.** Confirm the diagnosis by comparing the host and the container:

```bash
nvidia-smi                          # on the host
nvidia-smi                          # inside the container — expect this one to fail
```

Then repair the container's device access. On this project the platform provides a repair script for exactly this; if yours does not, restarting the container runtime restores the devices. Do not try to work around it from inside the Python environment.

---

## 4. The camera view is black, or the client reports `refused` / `EOF`

**Symptom.** A remote camera panel shows nothing. The task itself may be running fine.

**What it means.** On an appliance-style deployment the cameras are held by an **idle preview publisher** when no task is running, and by a **fan-out publisher** during a task. If neither is up, there is nothing on the port and the client's connection is refused or closed immediately.

**Fix.** Check whether the idle publisher is running and start it:

```bash
pgrep -af fanout
echo "$PIPER_IDLE_FANOUT_SH"          # set in config/env.sh; empty means you have none
bash "$PIPER_IDLE_FANOUT_SH" start
```

If your deployment has no idle-preview helper at all, set `PIPER_IDLE_FANOUT_SH=` (empty) so the launchers stop looking for one, and make sure the fan-out inside `ops/fanout/` is what your viewer is connecting to.

Remember the underlying constraint: **one owner per V4L2 device.** A preview holding the cameras is also why a task can fail to start, and the handover logic in `ops/fanout/scripts/piper_lerobot_fanout_env.sh` exists to sequence the two.

---

## 5. `could not open camera` / `VIDIOC_STREAMON` fails only sometimes

**Symptom.** Runs fine, then after a reboot the arm moves to the wrong place, or a camera fails to open.

**What it means.** Two possibilities, and they look similar:

- `/dev/videoN` numbers changed because enumeration order changed. Covered by [HARDWARE.md](HARDWARE.md#why-symlinks-and-not-devvideoN) — this is why the code uses `/dev/camera_*` symlinks.
- A previous process still holds the device. V4L2 is exclusive; a crashed run that did not clean up leaves the camera busy.

**Fix.**

```bash
ls -l /dev/camera_* /dev/video*     # do the symlinks point where you expect?
sudo fuser -v /dev/video*           # who is holding them?
pgrep -af 'python|view_'            # any stray preview or recording process?
```

If a stale process owns the camera, stop it. If the symlinks are wrong, revisit the udev rules — and note that a camera silently taking another camera's name is worse than one that fails to open, because nothing errors and the policy is simply fed the wrong view.

---

## 6. `AttributeError: module 'rerun' has no attribute 'Scalars'`

**Symptom.** The run dies at start-up, the moment `--display_data=true` is processed.

**What it means.** The Rerun SDK is older than 0.31. Older SDKs have no `rr.Scalars` and no `rr.set_sinks`, and `visualization_utils.py` needs both.

**Fix.** Either upgrade the environment's SDK, or point `RERUN_SDK_ROOT` at a newer one without touching the environment:

```bash
RERUN_SDK_ROOT=/opt/rerun-sdk/0.31.3 bash launch/cube_loop_rerun_fast.sh
```

`launch/rerun_env.sh` prepends it to `PYTHONPATH` and warns if the directory does not exist. Check what you actually have with:

```bash
"$LEROBOT_ENV/bin/python" -c "import rerun; print(rerun.__version__, rerun.__file__)"
```

---

## 7. The viewer is connected, the run is going, and nothing appears

**Symptom.** No error anywhere. The viewer simply stays empty.

**What it means.** `rr.set_sinks()` **replaces** the sink list rather than adding to it. If any code calls `rr.save(...)` or `rr.set_sinks(...)` after the gRPC connection is established, the live sink is dropped and everything goes to the file instead — silently.

**Fix.** Set both sinks *at once*, and don't call either afterwards:

```python
rr.set_sinks(rr.FileSink(path), rr.GrpcSink(f"rerun+http://{host}:{port}/proxy"))
```

That is what `launch/fast/cube_rerun_archive.py` does. If you are debugging a viewer that shows nothing, check whether a `.rrd` is growing on disk — data arriving in the file and not the viewer is the signature of this bug.

---

## 8. `RuntimeError: Another cube runner is active`

**Symptom.** The loop refuses to start.

**What it means.** Another process holds `loop_runner/.run.lock`. This is the lock doing its job: two runners sharing a CAN bus and three cameras produce garbled frames and unpredictable motion, and the failure looks like a policy problem rather than a configuration one.

**Fix.** Find and stop the other runner rather than deleting the lock:

```bash
fuser -v loop_runner/.run.lock
pgrep -af resident
```

The lock is released automatically when the holding process exits, so a stale lock means a stale process — not a stale file.

---

## 9. `RuntimeError: Feature order mismatch: observation.state`

**Symptom.** The runner refuses to start with a feature-order complaint.

**What it means.** The order of names in `observation.state` or `action`, as declared by `loop_runner/resident_new_env.py:features()`, does not match what the robot driver reports. Normalising against the wrong layout silently corrupts the observation — so the runner checks before connecting rather than after.

**Fix.** This is a real mismatch, not a false alarm. Either the robot driver's feature order changed, or the policy you are loading was trained against a different layout. Compare against the `observation.state` indices documented in [ARCHITECTURE.md](ARCHITECTURE.md#observationstate--28-dimensions) — the interleaving of `pos` and `effort` is the usual culprit.

Do **not** relax the assertion to get a run started.

---

## 10. Reset ends with a gripper tracking error

**Symptom.** A stage completes, the reset begins, and the run halts with a gripper tracking error.

**What it means.** The reset opens the gripper while the arm is still low, and the gripper's tracking check (`--reset_gripper_tracking_limit=0.01`) is tight enough to trip when the jaws are obstructed or already near their limit. This is a known defect of the reset sequence, not a fault in your hardware.

**Fix.** Keep the start pose clear — nothing under or between the jaws. If it recurs at the same stage, a reset from a slightly different pose usually clears it:

```bash
bash ops/reset_piper_pose.sh
```

Raising `--reset_gripper_tracking_limit` would also stop the trip, and is exactly the kind of safety-parameter change you should **not** make to get a stage to pass.

---

## 11. The two arms mirror each other's motion

**Symptom.** Commanding one arm produces movement in both, mirrored. It looks like a policy failure.

**What it means.** The CAN ports are swapped. `can0` is the **right** arm and `can1` is the **left** arm, and every entry point passes them explicitly.

**Fix.**

```bash
ip -details link show can0
ip -details link show can1
```

Correct the port arguments, or the wiring. Do not "fix" it by retraining.

---

## 12. Runs are jittery, or the arm freezes for a fraction of a second

**Symptom.** Runs with Rerun instrumented are noticeably worse than runs without it. Occasionally a single step takes a second.

**What it means.** `rr.log()` blocks the control thread when the SDK's send channel fills, and the call site is inside the control period. This is the single most interesting problem in this repository and it has its own document.

**Fix.** See **[RERUN_PERFORMANCE.md](RERUN_PERFORMANCE.md)**. The short version:

```bash
CUBE_RERUN_IMG_HZ=0 bash launch/cube_loop_rerun_fast.sh   # keep curves, drop images
CUBE_RERUN_FAST=off bash launch/cube_loop_rerun_fast.sh   # disable the overlay entirely
```

The second is the full rollback: behaviour becomes identical to the un-overlaid launcher.

---

## 13. `RuntimeError: Interactive terminal required`

**Symptom.** The loop refuses to start when launched from a script, a CI job, or with stdin redirected.

**What it means.** The runner reads stdin on a background thread so that typing `stop` halts the loop cleanly. With no TTY it cannot do that, and it says so rather than starting a motion sequence with no way to stop it.

**Fix.** Run it from an interactive terminal. If you genuinely need it non-interactive, `--plan` and `--offline-check` do not require a TTY — but neither of them moves the arms.

---

## 14. `ERROR: checkpoint not found`

**Symptom.** A launcher exits before doing anything, printing a path.

**What it means.** The checkpoint path does not exist. This is a good failure: the launcher checks for `config.json` and `model.safetensors` before touching hardware.

**Fix.** The checkpoint is referenced in **two** places, and they must agree:

1. the `CHECKPOINT` variable at the top of each `launch/*.sh`, and
2. the `CHECKPOINTS` dict in `loop_runner/resident_new_env.py` (and in `launch/resident_new_env_rerun.py`).

The path is relative to `LEROBOT_REPO`, i.e. `$LEROBOT_REPO/outputs/train/<run>/checkpoints/<step>/pretrained_model`. If you trained your own policy, update both.
