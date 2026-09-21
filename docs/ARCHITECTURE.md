# Architecture

## The control tick

Everything is built around one loop that must sustain **30 Hz** — a 33.3 ms budget per step. Each tick, in order:

1. **Read the observation** — three camera frames at 640×480 plus the 28-dimensional joint state (position and effort for every joint on both arms), pulled over CAN.
2. **Preprocess** — normalise the observation with the statistics stored alongside the checkpoint.
3. **Infer** — the ACT policy predicts a 100-step action chunk.
4. **Ensemble** — average the overlapping predictions into the single action that will actually be executed (see below).
5. **Send the action** to the arms over CAN.
6. **Write the frame** to the dataset, if recording.
7. **Log to Rerun** — 42 scalar curves plus the camera frames.
8. **Sleep** for whatever remains of the 33.3 ms.

Steps 5 and 6 are the ones that must not be delayed. Step 7 is the one that historically was, and it sits *between* the action being sent and the sleep — so any time it loses comes straight out of the control period. That ordering is why the Rerun work in this repository was necessary at all; see [RERUN_PERFORMANCE.md](RERUN_PERFORMANCE.md).

A subtlety worth knowing if you add anything to this path: the tick is not budgeted. A step that overruns does not get a shorter sleep to compensate — it just runs late. Sustained overrun shows up as a lower effective control rate, and a single long overrun shows up as a visible jerk.

---

## Observation and action spaces

Defined once, in `loop_runner/resident_new_env.py:features()`, and asserted against the robot's own feature builders at start-up. If the real robot's feature order differs from this table, the runner raises before connecting.

### `action` — 14 dimensions

Fourteen motors, in this order:

| Index | Motor |
|---|---|
| 0–5 | `left_joint_1` … `left_joint_6` |
| 6 | `left_gripper` |
| 7–12 | `right_joint_1` … `right_joint_6` |
| 13 | `right_gripper` |

### `observation.state` — 28 dimensions

The same 14 motors, each contributing **two** values — commanded position and measured effort — interleaved:

```
index 0  left_joint_1.pos      index 1  left_joint_1.effort
index 2  left_joint_2.pos      index 3  left_joint_2.effort
...
index 12 left_gripper.pos      index 13 left_gripper.effort
index 14 right_joint_1.pos     index 15 right_joint_1.effort
...
index 26 right_gripper.pos     index 27 right_gripper.effort
```

**Even indices are position, odd indices are effort.** This layout is load-bearing for the normalisation fix below, and getting it wrong silently corrupts training rather than failing.

### `observation.images.*`

`left`, `middle`, `right` — each `480×640×3`, stored as video.

---

## Two policies, one per direction

The system does not have a "move the cube" policy. It has:

| Policy | Activated for | Idle arm |
|---|---|---|
| `l2r` | left arm carries the cube left→right | right |
| `r2l` | right arm carries the cube right→left | left |

Why split it:

- A single policy would have to infer the intended direction from the scene, which is exactly the sort of decision an operator can make for free by launching the right script.
- Each directional policy gets the whole trajectory budget for one motion, instead of splitting capacity across two.
- The two directions can be trained from different amounts of data, and improved independently. That is what happened here: the right-arm policy went through two more fine-tuning generations than the left.

The cost is that the idle arm is nearly constant, which creates a normalisation problem — next section.

---

## Normalisation flooring for the idle arm

ACT normalises `observation.state` by the mean and standard deviation stored in the dataset's `meta/stats.json`. Those statistics come from the demonstrations, and in those demonstrations the idle arm barely moves. Its standard deviation therefore collapses towards zero.

Dividing by a near-zero σ is a large amplification. The idle arm's dimensions end up dominated by sensor noise and quantisation, and the policy becomes hypersensitive to a channel that carries no information about the task at all.

The training scripts therefore rewrite `meta/stats.json` before training, flooring only the **idle** arm's standard deviations:

| Quantity | Floor |
|---|---|
| Idle arm position σ | `0.05` |
| Idle arm effort σ | `0.10` |

Two details that make this safe rather than reckless:

- The original file is copied to `meta/stats.before_robust_normalization.json` first.
- The script prints every dimension it changed, before and after, so the edit is auditable rather than silent.

Note the asymmetry between the two policies: for left→right it is the **right** arm that is idle, so the right-arm indices get floored. For right→left it is the left arm. Each script floors the arm that is *not* doing the work.

---

## The data-direction gate

Before training the left-arm policy, `launch_original/train_cube_l2r_50k.sh` opens the merged dataset's parquet files and measures, per action dimension, the peak-to-peak range of the recorded actions. It then requires:

```
max(range over left-arm dims)  >  max(range over right-arm dims)
```

If that fails, it prints `WARNING: 数据中右臂动作范围不小于左臂` and the script comments make the intent explicit: *confirm this session really is a left-arm session before training*.

This exists because a recording session in this project was mislabelled — the directories said `l2r` while the data was actually right-arm motion. A policy trained on that data learns to move the wrong arm, and nothing about the training run looks abnormal: the loss curve is fine, the checkpoints save, the inference runs. The only cheap place to catch it is before training starts.

The gate is a heuristic, not a proof. It is cheap and it has earned its place.

---

## Temporal ensembling

Training and inference use different action-chunk settings, on purpose:

| | `chunk_size` | `n_action_steps` |
|---|---|---|
| Training | 100 | 50 |
| Inference (single shot) | 100 | **1** + `temporal_ensemble_coeff=0.01` |

At every timestep the policy re-predicts a full 100-step chunk, and the action that actually goes to the motors is a weighted average over every prediction that covers the current step:

```
w_i = exp(-coeff × i)
```

where `i` counts how many steps ago prediction `i` was made. Recent predictions dominate; older ones fade smoothly. With `coeff = 0.01` the decay is slow, so the average spans a wide window and the resulting trajectory is smooth.

This is why the extra action filter is switched off. `PIPER_ACTION_FILTER_ALPHA=1` disables LeRobot's additional smoothing — the ensemble has already done the job, and stacking a second filter on top would add lag without adding smoothness.

The ensemble is also the reason the loop runner's stage-level motion looks continuous while inference itself is chunked.

---

## The resident loop

Reconnecting CAN, loading two policies and rebuilding the processors costs seconds and adds failure modes. So the loop runner does all of it **once**:

```
connect robot + load both policies + build pipelines
  → STAGE 1/20 l2r   → 15 s rollout → reset(left)
  → STAGE 2/20 r2l   → 15 s rollout → reset(right)
  → … repeated for --cycles cycles (max 10, so 20 stages)
  → RESIDENT_COMPLETE
```

Per stage:

- `STAGE <n>/20 <direction>` is printed before the rollout.
- The rollout runs `record_loop` for 15 s at 30 fps, with `display_data=False` — the loop's own telemetry is the Rerun path in `launch/`, not this one.
- `reset(arm)` moves **the arm that just did the work** back to the training start pose: joint speed 0.8, gripper speed 0.01, settle timeout 8 s, settle tolerance 0.02, gripper tracking limit 0.01.
- `RESET_DONE <arm> seconds=<t>` reports how long it took. The next policy is already loaded, so nothing is reloaded between stages.

Each stage's numeric trace is written to `<run_dir>/<NN>_<direction>.jsonl` (`time`, `state`, `action` per row), and `run_config.json` records the checkpoints, direction plan, camera indices and ensemble coefficient for the run. `tools/log2png.py` renders a `.jsonl` into a plot.

### Stopping

`Ctrl+C`, `SIGTERM`, or typing `stop` at the prompt all land on the same handler, and the behaviour is deliberate:

> No further stage is started, and **no extra return-home motion is performed.**

So the arms may be left energised and mid-pose. If you need them parked, run a reset explicitly afterwards.

### The lock

`main()` takes an exclusive `flock` on `loop_runner/.run.lock` before doing anything. Its stated purpose, in the source comment, is to prevent an old and a new entry point from being live at the same time and fighting over the CAN bus and cameras.

The Rerun-wired variant lives in `launch/resident_new_env_rerun.py`, a different directory. Its lock is deliberately pointed back at the *original* directory's lock file, because a lock in its own directory would let the new entry point and the old one coexist — which is exactly the case the lock exists to prevent.

---

## Cameras: the handover

Three V4L2 devices, one possible owner at a time. Two things want them:

- an **idle preview** that a remote operator watches when no task is running, and
- **the task process**, which needs them for observations.

`ops/fanout/scripts/piper_lerobot_fanout_env.sh` is sourced by the launchers and does the handover: stop the idle publisher, wait ~0.5 s for V4L2 to actually release the devices (OpenCV fails immediately if it opens too soon), install an `EXIT` trap to restart the preview, then export the fan-out paths and publishing flags.

The offline modes skip all of it. Both dry-run entry points set `OFFLINE=1` when they see `--plan` or `--offline-check`, and the handover is only sourced when `OFFLINE=0`. **The self-checks therefore touch no camera service and no hardware** — that is a property worth preserving if you modify the launchers.

---

## Where the Rerun hook sits

`lerobot_snapshot/utils/visualization_utils.py` owns `log_rerun_data()`. `piper_cube_record.py` imports it **by name**:

```python
from ...visualization_utils import init_rerun, log_rerun_data
```

That is why the fast overlay has to install itself at interpreter start-up, before `piper_cube_record` is imported — see [RERUN_PERFORMANCE.md](RERUN_PERFORMANCE.md) for the whole mechanism.
