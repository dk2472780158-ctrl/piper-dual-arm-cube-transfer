# Simulated cell (MuJoCo)

A MuJoCo replica of the real two-arm cell, used to generate training episodes and to
measure what the policy does when the pictures come from the simulator instead of the
cameras. The task is the same one the real robot does: carry the cube from one pad to
the other.

## What it produces

`gen_sim_data.py` replays a recorded real demonstration through the physics engine and
records the renders as a lerobot v3.0 dataset with the same schema as the real one
(14-dim action, 28-dim state, three 480x640 cameras, 30 fps). Each episode randomises
the surfaces, the two lights, where the cube is put down, its mass and friction, and the
arms' start pose -- everything the policy must not depend on. Actions are recorded clean
and commanded noisy, so a state that drifted is labelled with the action that would have
prevented the drift.

A take runs twice: once with the cameras off to see whether it lands, only then once
with them on to record it. About **89 s per episode** including retries, so 80 episodes
is roughly two hours.

## Requirements

- MuJoCo with EGL (`MUJOCO_GL=egl`), `imageio`, `numpy`, `pandas`
- lerobot 0.4.4 for the dataset writer and the policy probes
- The PiPER arm STL meshes from AgileX's `piper_description` package, placed in
  `assets/meshes` next to this directory; `scene.xml` reaches them as `../assets/meshes`
- A recorded real dataset to replay: the scripts expect one at the path in `ROOT`
  inside `gen_sim_data.py` and the probe scripts

Paths are absolute and point at the lab machine layout (`/home/databall/...`). Change
`ROOT` in `gen_sim_data.py` and `CKPT` / `ROOT` in the probe scripts to run elsewhere.

## Layout

| File | Role |
|---|---|
| `piper_dual_sim.py` | The simulated arms. Same call surface as the real `piper_dual` robot, so the recording loop does not know which one it is driving. |
| `scene.xml` | The cell: table, both pads, cube, lights, cameras. |
| `scene_real.xml` | Same, with the recorded table texture wired in. |
| `build_scene.py` | Generates the XML from the fitted geometry. |
| `gen_sim_data.py` | Records training episodes. |
| `directions.py` | Per-direction spawn points, mirroring and the action stream. |
| `render_sensor.py` + `render_model.json` | The camera's colour response, fitted per camera against recorded frames and applied after the render. |
| `render_lens.py` | Lens model: barrel distortion, vignetting, blur, sensor noise. Enabled with `CUBE_SIM_LENS=1`, off by default. |
| `make_textures.py`, `textures/` | Table and cube textures taken from recorded frames. |
| `fit_middle.py`, `fit_wrist.py`, `fit_render.py`, `solve_cell.py`, `solve_wrist.py`, `tune_wrist.py`, `calibrate_*.py` | Camera pose, gain field and effort calibration fitted against recorded frames. |
| `train_sim_act.sh` | Train ACT on sim data. `--from-scratch` for a cold start, otherwise it fine-tunes from a real-data checkpoint. |
| `train_mix_act.sh` | Train on the merged real+sim dataset. |
| `policy_probe*.py` | Diagnostics: replay a demonstration through several picture variants and see how far the policy drifts. |
| `diag_*.py` | One-off measurements taken while fitting the scene (grasp aperture, contact, wrist camera, penetration, ...). Kept as the record of how each number was obtained. |
| `loop.py`, `viewer_run.py`, `replay_demo.py`, `movie.py` | Closed-loop rollout, interactive viewer, demo replay and video export. |
| `smoke_sim.py`, `eval_run.py`, `eval_check.py`, `render_check.py` | Short self-checks. |

## Running

```bash
# 80 episodes into the lerobot cache under local/cube_sim_v1
MUJOCO_GL=egl python gen_sim_data.py --episodes 80 --out cube_sim_v1

# the same with the lens model on
MUJOCO_GL=egl CUBE_SIM_LENS=1 python gen_sim_data.py --episodes 80 --out cube_sim_v2

# train
bash train_sim_act.sh --from-scratch      # 20k steps, cold start
bash train_sim_act.sh                     # 12k steps, fine-tune from a real checkpoint
```

## What it has shown so far

| Weights | Simulated closed loop (5 trials) | On the real robot |
|---|---|---|
| Trained on sim only, fine-tuned from real | 5/5 | not run |
| Trained on real + sim | 3/5 | 3/3 |
| Trained on sim only, cold start | -- | 0/1, arm froze |

Cold start from simulated pictures alone does not transfer: fed recorded frames the same
weights score 0.220 rad mean error against 0.010 rad on rendered ones, and in closed loop
the output collapses to the dataset's mean pose. Matching the colour statistics of the
recorded frames before inference changes nothing (0.220 -> 0.228 rad), so the gap is what
the pictures contain, not how they are tinted.

## Known gaps

- The scene was re-calibrated after the first dataset was rendered. Measured against a
  recorded frame, a render from the current scene differs by 6.29/255 on average; a frame
  from the first dataset differs by 14.63/255.
- The wrist camera is the least faithful of the three and has been the limiting factor in
  every closed-loop result so far.
- The three camera intrinsics are fitted, not read from the devices.
