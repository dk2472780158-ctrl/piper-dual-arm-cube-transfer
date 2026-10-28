#!/usr/bin/env python3
"""Check the sim robot contract without involving lerobot or the policy."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from piper_dual_sim import MOTORS, SIDES, PiperDualSim  # noqa: E402


def main() -> int:
    sim = PiperDualSim(None)
    sim.connect()
    print("camera_features:", sorted(sim.camera_features))
    print("action_features:", len(sim.action_features))
    print("observation_features:", len(sim.observation_features))

    obs = sim.get_observation()
    missing = [f"{side}_{motor}.{k}" for side in SIDES
               for motor in MOTORS for k in ("pos", "effort")]
    missing = [key for key in missing if key not in obs]
    print("missing state keys:", missing)
    for name in ("left", "middle", "right"):
        frame = obs[name]
        print(f"  {name} frame {frame.shape} {frame.dtype} mean {frame.mean():.1f}")

    state = np.array([obs[key] for key in sim.observation_features
                      if key.endswith(('.pos', '.effort'))])
    print("state vector", state.shape, "finite", bool(np.isfinite(state).all()))

    before = sim.left_bus.read()
    target = {}
    for side in SIDES:
        for motor in MOTORS:
            target[f"{side}_{motor}.pos"] = 0.0
    for index in range(1, 7):
        target[f"left_joint_{index}.pos"] = float(before[f"joint_{index}_pos"]) + 0.05
    sim.send_action(target)
    for _ in range(20):
        sim.get_observation()
    after = sim.left_bus.read()
    delta = after["joint_1_pos"] - before["joint_1_pos"]
    print(f"joint_1 moved {delta:.4f} rad toward +0.05")

    gripper = dict(target)
    gripper["left_gripper.pos"] = 0.06
    sim.send_action(gripper)
    for _ in range(20):
        sim.get_observation()
    print("gripper opening", round(sim.left_bus.read()["gripper_pos"], 4), "m")

    obs = sim.get_observation()
    from PIL import Image
    out = HERE.parent / "out"
    out.mkdir(parents=True, exist_ok=True)
    for name in ("left", "middle", "right"):
        Image.fromarray(obs[name]).save(out / f"smoke_{name}.png")
    sim.disconnect()
    print("SMOKE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
