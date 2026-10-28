#!/usr/bin/env python3
"""Run the production resident loop unchanged, minus the USB camera lookup.

resident_new_env.py resolves /dev/camera_* before it builds the robot. Simulation
has no such devices, so this loader executes the same file and replaces only that
lookup; every other line, including the lock, the reset sequence and the two ACT
policies, is the production code path.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

TARGET = Path("/home/databall/VLA/lerobot_piper/cube_transfer_best_v1/resident_new_env.py")
SIM_CAMERA_INDEX = {"left": 0, "middle": 1, "right": 2}


def main() -> int:
    if not TARGET.is_file():
        print(f"missing production runner: {TARGET}", file=sys.stderr)
        return 2
    spec = importlib.util.spec_from_file_location("resident_new_env_cube_sim", TARGET)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    def camera_index(name: str) -> int:
        return SIM_CAMERA_INDEX[name]

    module.camera_index = camera_index
    return module.main()


if __name__ == "__main__":
    raise SystemExit(main())
