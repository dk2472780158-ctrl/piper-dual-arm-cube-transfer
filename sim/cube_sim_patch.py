"""Swap the hardware Piper for the MuJoCo one, without editing any upstream file.

`sitecustomize.py` installs an import hook that calls apply() right after
`lerobot.robots.piper_dual.piper_dual` or `lerobot.robots.utils` finishes
executing, which is before any caller binds the names.
"""
from __future__ import annotations


def apply(fullname: str, module) -> None:
    if fullname == "lerobot.robots.piper_dual.piper_dual":
        from piper_dual_sim import PiperDualSim

        module.PIPERDual = PiperDualSim
        print("[cube_sim] PIPERDual -> PiperDualSim (MuJoCo)", flush=True)
        return

    if fullname == "lerobot.robots.utils":
        original = getattr(module, "make_robot_from_config", None)
        if original is None:
            return
        from piper_dual_sim import PiperDualSim

        def make_robot_from_config(config):
            if getattr(config, "type", None) in ("piper_dual", "piper_dual_sim"):
                return PiperDualSim(config)
            return original(config)

        module.make_robot_from_config = make_robot_from_config
        print("[cube_sim] make_robot_from_config -> PiperDualSim", flush=True)
