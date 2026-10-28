"""Run the cube-transfer scripts against MuJoCo instead of the real arms.

Put this directory on PYTHONPATH and set CUBE_SIM=1. Python imports this module
at startup, so the swap happens before the entry point touches lerobot. With
CUBE_SIM unset the file does nothing and the real hardware path is untouched.
"""
from __future__ import annotations

import os
import sys
from importlib.machinery import PathFinder

_TARGETS = {
    "lerobot.robots.piper_dual.piper_dual",
    "lerobot.robots.utils",
}


class _Loader:
    def __init__(self, inner):
        self._inner = inner

    def create_module(self, spec):
        return self._inner.create_module(spec)

    def exec_module(self, module):
        self._inner.exec_module(module)
        try:
            import cube_sim_patch
            cube_sim_patch.apply(module.__name__, module)
        except Exception as exc:  # never take the process down over a patch
            print(f"[cube_sim] patch failed for {module.__name__}: {exc}",
                  file=sys.stderr)


class _Finder:
    def find_spec(self, fullname, path=None, target=None):
        if fullname not in _TARGETS:
            return None
        spec = PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None or isinstance(spec.loader, _Loader):
            return spec
        spec.loader = _Loader(spec.loader)
        return spec


if os.environ.get("CUBE_SIM") == "1":
    sys.meta_path.insert(0, _Finder())
