"""P3 hook tests (no camera)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from fanout.lerobot_hook import maybe_publish_camera_frame, shutdown_publishers


def test_hook_default_off():
    os.environ.pop("ENABLE_PIPER_RGB_PUBLISH", None)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    maybe_publish_camera_frame("middle", frame)  # no raise
    shutdown_publishers()
    print("OK hook_default_off")


def test_hook_enabled_middle():
    os.environ["ENABLE_PIPER_RGB_PUBLISH"] = "1"
    os.environ["PIPER_RGB_PUBLISH_CAMERAS"] = "middle"
    os.environ["PIPER_RGB_PUBLISH_FPS"] = "10"
    port = 19221
    os.environ["PIPER_RGB_MIDDLE_PORT"] = str(port)
    # make_publisher uses DEFAULT_PORTS; override via env not supported — use default 19011
    # Instead verify ensure path with custom port via direct publisher in bgr tests.
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    maybe_publish_camera_frame("middle", frame)
    shutdown_publishers()
    os.environ.pop("ENABLE_PIPER_RGB_PUBLISH", None)
    print("OK hook_enabled_middle (no viewer drop)")


if __name__ == "__main__":
    test_hook_default_off()
    test_hook_enabled_middle()
    print("P3 hook PASSED 2/2")
