"""Optional LeRobot camera fan-out hook (default off).

Set ENABLE_PIPER_RGB_PUBLISH=1 and add parent of `fanout/` to PYTHONPATH.
Camera names left/middle/right map to TCP publishers on 19012/19011/19013.
"""
from __future__ import annotations

import logging
import os
import threading
from typing import Optional

import numpy as np

from .bgr_tcp_mpegts import DEFAULT_PORTS, BgrTcpMpegtsPublisher, make_publisher

log = logging.getLogger(__name__)

_lock = threading.Lock()
_publishers: dict[str, BgrTcpMpegtsPublisher] = {}
_started = False


def _enabled() -> bool:
    return os.environ.get("ENABLE_PIPER_RGB_PUBLISH", "0").strip() in ("1", "true", "yes", "on")


def _publish_width() -> int:
    return int(os.environ.get("PIPER_RGB_PUBLISH_WIDTH", "640"))


def _publish_height() -> int:
    return int(os.environ.get("PIPER_RGB_PUBLISH_HEIGHT", "480"))


def _publish_fps() -> int:
    return int(os.environ.get("PIPER_RGB_PUBLISH_FPS", "15"))


def _cameras_filter() -> Optional[set[str]]:
    raw = os.environ.get("PIPER_RGB_PUBLISH_CAMERAS", "middle").strip()
    if not raw or raw.lower() in ("all", "*"):
        return None
    return {x.strip() for x in raw.split(",") if x.strip()}


def ensure_publishers() -> None:
    """Start publishers for configured camera names (idempotent)."""
    global _started
    if not _enabled():
        return
    with _lock:
        if _started:
            return
        allowed = _cameras_filter()
        names = list(DEFAULT_PORTS.keys()) if allowed is None else sorted(allowed)
        w, h, fps = _publish_width(), _publish_height(), _publish_fps()
        for name in names:
            if name not in DEFAULT_PORTS:
                log.warning("skip unknown camera name for fan-out: %s", name)
                continue
            pub = make_publisher(name, width=w, height=h, fps=fps)
            pub.start()
            _publishers[name] = pub
            log.info("piper rgb publish ready name=%s url=%s", name, pub.url)
        _started = True


def maybe_publish_camera_frame(name: str, frame: np.ndarray) -> None:
    """Publish one RGB frame from LeRobot obs loop; never raises."""
    if not _enabled():
        return
    try:
        ensure_publishers()
        pub = _publishers.get(name)
        if pub is None:
            return
        if frame is None or not isinstance(frame, np.ndarray):
            return
        # LeRobot cameras usually return RGB; fan-out expects BGR for OpenCV/ffmpeg rawvideo.
        if frame.ndim == 3 and frame.shape[2] == 3:
            bgr = frame[:, :, ::-1].copy()
        else:
            return
        pub.publish_bgr_frame(bgr)
    except Exception as e:
        log.debug("fan-out publish skipped name=%s err=%s", name, e)


def shutdown_publishers() -> None:
    """Close all publishers (e.g. on process exit)."""
    global _started
    with _lock:
        for pub in _publishers.values():
            try:
                pub.close()
            except Exception:
                pass
        _publishers.clear()
        _started = False
