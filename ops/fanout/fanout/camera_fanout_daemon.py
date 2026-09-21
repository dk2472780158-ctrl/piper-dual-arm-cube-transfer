#!/usr/bin/env python3
"""Idle daemon: open camera_* RGB + TCP fan-out until SIGTERM (for go2rtc 24/7 preview)."""
from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from pathlib import Path

import cv2
import numpy as np

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from fanout.bgr_tcp_mpegts import make_publisher, publish_bgr_frame

CAMERAS = {
    "left": "/dev/camera_left",
    "middle": "/dev/camera_middle",
    "right": "/dev/camera_right",
}

_stop = False


def _handle_sig(signum, _frame) -> None:
    global _stop
    print(f"[signal] {signum} — shutting down", flush=True)
    _stop = True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=int(os.environ.get("PIPER_RGB_PUBLISH_FPS", "15")))
    args = parser.parse_args()

    signal.signal(signal.SIGTERM, _handle_sig)
    signal.signal(signal.SIGINT, _handle_sig)

    names = os.environ.get("PIPER_RGB_PUBLISH_CAMERAS", "all").strip()
    if names.lower() in ("all", "*"):
        selected = list(CAMERAS.keys())
    else:
        selected = [x.strip() for x in names.split(",") if x.strip()]

    pubs: dict = {}
    caps: dict = {}
    interval = 1.0 / max(args.fps, 1)

    for name in selected:
        path = CAMERAS.get(name)
        if not path:
            print(f"[WARN] unknown camera {name}", file=sys.stderr)
            continue
        pub = make_publisher(name, width=args.width, height=args.height, fps=args.fps)
        pub.start()
        pubs[name] = pub
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            print(f"[FAIL] cannot open {path} ({name})", file=sys.stderr)
            return 1
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        caps[name] = cap
        print(f"[OK] daemon open {name} {path} → {pub.url}", flush=True)

    if not caps:
        print("[FAIL] no cameras opened", file=sys.stderr)
        return 1

    print(f"[RUN] idle fan-out @ {args.fps}fps — stop before lerobot-record", flush=True)
    heartbeat = time.time()

    try:
        while not _stop:
            loop_start = time.time()
            for name, cap in caps.items():
                ok, frame = cap.read()
                if not ok or frame is None:
                    continue
                if frame.shape[1] != args.width or frame.shape[0] != args.height:
                    frame = cv2.resize(frame, (args.width, args.height))
                publish_bgr_frame(pubs[name], np.ascontiguousarray(frame))
            elapsed = time.time() - loop_start
            if elapsed < interval:
                time.sleep(interval - elapsed)
            if time.time() - heartbeat >= 60:
                heartbeat = time.time()
                parts = []
                for n, p in pubs.items():
                    s = p.stats()
                    parts.append(f"{n}:sent={s.get('frames_sent', 0)} drop={s.get('frames_drop', 0)}")
                print("[heartbeat]", " ".join(parts), flush=True)
    finally:
        for cap in caps.values():
            cap.release()
        for pub in pubs.values():
            pub.close()
        print("[DONE] daemon stopped", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
