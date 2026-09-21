#!/usr/bin/env python3
"""P5 smoke: exclusive-open camera_* RGB + TCP fan-out while go2rtc pulls."""
from __future__ import annotations

import argparse
import os
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=45.0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=int(os.environ.get("PIPER_RGB_PUBLISH_FPS", "15")))
    args = parser.parse_args()

    names = os.environ.get("PIPER_RGB_PUBLISH_CAMERAS", "all").strip()
    if names.lower() in ("all", "*"):
        selected = list(CAMERAS.keys())
    else:
        selected = [x.strip() for x in names.split(",") if x.strip()]

    pubs = {}
    caps = {}
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
        print(f"[OK] open {name} {path} publish {pub.url}")

    if not caps:
        print("[FAIL] no cameras opened", file=sys.stderr)
        return 1

    t_end = time.time() + args.duration
    sent = {n: 0 for n in caps}
    drop = {n: 0 for n in caps}
    print(f"[RUN] publishing {args.duration}s @ {args.fps}fps — open /videos/ PiPER tab or curl frame.jpeg")

    try:
        while time.time() < t_end:
            loop_start = time.time()
            for name, cap in caps.items():
                ok, frame = cap.read()
                if not ok or frame is None:
                    drop[name] += 1
                    continue
                if frame.shape[1] != args.width or frame.shape[0] != args.height:
                    frame = cv2.resize(frame, (args.width, args.height))
                if publish_bgr_frame(pubs[name], np.ascontiguousarray(frame)):
                    sent[name] += 1
                else:
                    drop[name] += 1
            elapsed = time.time() - loop_start
            if elapsed < interval:
                time.sleep(interval - elapsed)
    finally:
        for cap in caps.values():
            cap.release()
        for pub in pubs.values():
            pub.close()

    for name in caps:
        st = pubs[name].stats() if hasattr(pubs[name], "stats") else {}
        print(f"[STAT] {name} sent={sent[name]} drop={drop[name]} pub={st}")
    print("[DONE] P5 smoke finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
