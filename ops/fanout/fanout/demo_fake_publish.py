#!/usr/bin/env python3
"""Smoke: push fake BGR bars for ~N seconds (optional side go2rtc check).

Usage:
  python3 ops/fanout/fanout/demo_fake_publish.py --seconds 8 --port 19121
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fanout.bgr_tcp_mpegts import make_color_bar_bgr, make_publisher, publish_bgr_frame


def main() -> int:
    """Publish synthetic frames until timeout."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="middle")
    ap.add_argument("--port", type=int, default=19121)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--seconds", type=float, default=5.0)
    args = ap.parse_args()

    pub = make_publisher(
        args.name, port=args.port, width=args.width, height=args.height, fps=args.fps
    )
    pub.start()
    print("listen", pub.url)
    print("go2rtc_src", pub.go2rtc_exec_src)
    t0 = time.time()
    i = 0
    ok = 0
    dt = 1.0 / max(args.fps, 1)
    try:
        while time.time() - t0 < args.seconds:
            frame = make_color_bar_bgr(args.width, args.height, t=i / float(args.fps))
            if publish_bgr_frame(pub, frame):
                ok += 1
            i += 1
            time.sleep(dt)
    finally:
        st = pub.stats()
        pub.close()
    print("done", {"ok": ok, "iters": i, "stats": st})
    return 0 if ok > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
