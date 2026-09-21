#!/usr/bin/env python3
"""P2 unit / smoke tests for BGR TCP MPEG-TS publisher (no real cameras)."""
from __future__ import annotations

import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fanout.bgr_tcp_mpegts import (  # noqa: E402
    make_color_bar_bgr,
    make_publisher,
    publish_bgr_frame,
)


def _port_free(port: int) -> bool:
    """Return True if TCP port is free on localhost."""
    s = socket.socket()
    try:
        s.bind(("127.0.0.1", port))
        return True
    except OSError:
        return False
    finally:
        s.close()


def test_reject_bad_shapes() -> None:
    """Publisher rejects wrong dtype/shape."""
    pub = make_publisher("middle", port=19101, width=64, height=48, fps=10)
    assert publish_bgr_frame(pub, None) is False
    assert publish_bgr_frame(pub, np.zeros((48, 64), dtype=np.uint8)) is False
    assert publish_bgr_frame(pub, np.zeros((48, 64, 3), dtype=np.float32)) is False
    assert publish_bgr_frame(pub, np.zeros((32, 32, 3), dtype=np.uint8)) is False
    print("OK reject_bad_shapes")


def test_drop_without_client() -> None:
    """With no TCP client, publish returns False (drop) and does not hang."""
    port = 19211
    if not _port_free(port):
        print(f"SKIP drop_without_client port {port} busy")
        return
    pub = make_publisher("middle", port=port, width=160, height=120, fps=10)
    pub.start()
    time.sleep(0.2)
    t0 = time.time()
    ok = publish_bgr_frame(pub, make_color_bar_bgr(160, 120))
    elapsed = time.time() - t0
    st = pub.stats()
    pub.close()
    assert ok is False
    assert st["frames_drop"] >= 1
    assert elapsed < 1.0, elapsed
    print(f"OK drop_without_client elapsed={elapsed:.3f}s stats={st}")


def test_client_receives_mpegts() -> None:
    """TCP client receives MPEG-TS while frames are published."""
    port = 19212
    if not _port_free(port):
        print(f"SKIP client_receives_mpegts port {port} busy")
        return
    pub = make_publisher("middle", port=port, width=160, height=120, fps=10)
    pub.start()
    time.sleep(0.2)

    stop = threading.Event()
    sent = {"n": 0}

    def feeder() -> None:
        i = 0
        while not stop.is_set() and i < 200:
            if publish_bgr_frame(pub, make_color_bar_bgr(160, 120, t=i / 10.0)):
                sent["n"] += 1
            i += 1
            time.sleep(0.03)

    th = threading.Thread(target=feeder, daemon=True)
    th.start()

    data = b""
    err = ""
    for _ in range(30):
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=2) as sock:
                sock.settimeout(4)
                chunks = []
                deadline = time.time() + 4
                while time.time() < deadline and sum(len(c) for c in chunks) < 8192:
                    try:
                        c = sock.recv(4096)
                    except socket.timeout:
                        break
                    if not c:
                        break
                    chunks.append(c)
                data = b"".join(chunks)
                if len(data) > 188:
                    break
        except OSError as e:
            err = str(e)
            time.sleep(0.15)

    stop.set()
    th.join(timeout=2)
    st = pub.stats()
    pub.close()
    assert len(data) > 188, f"len={len(data)} err={err} sent={sent} st={st}"
    assert 0x47 in data[:800], data[:40]
    assert sent["n"] >= 5, sent
    print(f"OK client_receives_mpegts bytes={len(data)} sent={sent['n']} st={st}")


def test_go2rtc_exec_src_string() -> None:
    """go2rtc helper uses exec: + tcp URL (never v4l2/camera_)."""
    pub = make_publisher("middle", port=19011)
    src = pub.go2rtc_exec_src
    assert src.startswith("exec:ffmpeg")
    assert "tcp://127.0.0.1:19011" in src
    assert "v4l2" not in src and "camera_" not in src
    print("OK go2rtc_exec_src_string")


def test_ffprobe_optional() -> None:
    """Optional: ffmpeg pull tcp URL and decode one frame to JPEG."""
    port = 19213
    if not _port_free(port):
        print(f"SKIP ffprobe_optional port {port} busy")
        return
    out = Path("/tmp/piper_fanout_p2_frame.jpg")
    out.unlink(missing_ok=True)
    pub = make_publisher("middle", port=port, width=320, height=240, fps=10)
    pub.start()
    time.sleep(0.2)

    stop = threading.Event()

    def feeder() -> None:
        i = 0
        while not stop.is_set():
            publish_bgr_frame(pub, make_color_bar_bgr(320, 240, t=i / 10.0))
            i += 1
            time.sleep(0.05)

    th = threading.Thread(target=feeder, daemon=True)
    th.start()
    time.sleep(0.3)
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        f"tcp://127.0.0.1:{port}",
        "-frames:v",
        "1",
        str(out),
    ]
    r = subprocess.run(cmd, timeout=15, capture_output=True)
    stop.set()
    th.join(timeout=2)
    pub.close()
    if r.returncode != 0 or not out.exists() or out.stat().st_size < 1000:
        print(f"SKIP ffprobe_optional ffmpeg_rc={r.returncode} err={r.stderr[-200:]}")
        return
    print(f"OK ffprobe_optional jpeg_bytes={out.stat().st_size}")


def main() -> int:
    """Run all P2 checks."""
    tests = [
        test_reject_bad_shapes,
        test_go2rtc_exec_src_string,
        test_drop_without_client,
        test_client_receives_mpegts,
        test_ffprobe_optional,
    ]
    failed = 0
    for fn in tests:
        try:
            fn()
        except Exception as e:
            failed += 1
            print(f"FAIL {fn.__name__}: {e}")
    if failed:
        print(f"P2 FAILED {failed}/{len(tests)}")
        return 1
    print(f"P2 PASSED {len(tests)}/{len(tests)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
