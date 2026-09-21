"""BGR → H.264 MPEG-TS over TCP for go2rtc fan-out (no V4L2).

Design: Python listens on TCP; when go2rtc connects, spawn ffmpeg
(stdin=raw BGR, stdout=socket). With no viewer, frames are dropped so the
capture hot path never blocks on a full pipe.

go2rtc source (Ubuntu ffmpeg 4.4): use exec:ffmpeg -i tcp://… — not built-in
`ffmpeg:` (the built-in source is not supported by the ffmpeg we have).
"""
from __future__ import annotations

import logging
import shutil
import socket
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

DEFAULT_PORTS = {
    "left": 19012,
    "middle": 19011,
    "right": 19013,
}


@dataclass(frozen=True)
class PublisherConfig:
    """Listen endpoint + encode settings."""

    name: str
    host: str = "127.0.0.1"
    port: int = 19011
    width: int = 640
    height: int = 480
    fps: int = 15
    ffmpeg_bin: str = "ffmpeg"


class BgrTcpMpegtsPublisher:
    """Accept one MPEG-TS client at a time; encode BGR only while connected."""

    def __init__(self, cfg: PublisherConfig):
        self.cfg = cfg
        self._sock: Optional[socket.socket] = None
        self._accept_thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._ffmpeg: Optional[subprocess.Popen] = None
        self._client: Optional[socket.socket] = None
        self._frames_sent = 0
        self._frames_drop = 0
        self._last_err = ""

    @property
    def url(self) -> str:
        """TCP URL for go2rtc `exec:ffmpeg -i …`."""
        return f"tcp://{self.cfg.host}:{self.cfg.port}"

    @property
    def go2rtc_exec_src(self) -> str:
        """go2rtc stream line: pull this listen URL (copy H.264)."""
        return (
            "exec:ffmpeg -hide_banner -fflags nobuffer+genpts -flags low_delay "
            f"-i {self.url} -an -c:v copy -f mpegts -"
        )

    def start(self) -> None:
        """Bind TCP listen and start accept loop."""
        if self._accept_thread and self._accept_thread.is_alive():
            return
        if not shutil.which(self.cfg.ffmpeg_bin):
            raise FileNotFoundError(f"ffmpeg not found: {self.cfg.ffmpeg_bin}")
        self._stop.clear()
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((self.cfg.host, self.cfg.port))
        sock.listen(1)
        sock.settimeout(0.5)
        self._sock = sock
        self._accept_thread = threading.Thread(
            target=self._accept_loop, name=f"fanout-{self.cfg.name}", daemon=True
        )
        self._accept_thread.start()
        log.info(
            "publisher listen name=%s url=%s %dx%d@%d",
            self.cfg.name,
            self.url,
            self.cfg.width,
            self.cfg.height,
            self.cfg.fps,
        )

    def publish_bgr_frame(self, arr: np.ndarray) -> bool:
        """Write one HxWx3 uint8 BGR frame if a viewer is connected; else drop."""
        if arr is None or not isinstance(arr, np.ndarray):
            return False
        if arr.dtype != np.uint8 or arr.ndim != 3 or arr.shape[2] != 3:
            return False
        h, w, _ = arr.shape
        if h != self.cfg.height or w != self.cfg.width:
            return False
        with self._lock:
            ff = self._ffmpeg
            if not ff or ff.poll() is not None or not ff.stdin:
                self._frames_drop += 1
                return False
            try:
                ff.stdin.write(np.ascontiguousarray(arr).tobytes())
                self._frames_sent += 1
                return True
            except (BrokenPipeError, ValueError, OSError) as e:
                self._last_err = str(e)
                self._frames_drop += 1
                return False

    def stats(self) -> dict:
        """Health counters for tests / ops."""
        with self._lock:
            alive_ff = bool(self._ffmpeg and self._ffmpeg.poll() is None)
            has_client = self._client is not None
        return {
            "name": self.cfg.name,
            "url": self.url,
            "listening": bool(self._sock),
            "client": has_client,
            "ffmpeg_alive": alive_ff,
            "frames_sent": self._frames_sent,
            "frames_drop": self._frames_drop,
            "last_err": self._last_err[-400:],
        }

    def close(self) -> None:
        """Stop accept loop, ffmpeg, and sockets."""
        self._stop.set()
        self._teardown_client()
        if self._sock:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None
        if self._accept_thread:
            self._accept_thread.join(timeout=2)
            self._accept_thread = None
        log.info(
            "publisher stop name=%s sent=%d drop=%d",
            self.cfg.name,
            self._frames_sent,
            self._frames_drop,
        )

    def _accept_loop(self) -> None:
        """Accept one client, run ffmpeg until disconnect, repeat."""
        assert self._sock is not None
        while not self._stop.is_set():
            try:
                client, addr = self._sock.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            log.info("publisher client connected name=%s from=%s", self.cfg.name, addr)
            try:
                self._run_session(client)
            finally:
                self._teardown_client()
                log.info("publisher client gone name=%s", self.cfg.name)

    def _run_session(self, client: socket.socket) -> None:
        """Pipe BGR stdin → H.264 MPEG-TS → client socket until EOF/error."""
        c = self.cfg
        cmd = [
            c.ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{c.width}x{c.height}",
            "-r",
            str(c.fps),
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-tune",
            "zerolatency",
            "-pix_fmt",
            "yuv420p",
            "-g",
            str(max(c.fps, 1)),
            "-bf",
            "0",
            "-f",
            "mpegts",
            "pipe:1",
        ]
        try:
            ff = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
            )
        except Exception as e:
            self._last_err = str(e)
            client.close()
            return

        with self._lock:
            self._ffmpeg = ff
            self._client = client

        def pump_out() -> None:
            """Copy MPEG-TS bytes from ffmpeg stdout to the TCP client."""
            assert ff.stdout is not None
            try:
                while True:
                    chunk = ff.stdout.read(4096)
                    if not chunk:
                        break
                    client.sendall(chunk)
            except (BrokenPipeError, ConnectionResetError, OSError) as e:
                self._last_err = f"pump: {e}"
            finally:
                try:
                    if ff.stdin:
                        ff.stdin.close()
                except Exception:
                    pass

        pump = threading.Thread(target=pump_out, name=f"fanout-pump-{c.name}", daemon=True)
        pump.start()
        try:
            ff.wait(timeout=3600)
        except Exception:
            ff.kill()
        pump.join(timeout=2)
        err = b""
        try:
            if ff.stderr:
                err = ff.stderr.read() or b""
        except Exception:
            pass
        if err:
            self._last_err = err.decode("utf-8", errors="replace")

    def _teardown_client(self) -> None:
        """Close ffmpeg + client under lock."""
        with self._lock:
            ff = self._ffmpeg
            client = self._client
            self._ffmpeg = None
            self._client = None
        if ff:
            try:
                if ff.stdin:
                    ff.stdin.close()
            except Exception:
                pass
            if ff.poll() is None:
                ff.kill()
                try:
                    ff.wait(timeout=2)
                except Exception:
                    pass
        if client:
            try:
                client.close()
            except OSError:
                pass

    def __enter__(self) -> "BgrTcpMpegtsPublisher":
        self.start()
        return self

    def __exit__(self, *args) -> None:
        self.close()


def publish_bgr_frame(publisher: BgrTcpMpegtsPublisher, arr: np.ndarray) -> bool:
    """Module-level helper matching the target API name."""
    return publisher.publish_bgr_frame(arr)


def make_publisher(
    name: str,
    *,
    port: Optional[int] = None,
    width: int = 640,
    height: int = 480,
    fps: int = 15,
    host: str = "127.0.0.1",
) -> BgrTcpMpegtsPublisher:
    """Build publisher; default port from DEFAULT_PORTS for left/middle/right."""
    if port is None:
        port = DEFAULT_PORTS.get(name)
        if port is None:
            raise ValueError(f"unknown name {name!r}; pass port= explicitly")
    return BgrTcpMpegtsPublisher(
        PublisherConfig(name=name, host=host, port=port, width=width, height=height, fps=fps)
    )


def make_color_bar_bgr(width: int, height: int, t: float = 0.0) -> np.ndarray:
    """Synthetic BGR test pattern (no camera)."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    third = max(width // 3, 1)
    img[:, :third] = (0, 0, 255)
    img[:, third : 2 * third] = (0, 255, 0)
    img[:, 2 * third :] = (255, 0, 0)
    x = int((t * 80) % width)
    img[:, x : x + 4] = (255, 255, 255)
    return img
