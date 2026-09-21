#!/usr/bin/env python3
"""Rerun 服务端「收没收到数据」的对照探针。

probe_rerun.py 只证明「不发错」；这个探针要证明「数据真的进了服务端内存」。

手段：往服务端灌 ~几十 MB 数据，同时在外面采样 rerun 服务端进程的 VmRSS。
服务端带 --server-memory-limit 64MiB，数据是真进了内存，RSS 就会涨。
配合一个「打到没人监听的端口」的对照组，RSS 不涨 —— 一涨一不涨，才算证据。

不连机械臂、不开相机、不开 CAN。
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

SESSION = os.environ.get("CUBE_PROBE_SESSION", "cube_volume_probe")
PORT = int(os.environ.get("CUBE_PROBE_PORT", "9876"))
N_STEPS = int(os.environ.get("CUBE_PROBE_STEPS", "120"))
H, W = 720, 1280


def main() -> int:
    import rerun as rr

    print(f"[vol] rerun SDK {rr.__version__}", flush=True)
    rr.init(SESSION)
    url = f"rerun+http://127.0.0.1:{PORT}/proxy"
    print(f"[vol] connect -> {url}", flush=True)
    rr.connect_grpc(url=url)

    rng = np.random.default_rng(0)
    t0 = time.time()
    for step in range(N_STEPS):
        # 每步一张全新的随机图 —— 不是 static，必须真的存下来
        img = rng.integers(0, 256, size=(H, W, 3), dtype=np.uint8)
        rr.set_time("step", sequence=step)
        rr.log("vol/cam", rr.Image(img).compress(), static=False)
        for i in range(42):
            rr.log(f"vol/curve_{i}", rr.Scalars(float(np.sin(step * 0.1 + i))))
        if step % 20 == 0:
            print(f"[vol] step={step} t={time.time()-t0:.1f}s", flush=True)

    time.sleep(3.0)  # 等 flush
    print(f"[vol] VOLUME_DONE steps={N_STEPS} elapsed={time.time()-t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
