#!/usr/bin/env python3
"""方块搬运 Rerun 记录的"快版"：让控制线程永远不等待 Rerun。

为什么需要它（实测 + SDK 反汇编证据）
------------------------------------
1. ``rr.log()`` 在**调用方线程**上同步执行：rerun/_log.py:212 →
   ChunkBatcherInner::send_cmd → re_quota_channel::sync::Sender::send_impl →
   ``crossbeam_channel::Sender::send``（阻塞版，不是 try_send）。SDK 二进制里还留着
   它自己的告警文案 "Channel byte budget (N) exceeded. Blocking until space is
   available"。0.31.3 的 ChunkBatcherConfig 只有 4 个 flush 字段，**没有任何
   "满了就丢"的开关**。所以下游服务端跟不上时，这个调用会**卡住调用线程** ——
   而调用它的地方正是控制循环（piper_cube_record.py:653，
   位于 send_action() 之后、precise_sleep 之前，直接吃掉 1/30 s 的周期预算）。
2. 相机实体是 ``static=True`` 的**覆盖式**实体（visualization_utils.py:94），
   30 Hz 发 30 次只有最后 1 次有效，另外 29 次是纯重复。
3. 单次脚本的复位阶段（piper_cube_record.py:931-944 没传
   display_compressed_images）走 compress_images=False，每步发**未压缩**的
   480x640x3 = 2.76 MB。

本模块做三件事
--------------
1. **判帧**：标量（关节曲线）每步都发；相机帧按 CUBE_RERUN_IMG_HZ 降频（默认 5 Hz）。
2. **搬线程**：默认 async —— 控制线程只做"判帧 + 拷贝 + put_nowait"，
   由后台线程调用**原来的** log_rerun_data；队列满则**丢最旧**。
   最坏情况是曲线缺几个点、画面跳一帧，机械臂的动作节奏不受影响。
3. **强制 compress_images=True**：把复位阶段每步 2.76 MB 的未压缩图变成 JPEG。

边界：本模块不改动任何既有文件；它由 sitecustomize.py 在解释器启动时装上。
导入本模块本身**不** import rerun，也不连任何服务端。
"""
from __future__ import annotations

import atexit
import os
import queue
import threading
import time

import numpy as np

__all__ = ["FastLogger", "install", "get_logger"]

_DISABLE_VALUES = ("0", "off", "false", "no")

DEFAULT_IMG_HZ = 5.0
DEFAULT_QUEUE_LEN = 3

_LOGGER: FastLogger | None = None


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return float(default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return int(default)


class FastLogger:
    """把 lerobot 的 log_rerun_data 包一层：判帧 + 可选异步。

    实例本身是"可调用对象"，签名与原来的 log_rerun_data 一致，所以能直接替换。
    """

    def __init__(self, original):
        self._original = original
        self.mode = os.environ.get("CUBE_RERUN_MODE", "async").strip().lower()
        if self.mode not in ("async", "sync"):
            self.mode = "async"
        self.img_hz = _env_float("CUBE_RERUN_IMG_HZ", DEFAULT_IMG_HZ)
        self.queue_len = max(1, _env_int("CUBE_RERUN_QUEUE", DEFAULT_QUEUE_LEN))
        self._interval = (1.0 / self.img_hz) if self.img_hz > 0 else None
        self._next_img_t = 0.0  # 第一帧立刻发，页面上马上有画面

        self.steps = 0
        self.img_batches = 0
        self.img_frames = 0
        self.dropped = 0
        self.errors = 0
        self._error_reported = False

        self._queue = queue.Queue(maxsize=self.queue_len) if self.mode == "async" else None
        self._worker = None
        if self._queue is not None:
            self._worker = threading.Thread(
                target=self._worker_loop, name="cube-rerun-fast", daemon=True
            )
            self._worker.start()

    # ------------------------------------------------------------------ 描述
    def describe(self) -> str:
        flush = os.environ.get("RERUN_FLUSH_NUM_BYTES", "(未设)")
        return (
            f"模式={self.mode} 相机={self.img_hz:g}Hz 队列={self.queue_len} "
            f"flush={flush}B"
        )

    def summary(self) -> str:
        return (
            f"步数={self.steps} 相机帧批次={self.img_batches} 相机图={self.img_frames} "
            f"丢弃={self.dropped} 记录错误={self.errors} {self.describe()}"
        )

    # ------------------------------------------------- 控制线程侧（必须极快）
    def __call__(self, observation=None, action=None, compress_images=False):
        """替换原 log_rerun_data。控制线程只走这一条路径，永不阻塞。"""
        self.steps += 1
        payload = self._prepare(observation, action)
        if self._queue is None:  # sync 模式：原地调，但已经降过频
            self._call(payload)
        else:
            self._enqueue(payload)

    @staticmethod
    def _is_image(value) -> bool:
        """与原 log_rerun_data:84 的判据保持一致：只认 numpy 数组，ndim>=2 当图像。

        故意不把 torch tensor 之类转成 numpy —— 原函数对它们本来就是忽略，
        转过来反而会凭空多出"图像"。
        """
        return isinstance(value, np.ndarray) and value.ndim >= 2

    def _prepare(self, observation, action):
        """判帧 + 拷贝。图像只在**要发的那一帧**才拷贝。"""
        now = time.monotonic()
        if self._interval is None:
            send_images = False
        elif now >= self._next_img_t:
            send_images = True
            self._next_img_t = now + self._interval
        else:
            send_images = False

        obs = self._copy_observation(observation, send_images)
        act = self._copy_plain(action)
        return (obs, act, True)  # 一律按 JPEG 发送

    def _copy_observation(self, observation, send_images: bool):
        if observation is None:
            return None
        if not isinstance(observation, dict):
            return observation
        out = {}
        for key, value in observation.items():
            if self._is_image(value):
                if not send_images:
                    # 这一帧不发画面。实体是覆盖式的，页面继续显示上一帧。
                    continue
                out[key] = np.array(value, copy=True)
                self.img_frames += 1
            elif isinstance(value, np.ndarray):
                out[key] = np.array(value, copy=True)
            else:
                out[key] = value
        if send_images:
            self.img_batches += 1
        return out

    @staticmethod
    def _copy_plain(mapping):
        """action 侧：只拷贝 ndarray，其余原样透传（保持原函数的判据不变）。"""
        if mapping is None or isinstance(mapping, np.ndarray):
            return np.array(mapping, copy=True) if isinstance(mapping, np.ndarray) else mapping
        if not isinstance(mapping, dict):
            return mapping
        return {
            key: (np.array(value, copy=True) if isinstance(value, np.ndarray) else value)
            for key, value in mapping.items()
        }

    # ----------------------------------------------------------- 后台线程侧
    def _enqueue(self, payload) -> None:
        try:
            self._queue.put_nowait(payload)
            return
        except queue.Full:
            pass
        # 队列满 = 下游已经跟不上。丢**最旧**的，让页面尽快追上当前状态。
        self.dropped += 1
        try:
            self._queue.get_nowait()
        except queue.Empty:
            pass
        try:
            self._queue.put_nowait(payload)
        except queue.Full:
            pass

    def _worker_loop(self) -> None:
        while True:
            payload = self._queue.get()
            if payload is None:  # 收尾哨兵
                return
            self._call(payload)

    def _call(self, payload) -> None:
        """真正调用原 log_rerun_data。它会阻塞（这是 SDK 的行为），
        但阻塞的是这个后台线程，不是控制线程。"""
        observation, action, compress_images = payload
        try:
            self._original(
                observation=observation, action=action, compress_images=compress_images
            )
        except Exception as exc:  # noqa: BLE001  监控绝不能搞崩业务
            self.errors += 1
            if not self._error_reported:
                self._error_reported = True
                print(
                    f"[cube_rerun_fast] 记录失败（只报一次，之后静默）："
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )

    def close(self, timeout: float = 2.0) -> None:
        """尽量把队列里剩下的发完；发不完就算了（daemon 线程随进程退出）。"""
        if self._worker is None:
            return
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        self._worker.join(timeout=timeout)


def install(original):
    """用 original（原来的 log_rerun_data）构造快版并登记退出汇总。幂等。"""
    global _LOGGER
    if _LOGGER is None:
        _LOGGER = FastLogger(original)
        atexit.register(_on_exit, _LOGGER)
    return _LOGGER


def get_logger():
    return _LOGGER


def _on_exit(logger: FastLogger) -> None:
    logger.close()
    print(f"[cube_rerun_fast] 汇总 {logger.summary()}", flush=True)


def is_disabled() -> bool:
    return os.environ.get("CUBE_RERUN_FAST", "1").strip().lower() in _DISABLE_VALUES
