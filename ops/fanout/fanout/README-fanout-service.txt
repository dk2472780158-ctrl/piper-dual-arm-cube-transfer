PiPER RGB idle fan-out (go2rtc 空闲预览) — 默认关闭

政策（2026-08-31）：宿主机默认不占 camera_*，留给 lab 容器。
安装脚本：disable --now，勿 enable。

可选空闲预览：
      sudo systemctl start piper-rgb-fanout.service
      → 推 TCP 19011-19013 → /videos/ Tab「PiPER 工位」
      用完：sudo systemctl stop piper-rgb-fanout.service

日常采集/推理（推荐）：
      容器内 run_piper_record.sh / act_inference_*.sh
      ENABLE_PIPER_RGB_PUBLISH=1 旁路推流（与 idle 互斥）

安装：bash scripts/install_piper_rgb_fanout_service.sh
