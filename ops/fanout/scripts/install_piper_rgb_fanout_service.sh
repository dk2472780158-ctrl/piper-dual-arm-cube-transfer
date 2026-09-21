#!/usr/bin/env bash
# Install the piper-rgb-fanout unit（默认装好就 disable，不占相机）。
# 空闲远程预览按需：sudo systemctl start piper-rgb-fanout.service
# 日常采集/推理：任务进程独占 camera_*，脚本 ENABLE_PIPER_RGB_PUBLISH=1 旁路发布。
set -euo pipefail
# --- repository configuration (config/env.sh) -------------------------------
_piper_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
while [[ "$_piper_dir" != "/" && ! -f "$_piper_dir/config/env.sh" ]]; do
    _piper_dir="$(dirname -- "$_piper_dir")"
done
# shellcheck source=/dev/null
source "$_piper_dir/config/env.sh"
unset _piper_dir

ROOT="${PIPER_FANOUT_ROOT}"
UNIT=piper-rgb-fanout.service
SRC="${ROOT}/scripts/${UNIT}"
DST=/etc/systemd/system/${UNIT}

if [[ ! -f "${SRC}" ]]; then
  echo "[FAIL] missing ${SRC}" >&2
  exit 1
fi

sudo cp "${SRC}" "${DST}"
sudo systemctl daemon-reload
# 默认留给容器：不 enable、不 start（与 WantedBy 安装文件无关，显式 disable）
sudo systemctl disable --now "${UNIT}" 2>/dev/null || true
echo "installed ${DST}"
echo "default: disabled (cameras free for lab container)"
echo "idle preview (opt-in): sudo systemctl start ${UNIT}"
echo "Logs when started: journalctl -u ${UNIT} -f"
