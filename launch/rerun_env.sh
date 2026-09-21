# --- repository configuration (config/env.sh) -------------------------------
_piper_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
while [[ "$_piper_dir" != "/" && ! -f "$_piper_dir/config/env.sh" ]]; do
    _piper_dir="$(dirname -- "$_piper_dir")"
done
# shellcheck source=/dev/null
source "$_piper_dir/config/env.sh"
unset _piper_dir
# Rerun 通路的环境设置 —— 由 launch/ 里的启动脚本 source。
#
# 这份文件解决的是「环境自带的 rerun 版本太旧」：lerobot 环境里可能是 0.22.x，
# 而 `rr.Scalars` 要 0.31 才有，不顶掉的话 --display_data=true 一启动就
# AttributeError。所以它做三件事：可选地顶一个新版 SDK、设字节码缓存、给出
# 查看器地址。

PROJECT_ROOT="${PROJECT_ROOT:-$LEROBOT_REPO}"

# 1) 可选：把 `RERUN_SDK_ROOT` 指向的 Rerun SDK 顶到 sys.path 最前，压掉环境
#    自带的那份。config/env.sh 里默认留空 —— 留空就什么都不做，用环境里的版本
#    （前提是那份 >= 0.31）。要用就：
#        RERUN_SDK_ROOT=/opt/rerun-sdk/0.31.3 bash cube_loop_rerun.sh
if [[ -n "${RERUN_SDK_ROOT:-}" ]]; then
    if [[ -d "$RERUN_SDK_ROOT" ]]; then
        export PYTHONPATH="$RERUN_SDK_ROOT:$PROJECT_ROOT:$PROJECT_ROOT/src:${PYTHONPATH:-}"
    else
        echo "WARNING: RERUN_SDK_ROOT=$RERUN_SDK_ROOT 不存在，忽略" >&2
    fi
fi

# 2) 字节码缓存目录。若那份 SDK 装在只读目录里，.pyc 写不进去，每次 import
#    rerun 都要把几百个模块从源码重编译一遍，启动会多卡几分钟。
#    默认值在 config/env.sh（$PIPER_WS/.pycache）；这里只兜底。
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-$PIPER_WS/.pycache}"

# 3) 查看器地址。
export CUBE_RERUN_IP="${CUBE_RERUN_IP:-127.0.0.1}"
export CUBE_RERUN_PORT="${CUBE_RERUN_PORT:-9876}"

# 注意：如果 vendor 目录里另有一份 python 包集合，不要把那份整个加进
# PYTHONPATH —— 模块树太大会让 import 卡死。只加 rerun 自己的目录。

# 4) [快版] 把 log_rerun_data 挪到后台线程 + 相机帧降频 + flush 阈值改回 1 MiB。
#    做法：把 launch/fast/ 顶到 PYTHONPATH 最前，python 启动时自动 import
#    该目录下的 sitecustomize.py 装补丁，**不修改任何 lerobot 源码**。
#    即时回退：CUBE_RERUN_FAST=off 前缀一次即可，行为与改动前一致。
#    原理与开关见 docs/RERUN_PERFORMANCE.md。
if [[ -f "${CUBE_RERUN_FAST_ROOT}/fast_env.sh" ]]; then
    # shellcheck source=/dev/null
    source "${CUBE_RERUN_FAST_ROOT}/fast_env.sh"
fi
