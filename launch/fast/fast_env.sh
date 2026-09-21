# --- repository configuration (config/env.sh) -------------------------------
_piper_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
while [[ "$_piper_dir" != "/" && ! -f "$_piper_dir/config/env.sh" ]]; do
    _piper_dir="$(dirname -- "$_piper_dir")"
done
# shellcheck source=/dev/null
source "$_piper_dir/config/env.sh"
unset _piper_dir
# 方块搬运 Rerun「快版」的环境设置 —— 由本目录下的 *_fast.sh source。
#
# 必须在 launch/*_rerun.sh 之前 source：本目录要进 PYTHONPATH，
# 好让 python 解释器启动时自动 import sitecustomize.py（它负责把
# log_rerun_data 换成快版）。

CUBE_RERUN_FAST_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# 1) 本目录进 PYTHONPATH 最前。各 launcher 都是**追加** ${PYTHONPATH}，
#    所以不会被后续 export 顶掉。
export PYTHONPATH="$CUBE_RERUN_FAST_DIR:${PYTHONPATH:-}"

# 2) 把 flush 阈值从 lerobot 写死的 8000 字节改回 SDK 官方默认的 1 MiB。
#    visualization_utils.py:37 读这个变量；8000 会让每步几十 KB 的图像
#    触发好几次发送，30 Hz 下每秒上百个小包。
export RERUN_FLUSH_NUM_BYTES="${RERUN_FLUSH_NUM_BYTES:-1048576}"

# 3) 开关（都有默认值，一般不用改）
export CUBE_RERUN_FAST="${CUBE_RERUN_FAST:-1}"        # off = 完全不装补丁，行为与原来一致
export CUBE_RERUN_MODE="${CUBE_RERUN_MODE:-async}"    # async（默认，控制线程不等）| sync（原地发但降频）
export CUBE_RERUN_IMG_HZ="${CUBE_RERUN_IMG_HZ:-5}"    # 相机帧频率；0 = 完全不记相机画面（关节曲线照旧）
export CUBE_RERUN_QUEUE="${CUBE_RERUN_QUEUE:-3}"      # 异步队列长度，满了丢最旧

# 4) 录像落在哪（默认开）。落在**本地盘**：网络挂载的工作区每写一字节都要上行，
#    而一次 15 秒回合是几 MB 到几十 MB 的持续写入，会和 30 Hz 的控制链路抢带宽。
#    想写进工作区就自己设 CUBE_RERUN_SAVE_DIR。
# CUBE_RERUN_SAVE_DIR 的默认值在 config/env.sh 里给（$PIPER_WS/rerun_data）。
export CUBE_RERUN_SAVE="${CUBE_RERUN_SAVE:-1}"        # off = 不落录像（查看器照旧实时显示）
