#!/bin/bash
# ============================================================
# AI 辅助蛋白设计平台 - 服务启动脚本
# 用法:  bash run_server.sh [--reload] [--port 8848]
# ============================================================
set -euo pipefail

PROJECT_DIR="/home/quanyj/LIUDEMEI/projects/danbaizhi"
CONDA_ENV="dbz"
CONDA_SH="/home/quanyj/huguoqiang/anaconda3/etc/profile.d/conda.sh"

# shellcheck source=/dev/null
source "${CONDA_SH}"
conda activate "${CONDA_ENV}"

# 本机 ~/.local 下存在 python3.11 用户级 site-packages，会污染 conda 环境，必须屏蔽
export PYTHONNOUSERSITE=1
export PIP_USER=0
# 实测 huggingface.co 直连失败（SSL），统一走镜像
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

cd "${PROJECT_DIR}"

HOST="${DBZ_HOST:-0.0.0.0}"
PORT="${DBZ_PORT:-8848}"
RELOAD=""
#: --reload 的监听范围。默认 uvicorn 会递归监听**整个项目目录**，而本项目里有
#: models/hf_cache（5.2 GB 权重）、data/cache、logs 等大树，实测会直接顶爆
#: inotify 上限并报 "OS file watch limit reached (MaxFilesWatch)" ——
#: 结果是 --reload 看着在跑、实际**一个文件都不监听**，改完代码以为会自动生效
#: 其实永远不会。限定只监听 backend 代码目录即可彻底避开。
#: 注意：configs/*.yaml 不在此列（配置走 lru_cache，本来就需要重启才生效）。
RELOAD_DIR=""

# 必须用 while + shift 逐个消费参数：
# 早期版本写成 ``for arg in "$@"; do case ... --port) shift ;; esac; done``，
# 但 for 循环一开始就已固定了 "$@" 的取值，循环体内 shift 对本次迭代毫无影响，
# 结果是 ``--port 8849`` 被**静默忽略**（脚本头部却声明支持该用法）。
while [ "$#" -gt 0 ]; do
  case "$1" in
    --reload)
      RELOAD="--reload"
      RELOAD_DIR="--reload-dir backend"
      shift
      ;;
    --port)
      PORT="${2:-}"
      shift 2 2>/dev/null || shift
      ;;
    --port=*) PORT="${1#*=}"; shift ;;
    -h|--help)
      echo "用法: bash run_server.sh [--reload] [--port <端口>]"
      echo "  --reload        代码变更后自动重启（开发调试用）"
      echo "  --port <端口>   监听端口，默认 8848（也可用环境变量 DBZ_PORT）"
      exit 0
      ;;
    *)
      echo "未知参数: $1（用 --help 查看用法）"
      exit 1
      ;;
  esac
done

# 端口必须是纯数字，否则 ss 的过滤条件会失效并静默放行
case "${PORT}" in
  ''|*[!0-9]*)
    echo "错误: 端口必须是数字，收到 '${PORT}'"
    exit 1
    ;;
esac

echo "=========================================================="
echo " 启动 AI 辅助蛋白设计平台"
echo " 项目目录 : ${PROJECT_DIR}"
echo " conda 环境: ${CONDA_ENV}"
echo " 监听地址 : http://${HOST}:${PORT}"
echo " 接口文档 : http://${HOST}:${PORT}/docs"
echo "=========================================================="

# ------------------------------------------------------------
# --reload 可用性预检
# ------------------------------------------------------------
# uvicorn 的 --reload 依赖 inotify，而 inotify 的 watch 配额**按用户**计算，
# 与项目大小无关。本机实测已被三个 IDE 的远程服务进程吃满 65536/65536：
#     VS Code Server 22577 + CodeBuddy Server 21615 + Trae CN Server 17878
# 配额满时 watchfiles 抛 OSError(MaxFilesWatch)，而 uvicorn **只把它打进日志、
# 服务照常启动**——表面看 --reload 在跑，实际一个文件都不监听，改完代码永远
# 不生效（本项目就被这个坑过：改了校验逻辑却一直不生效）。
# 这里主动探测，把"静默失效"变成"明确告知 + 自动降级为手动重启"。
inotify_has_capacity() {
  python - <<'PY' 2>/dev/null
import ctypes, os, sys

try:
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    libc.inotify_init1.argtypes = [ctypes.c_int]
    libc.inotify_init1.restype = ctypes.c_int
    libc.inotify_add_watch.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_uint32]
    libc.inotify_add_watch.restype = ctypes.c_int

    fd = libc.inotify_init1(os.O_NONBLOCK)
    if fd < 0:
        sys.exit(1)
    # IN_ALL_EVENTS = 0x00000FFF；配额耗尽时返回 ENOSPC 并使 wd < 0
    wd = libc.inotify_add_watch(fd, os.getcwd().encode(), 0x00000FFF)
    os.close(fd)
    sys.exit(0 if wd >= 0 else 1)
except Exception:
    sys.exit(1)
PY
}

if [ -n "${RELOAD}" ] && ! inotify_has_capacity; then
  echo ""
  echo "!! --reload 不可用：当前用户的 inotify watch 配额已耗尽，文件监听无法建立。"
  echo "   （uvicorn 遇到这种情况只会把错误打进日志、服务照常启动，"
  echo "     表面看似正常，实际改代码永远不会生效）"
  echo ""
  echo "   当前上限： $(cat /proc/sys/fs/inotify/max_user_watches 2>/dev/null || echo '未知')"
  echo "   方案一（需 sudo，一劳永逸）："
  echo "       sudo sysctl -w fs.inotify.max_user_watches=524288"
  echo "       # 持久化： echo 'fs.inotify.max_user_watches=524288' | sudo tee /etc/sysctl.d/99-inotify.conf"
  echo "   方案二：关闭多余的 IDE 远程会话（本机 VS Code / Trae 各占 2 万+ watch）"
  echo ""
  echo "   本次将**不带 --reload** 启动（服务可正常使用），改代码后请手动重启本脚本。"
  RELOAD=""
  RELOAD_DIR=""
fi

# ------------------------------------------------------------
# 端口占用处理
# ------------------------------------------------------------
# 为什么需要这段：uvicorn 在端口被占用时只报一句
# ``[Errno 98] Address already in use`` 就退出，不告诉你是谁占的、
# 更不会区分"是上次没停干净的自己"还是"别的服务"。
# 这里让脚本自己把问题说清楚，并且只在确认占用者是**本项目自己的实例**时
# 才自动释放，避免误杀别的服务。
listening_pids() {
  if command -v ss >/dev/null 2>&1; then
    # ss -ltnp 的第 4 列是本地地址，形如 0.0.0.0:8848
    ss -ltnp 2>/dev/null \
      | awk -v p=":${PORT}" '$4 ~ p"$"' \
      | grep -o 'pid=[0-9]*' | cut -d= -f2 | sort -u
  elif command -v lsof >/dev/null 2>&1; then
    lsof -ti "tcp:${PORT}" -sTCP:LISTEN 2>/dev/null | sort -u
  fi
}

# 列出与某 PID 同属"本项目服务"的全部进程：它自身 + 祖先链上带 uvicorn 标记的进程。
#
# 为什么必须向上追溯
# ------------------
# ``uvicorn --reload`` 的真实工作进程由 multiprocessing 派生，命令行为
# ``python -s -c from multiprocessing.spawn import spawn_main ...``，
# **完全不含 uvicorn 字样**；带标记的是它的父进程（reloader）。只看自身命令行，
# 会把 reload 工作进程误判成"别人的进程"，导致自己的服务一旦用 --reload 启动，
# 就再也无法用本脚本重启。
#
# 为什么要把父进程也一起返回
# --------------------------
# reloader 父进程的主循环在子进程退出后会**立即再拉起一个新子进程**。只杀掉持有
# 端口的子进程，父进程马上会重新抢占端口，新服务照样绑定失败。必须连同 reloader
# 一并终止，因此这里返回整个"自有进程树"而不只是端口持有者。
OWN_SERVER_ANCESTOR_DEPTH=5
OWN_SERVER_MARKER="uvicorn backend.app.main:app"

collect_own_tree() {
  local cur="$1" depth=0 cmdline found=""
  while [ -n "${cur}" ] && [ "${cur}" != "0" ] && [ "${cur}" != "1" ] \
    && [ "${depth}" -lt "${OWN_SERVER_ANCESTOR_DEPTH}" ]; do
    if [ -r "/proc/${cur}/cmdline" ]; then
      cmdline="$(tr '\0' ' ' < "/proc/${cur}/cmdline" 2>/dev/null)"
      case "${cmdline}" in
        *"${OWN_SERVER_MARKER}"*) found="${found} ${cur}" ;;
      esac
    fi
    # /proc/<pid>/stat 的字段 2 是带括号的 comm（可能含空格），用 ps 取 ppid 更稳
    cur="$(ps -o ppid= -p "${cur}" 2>/dev/null | tr -d ' ')"
    depth=$((depth + 1))
  done
  echo "${found}"
}

is_own_server() {
  [ -n "$(collect_own_tree "$1")" ]
}

# 端口是否被监听。不依赖 ``ss -p``，因此对**其它用户**的进程同样有效——
# 否则遇到别人的进程时会误判为"端口空闲"，一路走到 uvicorn 报 Errno 98。
port_is_listening() {
  if command -v ss >/dev/null 2>&1; then
    ss -ltn 2>/dev/null | awk -v p=":${PORT}" '$4 ~ p"$"' | grep -q .
  elif command -v lsof >/dev/null 2>&1; then
    lsof -ti "tcp:${PORT}" -sTCP:LISTEN >/dev/null 2>&1
  else
    return 1
  fi
}

OCCUPIED_PIDS="$(listening_pids || true)"
if port_is_listening; then
  if [ -z "${OCCUPIED_PIDS}" ]; then
    echo ""
    echo "!! 端口 ${PORT} 已被占用，但无法识别占用进程（很可能属于其它用户）。"
    echo "   请自行确认，或换端口启动： bash run_server.sh --port 8849"
    echo "   排查命令： ss -ltnp | grep ':${PORT}'"
    exit 1
  fi

  OWN_PIDS=""
  FOREIGN_PIDS=""
  for pid in ${OCCUPIED_PIDS}; do
    if is_own_server "${pid}"; then
      OWN_PIDS="${OWN_PIDS} ${pid}"
    else
      FOREIGN_PIDS="${FOREIGN_PIDS} ${pid}"
    fi
  done

  if [ -n "${FOREIGN_PIDS}" ]; then
    echo ""
    echo "!! 端口 ${PORT} 被其它进程占用，脚本不会自动终止它："
    for pid in ${FOREIGN_PIDS}; do
      echo "   PID ${pid}: $(tr '\0' ' ' < "/proc/${pid}/cmdline" 2>/dev/null)"
    done
    echo "   请先确认后再手动处理，或换端口启动： bash run_server.sh --port 8849"
    exit 1
  fi

  if [ -n "${OWN_PIDS}" ]; then
    # 汇总"端口持有者 + 其 reloader 父进程"，去重后一并终止。
    # 只杀子进程不够：reloader 会立刻重启一个新的工作进程重新抢占端口。
    TARGET_PIDS=""
    for pid in ${OWN_PIDS}; do
      TARGET_PIDS="${TARGET_PIDS} ${pid} $(collect_own_tree "${pid}")"
    done
    TARGET_PIDS="$(echo ${TARGET_PIDS} | tr ' ' '\n' | grep -E '^[0-9]+$' | sort -un | tr '\n' ' ')"

    echo ""
    echo "检测到本项目旧实例仍占用端口 ${PORT}，正在释放：${TARGET_PIDS}"
    for pid in ${TARGET_PIDS}; do
      kill "${pid}" 2>/dev/null || true
    done
    # 给进程优雅退出的时间；超时则强杀
    for _ in $(seq 1 20); do
      still=""
      for pid in ${TARGET_PIDS}; do
        kill -0 "${pid}" 2>/dev/null && still="${still} ${pid}"
      done
      [ -z "${still}" ] && break
      sleep 0.5
    done
    for pid in ${TARGET_PIDS}; do
      if kill -0 "${pid}" 2>/dev/null; then
        echo "   PID ${pid} 未响应 SIGTERM，强制终止"
        kill -9 "${pid}" 2>/dev/null || true
      fi
    done
    # 进程退出 ≠ 端口立即可用：套接字回收需要一点时间，不等就启动仍会撞 Errno 98
    for _ in $(seq 1 20); do
      port_is_listening || break
      sleep 0.5
    done
    if port_is_listening; then
      echo "   警告：端口 ${PORT} 仍处于监听状态，继续启动可能失败"
    else
      echo "端口 ${PORT} 已释放"
    fi
  fi
fi

exec python -m uvicorn backend.app.main:app \
  --host "${HOST}" \
  --port "${PORT}" \
  --no-access-log \
  ${RELOAD} \
  ${RELOAD_DIR}
