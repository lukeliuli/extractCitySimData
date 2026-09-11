#!/usr/bin/env bash
# ============================================================
# Jupyter Lab 安装 + tmux 后台运行 整合脚本
#   功能：
#     1. 安装 jupyterlab 到指定 conda 环境（默认 base）
#     2. 自动生成/复用密码写入 ~/.jupyter/jupyter_server_config.py
#     3. 用 tmux 后台启动，指定目录 /home/tianbot、端口 8888
#     4. 日志输出到 ~/.jupyter/jupyter.log
#   用法：
#     bash jupyter_all.sh install [env_name]   # 安装到环境（默认 base）
#     bash jupyter_all.sh start                # tmux 后台启动
#     bash jupyter_all.sh status               # 查看状态/日志
#     bash jupyter_all.sh attach               # 进入会话
#     bash jupyter_all.sh stop                 # 结束会话
#     bash jupyter_all.sh [默认=start]
# ============================================================
set -o pipefail

ENV_NAME="${2:-base}"          # install 子命令的环境名
SESSION="jupyter"
NOTEBOOK_DIR="/home/tianbot"
CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"   # 按需改 conda 安装路径
PORT="8888"
JCONF="$HOME/.jupyter/jupyter_server_config.py"
LOG="$HOME/.jupyter/jupyter.log"

# ---------- 初始化 conda ----------
init_conda() {
    # shellcheck disable=SC1091
    source "$CONDA_BASE/etc/profile.d/conda.sh" 2>/dev/null \
        || source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null \
        || source "$HOME/anaconda3/etc/profile.d/conda.sh" 2>/dev/null \
        || { echo "[!] 未找到 conda，请设置 CONDA_BASE"; exit 1; }
}

# ---------- 安装 ----------
install() {
    echo ">> 安装到 conda 环境: $ENV_NAME"
    init_conda
    if [ "$ENV_NAME" = "base" ]; then
        conda activate base
    elif conda env list | grep -q "/$ENV_NAME$"; then
        conda activate "$ENV_NAME"
    else
        conda create -y -n "$ENV_NAME" python=3.10 && conda activate "$ENV_NAME"
    fi
    conda install -y -n "$ENV_NAME" jupyterlab || pip install --upgrade jupyterlab
    echo ">> 安装完成，可使用: bash $0 start"
}

# ---------- 密码配置（若无则生成） ----------
ensure_password() {
    mkdir -p "$HOME/.jupyter"
    if [ -f "$JCONF" ] && grep -q "ServerApp.password" "$JCONF"; then
        echo ">> 已存在密码配置: $JCONF"
        return
    fi
    PW=$(python -c "import secrets,string;print(''.join(secrets.choice(string.ascii_letters+string.digits) for _ in range(12)))")
    PWHASH=$(python -c "from jupyter_server.auth import passwd;print(passwd('$PW'))")
    cat > "$JCONF" <<EOF
# 由 jupyter_all.sh 自动生成
c.ServerApp.password = '$PWHASH'
c.ServerApp.port = $PORT
c.ServerApp.notebook_dir = '$NOTEBOOK_DIR'
c.ServerApp.open_browser = False
c.ServerApp.allow_root = True
c.ServerApp.ip = '0.0.0.0'
EOF
    echo ""
    echo "=================================================="
    echo "  首次运行，生成密码: $PW  (请保存)"
    echo "  配置已写入: $JCONF"
    echo "=================================================="
}

CMD_START() {
    # 若未激活 conda/jupyter，激活 base
    if ! command -v jupyter >/dev/null 2>&1; then
        # shellcheck disable=SC1091
        source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null \
            || source "$HOME/anaconda3/etc/profile.d/conda.sh" 2>/dev/null
        conda activate base 2>/dev/null || true
    fi
    jupyter lab \
        --port="$PORT" \
        --notebook-dir="$NOTEBOOK_DIR" \
        --ServerApp.ip='0.0.0.0' \
        --ServerApp.allow_root=True \
        --ServerApp.open_browser=False
}

start() {
    ensure_password
    if tmux has-session -t "$SESSION" 2>/dev/null; then
        echo ">> 会话 $SESSION 已存在，跳过启动。查看: bash $0 status"
        exit 0
    fi
    # 若 8888 端口已被 Jupyter 监听，说明已有实例在运行，不重复启动
    if command -v ss >/dev/null 2>&1; then
        L_PORT=$(ss -ltnp 2>/dev/null | grep -c ":$PORT " || true)
    else
        L_PORT=$(netstat -ltnp 2>/dev/null | grep -c ":$PORT " || true)
    fi
    if [ "${L_PORT:-0}" -gt 0 ]; then
        echo ">> 端口 $PORT 已有 Jupyter 实例在运行，跳过启动。"
        echo "   URL: http://localhost:$PORT"
        exit 0
    fi
    echo ">> 创建 tmux 会话 $SESSION 并后台运行 Jupyter"
    tmux new-session -d -s "$SESSION" "bash -c '$(declare -f CMD_START); CMD_START' 2>&1 | tee $LOG"
    echo ">> 已启动。日志: $LOG"
    sleep 2
    status
}

stop() {
    tmux kill-session -t "$SESSION" 2>/dev/null && echo ">> 已结束会话 $SESSION" \
        || echo ">> 会话 $SESSION 不存在"
}

status() {
    if tmux has-session -t "$SESSION" 2>/dev/null; then
        echo ">> 会话 $SESSION 运行中"
        echo "   URL: http://localhost:$PORT"
    else
        echo ">> 会话 $SESSION 未在运行"
    fi
    echo "---- 最近日志 ----"
    [ -f "$LOG" ] && tail -n 15 "$LOG" || echo "(无日志)"
}

case "${1:-start}" in
    install) install ;;
    start)   start ;;
    stop)    stop ;;
    status)  status ;;
    attach)  tmux attach -t "$SESSION" ;;
    *) echo "用法: $0 [install [env]|start|stop|status|attach]"; exit 1 ;;
esac