#!/usr/bin/env bash
# Fallback when systemd --user is unavailable: runs the queue in a detached tmux session that survives logout.
cd "$(dirname "$0")/../.."
command -v tmux >/dev/null || { echo "tmux not installed; use: nohup .venv/bin/python deploy/server/queue_runner.py deploy/server/queue.yaml > ~/aspire-queue/queue.out 2>&1 &"; exit 1; }
tmux new-session -d -s aspire-queue ".venv/bin/python deploy/server/queue_runner.py deploy/server/queue.yaml 2>&1 | tee -a $HOME/aspire-queue/queue.out"
echo "started; watch with: tmux attach -t aspire-queue   (detach: Ctrl-b d)"
