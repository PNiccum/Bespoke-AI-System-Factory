#!/usr/bin/env bash
# One-time setup of ASPIRE on the professor's Linux server. Safe to run again. Needs no root except for
# installing missing system packages (it tells you what to ask for instead of doing it).
#   bash deploy/server/setup.sh            (from a fresh clone, e.g. ~/aspire)
set -euo pipefail
cd "$(dirname "$0")/../.."
say() { printf '\n== %s\n' "$*"; }
fail() { printf 'PROBLEM: %s\n' "$*" >&2; exit 1; }

say "Python 3.11+"
PY=${PYTHON:-}
if [ -z "$PY" ]; then for c in python3.12 python3.11 python3; do command -v "$c" >/dev/null && PY=$c && break; done; fi
[ -n "$PY" ] || fail "no python3 found; ask the server admin for Python 3.11 or newer"
"$PY" -c 'import sys; sys.exit(sys.version_info < (3, 11))' || fail "$PY is older than 3.11; set PYTHON=/path/to/python3.11"
"$PY" --version
command -v git >/dev/null || fail "git is not installed"

say "Virtual environment (.venv) and install"
[ -d .venv ] || "$PY" -m venv .venv
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q -e ".[dev,ui]"

say ".env (keys): created empty, readable by you only"
if [ ! -f .env ]; then cp .env.example .env; fi
chmod 600 .env
echo "Edit .env yourself (nano .env) and put ARC_API_KEY in. Never paste a key into a chat or a command line."

say "Container sandbox for generated code"
RUNNER=""
for r in docker podman; do command -v "$r" >/dev/null && "$r" info >/dev/null 2>&1 && RUNNER=$r && break; done
if [ -n "$RUNNER" ]; then
  "$RUNNER" build -q -t aspire-sandbox sandbox/ && echo "built aspire-sandbox with $RUNNER"
  echo "To use it, set 'sandbox: {runner: $RUNNER}' in configs/factory.yaml or export ASPIRE_SANDBOX=$RUNNER in .env."
else
  echo "No working docker/podman for this user. Runs fall back to the 'process' sandbox (works, less isolated)."
  echo "Ask the admin to add you to the docker group (or install podman) if you want container isolation."
fi

say "Queue and logs"
mkdir -p "$HOME/aspire-queue/logs"
[ -f deploy/server/queue.yaml ] || cp deploy/server/queue.example.yaml deploy/server/queue.yaml
echo "Edit deploy/server/queue.yaml to pick jobs (dry run: .venv/bin/python deploy/server/queue_runner.py deploy/server/queue.yaml --dry-run)"

say "Systemd user units (no root)"
mkdir -p "$HOME/.config/systemd/user"
for u in aspire-queue.service aspire-app.service; do
  sed "s#__ROOT__#$PWD#g" "deploy/server/$u" > "$HOME/.config/systemd/user/$u"
done
if systemctl --user show-environment >/dev/null 2>&1; then
  systemctl --user daemon-reload
  echo "Installed. Start with: systemctl --user start aspire-queue   (and: loginctl enable-linger \$USER so it survives logout)"
else
  echo "systemd --user is not available in this shell; use deploy/server/run_tmux.sh instead."
fi

say "Next: bash deploy/server/preflight.sh"
