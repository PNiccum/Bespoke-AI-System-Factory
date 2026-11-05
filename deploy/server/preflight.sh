#!/usr/bin/env bash
# Checks the server is ready for unattended runs. Prints pass/fail lines only; no key is ever printed.
cd "$(dirname "$0")/../.."
ok() { printf 'PASS  %s\n' "$*"; }
bad() { printf 'FAIL  %s\n' "$*"; rc=1; }
rc=0
[ -x .venv/bin/python ] && ok "venv present" || bad "no .venv (run setup.sh)"
[ -f .env ] && [ "$(stat -c %a .env)" = "600" ] && ok ".env mode 600" || bad ".env missing or not mode 600 (chmod 600 .env)"
set -a; . ./.env 2>/dev/null; set +a
[ -n "${ARC_API_KEY:-}" ] && ok "ARC_API_KEY set" || bad "ARC_API_KEY empty in .env"
code=$(curl -s -o /dev/null -w '%{http_code}' -m 20 -H "Authorization: Bearer ${ARC_API_KEY:-}" -H "x-api-key: ${ARC_API_KEY:-}" https://llm-api.arc.vt.edu/api/models || true)
[ "$code" = "200" ] && ok "ARC answers HTTP 200 from this server (no VPN needed here)" || bad "ARC returned HTTP $code (403 = not on a VT network; 401 = key)"
for r in docker podman; do command -v $r >/dev/null && $r image inspect aspire-sandbox >/dev/null 2>&1 && ok "$r sandbox image present"; done
.venv/bin/python -m pytest -q -x 2>&1 | tail -2 | sed 's/^/      /'
[ "${PIPESTATUS[0]}" = 0 ] && ok "pytest" || bad "pytest"
.venv/bin/ruff check . >/dev/null 2>&1 && ok "ruff" || bad "ruff"
[ -n "${ANTHROPIC_API_KEY:-}${OPENAI_API_KEY:-}" ] && echo "NOTE  paid keys are in .env; queue jobs refuse --ask/--redesign unless marked allow_paid, and private jobs never get them"
df -h "$HOME" | tail -1 | awk '{print "INFO  disk free in $HOME: " $4}'
exit $rc
