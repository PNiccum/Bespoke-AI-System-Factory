# Running the queue on the professor's server

Goal: the live queue (bench projects, proteomics tier, bench, research jobs) runs unattended on the professor's
Linux server, which reaches ARC directly, so no laptop and no VPN session are needed for the runs. The server
itself is reachable only on eduroam or the VT VPN (matters for the Streamlit app and for logging in, not for runs).

Everything here is prepared in `deploy/server/` and has not been run on the server yet.

## Checklist for Parker

Before the server (due Nov 7):
- [ ] Login works (ssh from eduroam or VPN), and you know whether you have sudo.
- [ ] `python3 --version` is 3.11 or newer (else ask the admin, or set `PYTHON=`). `git --version` works.
- [ ] Ask the admin: can your user run `docker` or `podman` (docker group)? Optional; without it the sandbox falls back to `process`.
- [ ] Ask the admin: is `loginctl enable-linger <you>` allowed (lets the queue survive logout)? Else use `run_tmux.sh`.
- [ ] The server needs GitHub access to the private repo: run `git clone https://github.com/PNiccum/Bespoke-AI-System-Factory ~/aspire` yourself and sign in (or use a deploy key). Claude cannot do this part.

On the server:
1. `git clone ... ~/aspire && cd ~/aspire && bash deploy/server/setup.sh`
2. `nano .env` and paste `ARC_API_KEY` yourself (copy from `vt_arc_key` on the laptop with scp or by hand; do not paste it into a chat or onto a command line). Leave `ANTHROPIC_API_KEY` and `OPENAI_API_KEY` empty unless you want paid runs there (they are not needed for the queue).
3. `bash deploy/server/preflight.sh` : all PASS, in particular "ARC answers HTTP 200 from this server".
4. Edit `deploy/server/queue.yaml` (`--dry-run` first), then `systemctl --user start aspire-queue` (or `bash deploy/server/run_tmux.sh`).
5. Watch: `journalctl --user -u aspire-queue -f`, `cat ~/aspire-queue/state.json`, logs and copied `summary.md` files in `~/aspire-queue/logs/`.
6. App for the professor: `systemctl --user start aspire-app`, then open `http://<server>:8501` on eduroam/VPN.
7. Tell the laptop queue thread the server is live, so jobs are split between them and not run twice.

## How keys and rules are handled

- Keys live only in `~/aspire/.env`, mode 600, written by hand. Scripts check the mode and never print values; the
  repo's secret scrubber (`factory/secrets.py`) also removes them from logs.
- Queue jobs are free ARC only. A job with `--ask` or `--redesign` is refused unless it is marked `allow_paid: true`
  in `queue.yaml`; add that only with Parker's own go-ahead (his standing rule: re-design audits under $2 each, total real
  paid spend at most $25, $10 held for Dec 3).
- Private jobs (`private: true`, the Lazar lab list): the runner sets `ASPIRE_DATA_PRIVATE=1`, removes the Claude and
  OpenAI keys from that job's environment, never allows `allow_paid`, and skips the job until the data file exists.
  **The lab list must not be copied to the server until Parker says so** (Claude has not and cannot do this).
  When he does: `scp` it to `~/aspire-private/` (mode 700 folder), then uncomment the `mcf7-research` job. Only counts
  and stage outcomes leave the server.
- Control: `touch ~/aspire-queue/STOP` ends the queue after the current job; `touch ~/aspire-queue/PAUSE` holds it.
- If ARC stops answering the runner waits (up to 6 hours) before the next job and stops with exit 2 instead of
  recording a pile of gateway failures. Finished jobs are skipped after a restart or reboot.

## What differs from the laptop

- No VPN sign-in every 24 h, no Windows sleep, no Dropbox, no UTF-8 crash. Docker may be available.
- Results are plain files on the server; Parker (or a Remote Control session on the laptop with ssh) copies the
  `summary.md` files into the project folder for the queue thread.
- The cloud container cannot reach the server, so Claude cannot start or watch runs there directly.
