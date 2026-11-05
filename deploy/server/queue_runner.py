"""Run a list of ASPIRE jobs one after another, unattended, on the server.

    python deploy/server/queue_runner.py deploy/server/queue.yaml [--state DIR] [--dry-run]

What it does for each job in the file, in order:
- skips it if the state file already says it finished (so a restart after a reboot carries on where it stopped);
- checks that ARC answers (GET /api/models with the key from the environment; the key is never printed) and, if not,
  waits and tries again instead of burning a run on a gateway error;
- runs the command with its output going to a log file, with a time limit;
- records ok / failed / timeout and the exit code in ``state.json``, and copies the run's summary.md next to the log.

Rules it enforces in code, not by trust:
- A job marked ``private: true`` runs with ASPIRE_DATA_PRIVATE=1 and with ANTHROPIC_API_KEY / OPENAI_API_KEY removed
  from its environment, and is skipped unless every ``needs_files`` path exists. The lab protein list is such a job.
- No job gets ``--ask`` or ``--redesign`` unless the queue file says so on that job (``allow_paid: true``); a
  paid-capable job is also run with the keys only if it is not private.
- ``touch <state>/STOP`` ends the queue after the current job; ``touch <state>/PAUSE`` holds it before the next one.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

import yaml

ARC_MODELS = "https://llm-api.arc.vt.edu/api/models"
PAID_KEYS = ("ANTHROPIC_API_KEY", "OPENAI_API_KEY")
PAID_FLAGS = ("--ask", "--redesign")


def now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_env_file(path: Path) -> dict[str, str]:
    """KEY=value lines (no export, no quoting tricks). Values are never printed."""
    env: dict[str, str] = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip().strip("'\"")
    return env


def arc_ok(key: str, url: str = ARC_MODELS, timeout: float = 20) -> tuple[bool, str]:
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}", "x-api-key": key})  # noqa: S310
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310 (fixed https URL)
            return r.status == 200, f"HTTP {r.status}"
    except urllib.error.HTTPError as e:
        return False, f"HTTP {e.code}"
    except (urllib.error.URLError, OSError, TimeoutError) as e:
        return False, type(e).__name__


def job_env(job: dict, base: dict[str, str]) -> dict[str, str]:
    env = dict(base)
    env.update({str(k): str(v) for k, v in (job.get("env") or {}).items()})
    if job.get("private"):
        env["ASPIRE_DATA_PRIVATE"] = "1"
        for k in PAID_KEYS:
            env.pop(k, None)
    return env


def job_problem(job: dict) -> str | None:
    """Why this job must not run now, or None."""
    cmd = job.get("cmd", "")
    if not cmd:
        return "no cmd"
    if not job.get("allow_paid") and any(f in shlex.split(cmd) for f in PAID_FLAGS):
        return "cmd has --ask/--redesign but the job is not marked allow_paid"
    if job.get("private") and job.get("allow_paid"):
        return "a private job can never be allow_paid"
    missing = [p for p in job.get("needs_files", []) if not Path(os.path.expanduser(p)).exists()]
    if missing:
        return f"waiting for file(s): {', '.join(missing)}"
    return None


def run_queue(queue_file: Path, state_dir: Path, *, dry_run: bool = False, arc_retry_s: int = 300,
              arc_max_wait_s: int = 6 * 3600, check_arc=arc_ok) -> int:
    spec = yaml.safe_load(queue_file.read_text(encoding="utf-8")) or {}
    jobs = spec.get("jobs", [])
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "logs").mkdir(exist_ok=True)
    state_path = state_dir / "state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    base = {**os.environ, **load_env_file(Path(spec.get("env_file", ".env")).expanduser())}
    root = Path(spec.get("workdir", ".")).expanduser()

    def save() -> None:
        state_path.write_text(json.dumps(state, indent=1), encoding="utf-8")

    for job in jobs:
        name = job["name"]
        if (state_dir / "STOP").exists():
            print(f"{now()} STOP file found; leaving the rest of the queue", flush=True)
            break
        if state.get(name, {}).get("status") in ("ok", "failed", "timeout") and not job.get("rerun"):
            continue
        while (state_dir / "PAUSE").exists():
            time.sleep(30)
        problem = job_problem(job)
        if problem:
            print(f"{now()} skip {name}: {problem}", flush=True)
            state[name] = {"status": "skipped", "why": problem, "at": now()}
            save()
            continue
        if dry_run:
            print(f"would run {name}: {job['cmd']}  (private={bool(job.get('private'))})")
            continue
        waited = 0
        while True:
            ok, why = check_arc(base.get("ARC_API_KEY", ""))
            if ok:
                break
            if waited >= arc_max_wait_s:
                state[name] = {"status": "blocked", "why": f"ARC unreachable ({why}) for {waited}s", "at": now()}
                save()
                print(f"{now()} {name}: ARC unreachable ({why}); queue stops", flush=True)
                return 2
            print(f"{now()} ARC not answering ({why}); waiting {arc_retry_s}s", flush=True)
            time.sleep(arc_retry_s)
            waited += arc_retry_s
        log = state_dir / "logs" / f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{name}.log"
        state[name] = {"status": "running", "started": now(), "log": str(log)}
        save()
        print(f"{now()} start {name}", flush=True)
        t0 = time.time()
        try:
            with log.open("w", encoding="utf-8") as f:
                p = subprocess.run(shlex.split(job["cmd"]), cwd=root, env=job_env(job, base), stdout=f,  # noqa: S603
                                   stderr=subprocess.STDOUT, timeout=job.get("timeout_hours", 8) * 3600)
            status, code = ("ok" if p.returncode == 0 else "failed"), p.returncode
        except subprocess.TimeoutExpired:
            status, code = "timeout", None
        state[name] = {**state[name], "status": status, "exit": code, "minutes": round((time.time() - t0) / 60, 1),
                       "finished": now()}
        out_glob = job.get("results_glob", "bench/results/*/summary.md")
        newest = sorted(root.glob(out_glob), key=lambda p: p.stat().st_mtime)
        if newest:  # the summary is counts and outcomes only; private jobs keep it local (see ``keep_local``)
            dest = state_dir / "logs" / f"{log.stem}.summary.md"
            shutil.copy(newest[-1], dest)
            state[name]["summary"] = str(dest)
        save()
        print(f"{now()} done {name}: {status}", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("queue_file", type=Path)
    ap.add_argument("--state", type=Path, default=Path("~/aspire-queue").expanduser())
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    return run_queue(a.queue_file, a.state, dry_run=a.dry_run)


if __name__ == "__main__":
    sys.exit(main())
