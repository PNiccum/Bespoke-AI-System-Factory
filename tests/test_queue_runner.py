"""deploy/server/queue_runner.py: the rules it enforces in code (no network, no keys)."""
import importlib.util
import json
import sys
from pathlib import Path

spec = importlib.util.spec_from_file_location("queue_runner", Path(__file__).parents[1] / "deploy/server/queue_runner.py")
qr = importlib.util.module_from_spec(spec)
sys.modules["queue_runner"] = qr
spec.loader.exec_module(qr)


def test_private_job_loses_paid_keys_and_is_flagged():
    env = qr.job_env({"private": True}, {"ANTHROPIC_API_KEY": "a", "OPENAI_API_KEY": "b", "ARC_API_KEY": "c"})
    assert "ANTHROPIC_API_KEY" not in env and "OPENAI_API_KEY" not in env
    assert env["ARC_API_KEY"] == "c" and env["ASPIRE_DATA_PRIVATE"] == "1"


def test_paid_flags_refused_unless_allowed_and_private_never_allowed():
    assert qr.job_problem({"cmd": "python -m x --ask"})
    assert qr.job_problem({"cmd": "python -m x --ask", "allow_paid": True}) is None
    assert qr.job_problem({"cmd": "x", "private": True, "allow_paid": True})
    assert qr.job_problem({"cmd": "x", "needs_files": ["/nonexistent/lab.csv"]})


def test_queue_runs_skips_done_and_writes_state(tmp_path):
    q = tmp_path / "q.yaml"
    q.write_text(f"workdir: {tmp_path}\nenv_file: {tmp_path}/.env\njobs:\n"
                 f"  - {{name: a, cmd: 'python -c pass'}}\n  - {{name: b, cmd: 'python -c \"raise SystemExit(3)\"'}}\n")
    st = tmp_path / "state"
    assert qr.run_queue(q, st, check_arc=lambda k: (True, "ok")) == 0
    state = json.loads((st / "state.json").read_text())
    assert state["a"]["status"] == "ok" and state["b"]["status"] == "failed" and state["b"]["exit"] == 3
    assert qr.run_queue(q, st, check_arc=lambda k: (False, "down"), arc_max_wait_s=0) == 0  # all done: ARC not needed


def test_arc_down_stops_queue(tmp_path):
    q = tmp_path / "q.yaml"
    q.write_text(f"workdir: {tmp_path}\nenv_file: {tmp_path}/.env\njobs:\n  - {{name: a, cmd: 'python -c pass'}}\n")
    assert qr.run_queue(q, tmp_path / "s", check_arc=lambda k: (False, "HTTP 403"), arc_max_wait_s=0) == 2
