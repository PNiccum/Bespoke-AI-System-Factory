"""Package a finished run as a bundle that reruns without the factory.

A bug-fix bundle holds the patch, the workflow that produced it (steps, which needed a model and why, which agent
won each step), the evidence (test report, review), the run's provenance and cost, and ``rerun.py``: a standard
library script that applies the patch to a fresh copy of the repository and runs the tests again. Reproducing the
result needs Python, git and pytest, and no model.
"""

from __future__ import annotations

import json
import shutil
import zipfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from factory.orchestrator import RunResult, Task

RERUN = '''"""Re-check this fix without the factory: python rerun.py /path/to/repo

Copies the repository to a temporary directory, applies fix.patch, and runs the target tests and then the whole
suite with pytest. Exits 0 when everything passes.
"""

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).parent


def main(repo):
    targets = json.loads((HERE / "workflow.json").read_text())["targets"]
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp) / "repo"
        shutil.copytree(repo, work, ignore=shutil.ignore_patterns(".git", ".venv*", "__pycache__", ".env"))
        patch = (HERE / "fix.patch").read_text()
        if subprocess.run(["git", "apply", "-"], cwd=work, input=patch, text=True).returncode != 0:
            subprocess.run(["git", "apply", "--recount", "-"], cwd=work, input=patch, text=True, check=True)
        for label, args in (("target tests", targets), ("whole suite", [])):
            if label == "target tests" and not targets:
                continue
            code = subprocess.run([sys.executable, "-m", "pytest", "-q", *args], cwd=work).returncode
            print(f"{label}: {'PASS' if code == 0 else 'FAIL'}")
            if code != 0:
                return code
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "."))
'''


def export_bundle(result: RunResult, task: Task, run_dir: Path, out_dir: Path) -> Path:
    """Write the bundle to ``out_dir/<run name>/`` and zip it next to that folder; returns the zip's path."""
    if result.patch is None:
        raise ValueError(f"nothing to export: the run ended {result.status} ({result.stop_reason})")
    bundle = out_dir / f"bundle-{run_dir.name}"
    bundle.mkdir(parents=True, exist_ok=True)
    targets = task.failing_tests or next((e.data.get("failing", []) for e in result.events
                                          if e.step == "baseline" and e.status == "done"), [])
    models = sorted({e.data["model"] for e in result.events if "model" in e.data})
    awards = {e.step: e.agent for e in result.events if e.status in ("started", "waiting") and "parts" in e.data}

    (bundle / "fix.patch").write_text(result.patch)
    (bundle / "rerun.py").write_text(RERUN)
    workflow = {
        "task": task.model_dump(),
        "status": result.status,
        "attempts": result.attempts,
        "targets": targets,
        "steps": [{**s.model_dump(), "agent": awards.get(s.id)} for s in result.plan.steps],
        "models_used": models,
        "exported_at": datetime.now(timezone.utc).isoformat(),
    }
    (bundle / "workflow.json").write_text(json.dumps(workflow, indent=2))
    (bundle / "evidence.json").write_text(json.dumps({
        "test_report": result.check.model_dump() if result.check else None,
        "review": result.review.model_dump() if result.review else None,
        "events": [asdict(e) for e in result.events],
    }, indent=2, default=str))
    for name in ("provenance.jsonl", "llm_calls.jsonl", "cost_summary.json", "run_info.json"):
        if (run_dir / name).exists():
            shutil.copy2(run_dir / name, bundle / name)
    (bundle / "README.md").write_text(readme(task, result, targets, models))

    zip_path = bundle.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(bundle.iterdir()):
            z.write(f, f"{bundle.name}/{f.name}")
    return zip_path


def readme(task: Task, result: RunResult, targets: list[str], models: list[str]) -> str:
    steps = "\n".join(f"| {s.id} | {s.capability} | {'yes' if s.needs_llm else 'no'} | {s.reason} |"
                      for s in result.plan.steps)
    return f"""# Fix: {task.description.splitlines()[0][:80]}

Status: **{result.status}** after {result.attempts} attempt(s). Models used: {', '.join(models) or 'none'}.

## Check it yourself

```bash
python rerun.py /path/to/the/repo
```

This applies `fix.patch` to a copy of the repository and runs the tests that failed before the fix
({', '.join(targets) or 'none were failing'}), then the whole suite. No model is called.

## How it was made

| Step | Capability | Model? | Why |
|---|---|---|---|
{steps}

| File | What it is |
|---|---|
| `fix.patch` | The change, as a unified diff |
| `workflow.json` | The task, the plan, which agent did each step, the target tests |
| `evidence.json` | Test results with the patch, the review, and every step's event |
| `provenance.jsonl` | What each step read and produced, by content hash |
| `llm_calls.jsonl` | Every model call: model, tokens, cost, request hash (no prompt text) |
| `cost_summary.json` | Real and shadow cost of the run |
"""
