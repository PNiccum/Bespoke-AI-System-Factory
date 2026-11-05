"""Bundles for the test_gen and pr_review task types.

Each bundle carries the work (the test diff, or the review), the workflow and evidence, the run's provenance and
cost, and a ``rerun.py`` that recomputes every plain-code fact without the factory or a model: the tests pass and
catch the same planted bugs, or the diff gives the same rule findings and test delta. The plain-code agents
(``mutate.py``, ``diff_inspect.py``) use only the standard library, so the bundle ships a copy of them.
"""

from __future__ import annotations

import json
import shutil
import zipfile
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from factory.orchestrator import RunResult, Task
from factory.secrets import guard_bundle

TOOLS = Path(__file__).resolve().parents[1] / "tools"

RERUN_TESTS = '''"""Re-check these tests without the factory: python rerun.py /path/to/repo

Applies tests.patch to a copy of the repository, runs the new tests, then plants the same kinds of small bugs in the
target code and counts how many the tests catch. Exits 0 when the tests pass and catch as many as recorded.
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from mutate import ScoreIn, score_tests  # noqa: E402


def main(repo):
    wf = json.loads((HERE / "workflow.json").read_text(encoding="utf-8"))
    score = score_tests(ScoreIn(repo_path=repo, files=wf["targets"], tests=wf["new_tests"],
                                functions=wf["functions"], patch=(HERE / "tests.patch").read_text(encoding="utf-8")))
    if score.error:
        print(f"new tests: FAIL ({score.error})")
        return 1
    print("new tests: PASS")
    want = wf["mutation"]
    ok = score.killed >= want["killed"]
    print(f"planted bugs caught: {score.killed} of {score.total} (recorded {want['killed']} of {want['total']}): "
          f"{'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "."))
'''

RERUN_REVIEW = '''"""Re-check this review's facts without the factory: python rerun.py /path/to/repo

Re-runs the rule checks on change.patch and the test suite without and with it, and compares both with what the
review recorded. The model's comments are judgement and are not recomputed. Exits 0 when the facts match.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from diff_inspect import inspect  # noqa: E402


def clean_env():
    """This shell's environment without keys, tokens, secrets or passwords: tests never need a credential."""
    return {k: v for k, v in os.environ.items()
            if not any(m in k.upper() for m in ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL"))
            and not k.upper().startswith("GIT_CONFIG_")}


def broken(repo, patch):
    """Test names that fail or error, or None when the patch does not apply."""
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp) / "repo"
        shutil.copytree(repo, work, ignore=shutil.ignore_patterns(".git", ".venv*", "__pycache__", ".env"))
        if patch and all(subprocess.run(["git", "apply", *x, "-"], cwd=work, input=patch, text=True, encoding="utf-8", errors="replace",
                                        capture_output=True).returncode for x in ([], ["--recount"])):
            return None
        junit = Path(tmp) / "junit.xml"
        subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", f"--junitxml={junit}"],
                       cwd=work, capture_output=True, env=clean_env())
        if not junit.exists():
            return set()
        return {c.get("classname", "") + "::" + c.get("name", "") for c in ET.parse(junit).getroot().iter("testcase")
                if c.find("failure") is not None or c.find("error") is not None}


def main(repo):
    patch = (HERE / "change.patch").read_text(encoding="utf-8")
    recorded = json.loads((HERE / "review.json").read_text(encoding="utf-8"))
    rules = sorted((f["rule"], f["path"], f["line"]) for f in inspect(patch, repo)["findings"])
    want = sorted((f["rule"], f["path"], f["line"]) for f in recorded["facts"]["findings"])
    print(f"rule findings: {len(rules)} (recorded {len(want)}): {'PASS' if rules == want else 'FAIL'}")
    before, after = broken(repo, None), broken(repo, patch)
    t = recorded["tests"]
    applied = after is not None
    same = applied == t["applied"] and (not applied or (len(after - before) == len(t["newly_failing"])
                                                        and len(before - after) == len(t["newly_passing"])))
    print(f"test delta: {'does not apply' if not applied else f'fixes {len(before - after)}, breaks {len(after - before)}'}"
          f": {'PASS' if same else 'FAIL'}")
    return 0 if rules == want and same else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "."))
'''


def export_workflow_bundle(result: RunResult, task: Task, run_dir: Path, out_dir: Path) -> Path:
    """Write a test_gen or pr_review bundle to ``out_dir/<run name>/``, zip it, and return the zip's path."""
    if result.patch is None:
        raise ValueError(f"nothing to export: the run ended {result.status} ({result.stop_reason})")
    bundle = out_dir / f"bundle-{run_dir.name}"
    bundle.mkdir(parents=True, exist_ok=True)
    models = sorted({e.data["model"] for e in result.events if "model" in e.data})
    awards = {e.step: e.agent for e in result.events if e.status in ("started", "waiting") and "parts" in e.data}
    workflow = {
        "task": task.model_dump(exclude={"patch"}),
        "status": result.status,
        "attempts": result.attempts,
        "steps": [{**s.model_dump(), "agent": awards.get(s.id)} for s in result.plan.steps],
        "models_used": models,
        "exported_at": datetime.now(UTC).isoformat(),
    }
    if task.task_type == "test_gen":
        (bundle / "tests.patch").write_text(result.patch, encoding="utf-8")
        (bundle / "rerun.py").write_text(RERUN_TESTS, encoding="utf-8")
        shutil.copy2(TOOLS / "mutate.py", bundle / "mutate.py")
        workflow |= {k: result.report[k] for k in ("targets", "functions", "new_tests", "mutation")}
        readme = tests_readme(task, result, models)
    elif task.task_type == "pr_review":
        (bundle / "change.patch").write_text(result.patch, encoding="utf-8")
        (bundle / "rerun.py").write_text(RERUN_REVIEW, encoding="utf-8")
        shutil.copy2(TOOLS / "diff_inspect.py", bundle / "diff_inspect.py")
        (bundle / "review.json").write_text(json.dumps(result.report, indent=2), encoding="utf-8")
        (bundle / "review.md").write_text(review_markdown(task, result), encoding="utf-8")
        readme = review_readme(task, result, models)
    else:
        raise ValueError(f"no bundle format for task type {task.task_type!r}")
    (bundle / "workflow.json").write_text(json.dumps(workflow, indent=2), encoding="utf-8")
    (bundle / "evidence.json").write_text(json.dumps({
        "test_report": result.check.model_dump() if result.check else None,
        "review": result.review.model_dump() if result.review else None,
        "report": result.report,
        "events": [asdict(e) for e in result.events],
    }, indent=2, default=str), encoding="utf-8")
    for name in ("provenance.jsonl", "llm_calls.jsonl", "cost_summary.json", "run_info.json", "registry.json"):
        if (run_dir / name).exists():
            shutil.copy2(run_dir / name, bundle / name)
    (bundle / "README.md").write_text(readme + steps_table(result) + FILES, encoding="utf-8")

    guard_bundle(bundle)  # nothing that looks like a key leaves in a bundle

    zip_path = bundle.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(bundle.iterdir()):
            z.write(f, f"{bundle.name}/{f.name}")
    return zip_path


def steps_table(result: RunResult) -> str:
    rows = "\n".join(f"| {s.id} | {s.capability} | {'yes' if s.needs_llm else 'no'} | {s.reason} |"
                     for s in result.plan.steps)
    return f"\n## How it was made\n\n| Step | Capability | Model? | Why |\n|---|---|---|---|\n{rows}\n"


FILES = """
| File | What it is |
|---|---|
| `workflow.json` | The task, the plan, which agent did each step, the facts rerun.py checks |
| `evidence.json` | Test results, the review, and every step's event |
| `provenance.jsonl` | What each step read and produced, by content hash |
| `llm_calls.jsonl` | Every model call: model, tokens, cost, request hash (no prompt text) |
| `cost_summary.json` | Real and shadow cost of the run |
"""


def tests_readme(task: Task, result: RunResult, models: list[str]) -> str:
    m = result.report.get("mutation", {})
    return f"""# Tests: {task.description.splitlines()[0][:80]}

Status: **{result.status}** after {result.attempts} attempt(s). Models used: {', '.join(models) or 'none'}.
The new tests ({len(result.report.get('new_tests', []))}) caught {m.get('killed', 0)} of {m.get('total', 0)} small
bugs planted in {', '.join(result.report.get('targets', []))}.

## Check it yourself

```bash
python rerun.py /path/to/the/repo
```

This applies `tests.patch` to a copy of the repository, runs the new tests, and plants the bugs again with
`mutate.py` to count how many the tests catch. No model is called.
"""


def review_readme(task: Task, result: RunResult, models: list[str]) -> str:
    r = result.report["review"]
    return f"""# Review: {task.description.splitlines()[0][:80]}

Verdict: **{r['verdict']}**, {len(r['comments'])} comments. Models used: {', '.join(models) or 'none'}.
The review itself is in `review.md` (ready to paste into the pull request) and `review.json`.

## Check it yourself

```bash
python rerun.py /path/to/the/repo
```

This re-runs the rule checks (`diff_inspect.py`) on `change.patch` and the test suite without and with the change,
and compares them with what the review recorded. The model's comments are judgement and are not recomputed.
"""


def review_markdown(task: Task, result: RunResult) -> str:
    rep = result.report
    r, t = rep["review"], rep["tests"]
    lines = [f"**Verdict: {r['verdict']}**", "", r["summary"], ""]
    if rep["overridden"]:
        lines += [f"> {rep['overridden']}", ""]
    lines += ["### Tests", "",
              "The patch does not apply." if not t["applied"] else
              f"{t['passed']} passing, {t['failed']} failing. Fixes: {', '.join(t['newly_passing']) or 'none'}. "
              f"Breaks: {', '.join(t['newly_failing']) or 'none'}.", ""]
    if rep["facts"]["findings"]:
        lines += ["### Rule checks", ""] + [f"- `{f['path']}:{f['line']}` [{f['severity']}] {f['message']}"
                                             for f in rep["facts"]["findings"]] + [""]
    if r["comments"]:
        lines += ["### Comments", ""]
        for c in r["comments"]:
            lines.append(f"- `{c['file']}:{c['line']}` **{c['severity']}**: {c['problem']}"
                         + (f" Suggestion: {c['suggestion']}" if c["suggestion"] else ""))
    return "\n".join(lines) + "\n"
