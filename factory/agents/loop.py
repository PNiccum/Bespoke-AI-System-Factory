"""The agent loop: model turn, tool calls, results, repeat, under caps.

Every tool call is written to ``runs/<run-id>/agent_log.jsonl`` with the
model's stated reason (its text in that turn), the input, and the result, so
a run can be read in plain language and replayed without the model
(``replay``). Hitting a cap stops the run with a report, never with a silent
partial result. Unexpected exceptions inside a tool are not caught here: a
crash in our own code should stop the run, not be explained away by the agent.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from factory.agents.tools import Toolbox
from factory.config import AgentCaps
from factory.llm import LLM


@dataclass
class AgentResult:
    status: str  # done | cap:steps | cap:tokens | cap:seconds | cap:usd (real dollars)
    final_text: str
    steps: int
    tool_calls: int
    tokens: int
    priced_usd: float  # real + shadow, reported
    seconds: float
    log_path: Path
    messages: list[dict] = field(default_factory=list)
    real_usd: float = 0.0

    def report(self) -> str:
        head = "finished" if self.status == "done" else f"stopped at a cap ({self.status.split(':')[1]})"
        return (f"Agent {head} after {self.steps} turns and {self.tool_calls} tool calls; "
                f"{self.tokens:,} tokens, ${self.real_usd:.2f} real (${self.priced_usd:.2f} incl. shadow), "
                f"{self.seconds:.0f} s. "
                f"Decision record: {self.log_path}")


def run_agent(llm: LLM, role: str, system: str, task: str, toolbox: Toolbox, caps: AgentCaps | None = None, *,
              tag: str = "agent", data: str = "private") -> AgentResult:
    caps = caps or llm.cfg.agent_caps
    log_path = llm.run_dir / "agent_log.jsonl"
    messages: list[dict] = [{"role": "user", "content": task}]
    tokens, priced, real, n_calls, final = 0, 0.0, 0.0, 0, ""
    t0 = time.monotonic()

    def log(**rec):
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"at": datetime.now(UTC).isoformat(), "tag": tag, **rec}, default=str) + "\n")

    def result(status, steps):
        log(event="stop", status=status, steps=steps, tool_calls=n_calls, tokens=tokens, real_usd=round(real, 4),
            priced_usd=round(priced, 4))
        return AgentResult(status, final, steps, n_calls, tokens, priced, time.monotonic() - t0, log_path, messages,
                           real)

    log(event="start", role=role, model=llm.cfg.roles[role].model, tools=sorted(toolbox.tools), task=task[:2000])
    for step in range(1, caps.max_steps + 1):
        if tokens >= caps.max_tokens:
            return result("cap:tokens", step - 1)
        if real >= caps.max_real_usd:
            return result("cap:usd", step - 1)
        if time.monotonic() - t0 >= caps.max_seconds:
            return result("cap:seconds", step - 1)

        c = llm.converse(role, list(messages), system, tools=toolbox.schemas(), tag=f"{tag}:turn{step}", data=data)
        tokens += c.usage.total
        priced += c.priced_usd
        real += c.real_usd
        final = c.text
        content = ([{"type": "text", "text": c.text}] if c.text else []) + [
            {"type": "tool_use", "id": tc["id"], "name": tc["name"], "input": tc["input"]} for tc in c.tool_calls]
        messages.append({"role": "assistant", "content": content or [{"type": "text", "text": ""}]})
        if not c.tool_calls:
            log(event="answer", step=step, text=c.text[:4000])
            return result("done", step)

        results = []
        for tc in c.tool_calls:
            n_calls += 1
            r = toolbox.call(tc["name"], tc["input"])
            log(event="tool", step=step, reason=c.text[:2000], tool=tc["name"], input=tc["input"], ok=r.ok,
                result=r.content[:2000])
            results.append({"type": "tool_result", "tool_use_id": tc["id"], "content": r.content,
                            **({} if r.ok else {"is_error": True})})
        messages.append({"role": "user", "content": results})
        if caps.compact_every and step % caps.compact_every == 0:
            saved = compact(messages, caps.keep_recent)
            log(event="compact", step=step, chars_saved=saved)
    return result("cap:steps", caps.max_steps)


def compact(messages: list[dict], keep_recent: int) -> int:
    """Replace tool results older than the last ``keep_recent`` turns with stubs; returns characters saved.

    A stub names the tool and its input and keeps any references it held (``"ref"`` fields), so the agent can
    read that item again.
    """
    calls = {b["id"]: (b["name"], b.get("input", {})) for m in messages if m["role"] == "assistant"
             for b in (m["content"] if isinstance(m["content"], list) else []) if b.get("type") == "tool_use"}
    result_turns = [i for i, m in enumerate(messages) if m["role"] == "user" and isinstance(m["content"], list)]
    saved = 0
    for i in result_turns[:-keep_recent] if keep_recent else result_turns:
        new = []
        for b in messages[i]["content"]:
            text = b.get("content")
            if b.get("type") == "tool_result" and isinstance(text, str) and not text.startswith("[earlier result"):
                name, inp = calls.get(b["tool_use_id"], ("tool", {}))
                ids = sorted(set(re.findall(r'"ref": "([^"]+)"', text)))
                stub = (f"[earlier result of {name}({json.dumps(inp)[:120]}), {len(text)} characters, removed to "
                        f"save context" + (f"; references: {', '.join(ids[:20])}" if ids else "") + "]")
                saved += len(text) - len(stub)
                b = {**b, "content": stub}
            new.append(b)
        messages[i] = {**messages[i], "content": new}
    return saved


def replay(log_path: Path, toolbox: Toolbox, tag: str | None = None) -> list[tuple[str, bool, bool]]:
    """Re-run every logged tool call without the model. Returns (tool, ok now, same result as logged) per call,
    so a build's tool steps can be reproduced and any drift in a tool's output found."""
    out = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        if rec.get("event") != "tool" or (tag and rec.get("tag") != tag):
            continue
        r = toolbox.call(rec["tool"], rec["input"])
        out.append((rec["tool"], r.ok, r.content[:2000] == rec["result"]))
    return out
