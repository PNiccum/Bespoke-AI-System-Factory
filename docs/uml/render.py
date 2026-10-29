"""Draw SIMPLE's UML class and sequence diagrams as SVG: python docs/uml/render.py

The diagrams are data in this file (classes with their main fields and methods, relationships, and the messages of
one bug-fix run), so they are reviewed and updated like code. Open the SVGs in any browser; print to PDF from there,
or run docs/uml/to_pdf.mjs (Playwright) to write the PNGs and the combined PDF.
"""

from __future__ import annotations

import math
from html import escape
from pathlib import Path

HERE = Path(__file__).parent
FONT = "DejaVu Sans Mono, Consolas, Menlo, monospace"
CHAR = 7.2  # width of one character at 12px in the monospace font
LINE = 17
INK, MUTED, PAPER = "#1f2933", "#52606d", "#ffffff"

# Package colors: (fill, border)
PKG = {
    "ui": ("#eef6ff", "#4a7fc1"),
    "core": ("#fff7e8", "#c98a1b"),
    "registry": ("#eefaf1", "#3b9a57"),
    "llm": ("#f6efff", "#8a5cc7"),
    "tools": ("#ffeff0", "#c4505a"),
    "records": ("#f1f3f5", "#6b7785"),
}

# name: (package, stereotype, x, y, fields, methods)
CLASSES = {
    "StreamlitApp": ("ui", "boundary · app.py", 60, 100, ["session: RunSession"],
                     ["task_form()", "live_view()  # every 1 s", "approval_card()"]),
    "CLI": ("ui", "boundary · __main__.py", 60, 330, [], ["main(argv)", "ask_on_terminal(req): bool"]),
    "RunSession": ("ui", "", 60, 500, ["events: list[Event]", "pending: PendingApproval", "result: RunResult",
                                         "bundle: Path", "answer_timeout_s = 900"],
                   ["start(task)", "answer(approved)", "running(): bool"]),
    "PendingApproval": ("ui", "", 60, 790, ["request: SpendRequest", "decided: Event", "approved: bool"], []),

    "Orchestrator": ("core", "control", 500, 100, ["llm: LLM", "registry: Registry", "mcp: McpTools | None",
                                                  "provenance: ProvenanceLog", "max_attempts = 3"],
                     ["run(task): RunResult", "award(task, step): Bid", "tool(agent): Callable",
                      "run_tests(bid, task, patch, tests)", "coder_prompt(...)", "stop(result, step, reason)"]),
    "Task": ("core", "", 500, 440, ["description: str", "repo_path: str", "task_type = 'bug_fix'",
                                    "domain = 'software'", "failing_tests: list[str]"], []),
    "Plan": ("core", "", 500, 620, ["task_type: str", "steps: list[Step]"], ["plan_for(task)  # template"]),
    "Step": ("core", "", 500, 750, ["id: str", "capability: str", "needs_llm: bool", "reason: str"], []),
    "RunResult": ("core", "", 820, 440, ["status: fixed|unproven|failed|stopped", "patch: str", "attempts: int",
                                          "check: TestReport", "review: Review", "stop_reason: str"], []),
    "Event": ("core", "", 820, 650, ["step, agent, status", "detail: str", "data: dict"], []),
    "Review": ("core", "", 820, 800, ["verdict: accept|reject", "defects: list[Defect]"], []),

    "Registry": ("registry", "contract net", 1220, 100, ["agents: dict[str, AgentSpec]", "scores: ScoreBook"],
                 ["load(cfg)", "bids(announcement): list[Bid]", "award(announcement): Bid"]),
    "AgentSpec": ("registry", "registry/agents/*.yaml", 1220, 300,
                  ["name, description", "kind: llm | tool", "role: str  # llm agents", "entrypoint: str  # tool agents",
                   "capabilities: list[str]"], ["function(): Callable"]),
    "Announcement": ("registry", "", 1220, 650, ["capability: str", "domain: str", "task_type: str"], []),
    "Bid": ("registry", "", 1220, 500, ["agent: AgentSpec", "score: float", "parts: dict  # domain, record, ..."], []),
    "ScoreBook": ("registry", "scores.json", 1220, 800, [], ["record(agent, type, passed)", "pass_rate(agent, type)"]),

    "LLM": ("llm", "factory/llm", 1620, 100, ["cfg: FactoryConfig", "gate: SpendGate", "spend: SpendLedger",
                                             "cache: ResponseCache", "providers: arc, claude, openai"],
            ["complete(role, prompt, escalate=False)", "role_config(role, escalate)", "write_summary()"]),
    "SpendGate": ("llm", "human in the loop", 1620, 360, ["approver: SpendRequest -> bool", "approve_above_usd = 1.00"],
                  ["check(request)  # raises SpendDenied"]),
    "SpendRequest": ("llm", "", 1620, 510, ["role, model, provider", "estimate_usd: float", "run_spent_usd: float",
                                             "reason: str"], []),
    "SpendLedger": ("llm", "spend.json", 1620, 680, ["spent: dict[provider, usd]", "credits: claude 20, openai 15",
                                                      "reserve_usd = 10"], ["check(provider)", "add(provider, usd)"]),
    "FactoryConfig": ("llm", "configs/factory.yaml", 1620, 870, ["roles: dict[str, RoleConfig]", "budget, credits",
                                                                 "tier: economy | quality",
                                                                 "tool_transport: direct | mcp"],
                      ["separation_errors()  # no self-review"]),

    "McpTools": ("tools", "MCP client", 2040, 100, ["server: stdio, keys scrubbed"],
                 ["call(name, arguments)", "mcp_function(fn, tools)"]),
    "factory-tools": ("tools", "MCP server", 2040, 270, ["roots: allowed folders"],
                      ["locate(LocateIn): Locations", "read_file(repo, path)", "run_tests(RunTestsIn): TestReport",
                       "prove_fix(repo, patch, tests)"]),
    "repo": ("tools", "module · code_navigator", 2040, 470, [], ["locate(LocateIn): Locations"]),
    "sandbox": ("tools", "module · test_runner", 2040, 590, [], ["run_tests(RunTestsIn): TestReport",
                                                                   "prove_fix(...): FixProof", "scrubbed_env()"]),
    "TestReport": ("tools", "", 2040, 760, ["patch_applied: bool", "passed, failed, errors", "timed_out: bool",
                                             "output_tail: str"], []),

    "ProvenanceLog": ("records", "provenance.jsonl", 500, 1000, [], ["record(event)", "add_source(source)"]),
    "ProvenanceEvent": ("records", "", 820, 1000, ["kind: source|model_call|tool_call|approval", "actor: str",
                                                   "inputs, outputs: sha256"], []),
    "export": ("records", "module · export.py", 60, 1000, [], ["export_bundle(result, task, run_dir, out)",
                                                              "  -> fix.patch, rerun.py, workflow.json,",
                                                              "     evidence.json, provenance, cost"]),
}

# (from, to, kind, label[, route]). kind: assoc (arrow), dep (dashed arrow), comp (filled diamond at "from"),
# agg (hollow diamond at "from"). A route is the polyline's points, first and last on the two boxes' edges; edges
# without one are drawn straight between the box centers.
RELATIONS = [
    ("StreamlitApp", "RunSession", "assoc", "starts runs", [(60, 150), (35, 150), (35, 570), (60, 570)]),
    ("RunSession", "PendingApproval", "agg", "0..1"),
    ("RunSession", "Orchestrator", "assoc", "runs on a thread"),
    ("CLI", "Orchestrator", "assoc", "runs"),
    ("RunSession", "export", "dep", "bundle", [(60, 640), (45, 640), (45, 1050), (60, 1050)]),
    ("Orchestrator", "Task", "dep", "reads"),
    ("Orchestrator", "RunResult", "dep", "returns"),
    ("Orchestrator", "Plan", "dep", "plan_for", [(500, 290), (488, 290), (488, 660), (500, 660)]),
    ("Plan", "Step", "comp", "1..*"),
    ("RunResult", "Event", "comp", "*"),
    ("RunResult", "Review", "comp", "0..1", [(1060, 602), (1060, 847), (1002, 847)]),
    ("Orchestrator", "Registry", "assoc", "award per step"),
    ("Registry", "AgentSpec", "comp", "*"),
    ("Bid", "AgentSpec", "assoc", ""),
    ("Registry", "Bid", "dep", "returns", [(1453, 140), (1510, 140), (1510, 555), (1489, 555)]),
    ("Registry", "Announcement", "dep", "", [(1453, 165), (1525, 165), (1525, 705), (1390, 705)]),
    ("Registry", "ScoreBook", "comp", "1", [(1453, 190), (1540, 190), (1540, 855), (1438, 855)]),
    ("Orchestrator", "LLM", "assoc", "coder and reviewer calls", [(700, 100), (700, 74), (1700, 74), (1700, 100)]),
    ("LLM", "SpendGate", "comp", "1"),
    ("SpendGate", "SpendRequest", "dep", "asks about"),
    ("LLM", "SpendLedger", "comp", "1", [(1918, 200), (1950, 200), (1950, 750), (1853, 750)]),
    ("LLM", "FactoryConfig", "assoc", "", [(1918, 235), (1975, 235), (1975, 940), (1910, 940)]),
    ("Orchestrator", "McpTools", "assoc", "tool agents (tool_transport: mcp)",
     [(735, 100), (735, 58), (2100, 58), (2100, 100)]),
    ("McpTools", "factory-tools", "dep", "stdio, JSON-RPC"),
    ("factory-tools", "repo", "dep", ""),
    ("factory-tools", "sandbox", "dep", "", [(2302, 360), (2322, 360), (2322, 650), (2302, 650)]),
    ("sandbox", "TestReport", "dep", ""),
    ("Orchestrator", "ProvenanceLog", "assoc", "records", [(500, 325), (476, 325), (476, 1050), (500, 1050)]),
    ("ProvenanceLog", "ProvenanceEvent", "comp", "*"),
]


def box_size(name, stereo, fields, methods):
    rows = [name, f"«{stereo}»" if stereo else ""] + fields + methods
    w = max(len(r) for r in rows) * CHAR + 24
    h = (2 if stereo else 1) * LINE + 10 + max(1, len(fields)) * LINE + 8 + max(1, len(methods)) * LINE + 8
    return max(w, 170), h


def class_svg() -> str:
    geo, parts = {}, []
    for name, (pkg, stereo, x, y, fields, methods) in CLASSES.items():
        w, h = box_size(name, stereo, fields, methods)
        geo[name] = (x, y, w, h)
    width = max(x + w for x, y, w, h in geo.values()) + 40
    height = max(y + h for x, y, w, h in geo.values()) + 80

    for a, b, kind, label, *route in RELATIONS:
        parts.append(routed(route[0], kind, label) if route else edge(geo[a], geo[b], kind, label))
    for name, (pkg, stereo, x, y, fields, methods) in CLASSES.items():
        parts.append(class_box(name, pkg, stereo, fields, methods, *geo[name]))
    legend = legend_svg(40, height - 40)
    return svg(width, height, "SIMPLE: UML class diagram", parts + [legend])


def class_box(name, pkg, stereo, fields, methods, x, y, w, h):
    fill, line = PKG[pkg]
    out = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="4" fill="{fill}" stroke="{line}" stroke-width="1.5"/>']
    ty = y + LINE
    if stereo:
        out.append(text(x + w / 2, ty, f"«{stereo}»", size=11, color=MUTED, anchor="middle"))
        ty += LINE
    out.append(text(x + w / 2, ty, name, size=13, weight="bold", anchor="middle"))
    ty += 10
    out.append(f'<line x1="{x}" y1="{ty}" x2="{x + w}" y2="{ty}" stroke="{line}"/>')
    for f in fields or [""]:
        ty += LINE
        out.append(text(x + 10, ty - 3, f))
    ty += 8
    out.append(f'<line x1="{x}" y1="{ty}" x2="{x + w}" y2="{ty}" stroke="{line}"/>')
    for m in methods or [""]:
        ty += LINE
        out.append(text(x + 10, ty - 3, m))
    return "\n".join(out)


def clip(box, tx, ty):
    """Where the line from the box's center toward (tx, ty) leaves the box."""
    x, y, w, h = box
    cx, cy = x + w / 2, y + h / 2
    dx, dy = tx - cx, ty - cy
    if dx == 0 and dy == 0:
        return cx, cy
    s = min((w / 2) / abs(dx) if dx else math.inf, (h / 2) / abs(dy) if dy else math.inf)
    return cx + dx * s, cy + dy * s


def edge(a, b, kind, label):
    ac = (a[0] + a[2] / 2, a[1] + a[3] / 2)
    bc = (b[0] + b[2] / 2, b[1] + b[3] / 2)
    x1, y1 = clip(a, *bc)
    x2, y2 = clip(b, *ac)
    dash = ' stroke-dasharray="6 4"' if kind == "dep" else ""
    end = ' marker-end="url(#open)"' if kind in ("assoc", "dep") else ""
    start = {"comp": ' marker-start="url(#comp)"', "agg": ' marker-start="url(#agg)"'}.get(kind, "")
    out = [f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{MUTED}" stroke-width="1.3"'
           f'{dash}{end}{start}/>']
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        w = len(label) * 6.2 + 8
        out.append(f'<rect x="{mx - w / 2:.1f}" y="{my - 9:.1f}" width="{w:.1f}" height="15" fill="{PAPER}" '
                   f'opacity="0.9"/>')
        out.append(text(mx, my + 3, label, size=10.5, color=MUTED, anchor="middle"))
    return "\n".join(out)


def routed(points, kind, label):
    dash = ' stroke-dasharray="6 4"' if kind == "dep" else ""
    end = ' marker-end="url(#open)"' if kind in ("assoc", "dep") else ""
    start = {"comp": ' marker-start="url(#comp)"', "agg": ' marker-start="url(#agg)"'}.get(kind, "")
    d = " ".join(f"{x},{y}" for x, y in points)
    out = [f'<polyline points="{d}" fill="none" stroke="{MUTED}" stroke-width="1.3"{dash}{end}{start}/>']
    if label:
        (x1, y1), (x2, y2) = max(zip(points, points[1:]), key=lambda s: abs(s[1][0] - s[0][0]) + abs(s[1][1] - s[0][1]))
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        w = len(label) * 6.2 + 8
        turn = f' transform="rotate(-90 {mx} {my})"' if x1 == x2 else ""  # vertical lanes: text runs along the line
        out.append(f'<g{turn}><rect x="{mx - w / 2:.1f}" y="{my - 9:.1f}" width="{w:.1f}" height="15" '
                   f'fill="{PAPER}"/>' + text(mx, my + 3, label, size=10.5, color=MUTED, anchor="middle") + "</g>")
    return "\n".join(out)


def legend_svg(x, y):
    items = [(name, PKG[key]) for key, name in
             [("ui", "User interface"), ("core", "Orchestration"), ("registry", "Agent selection"),
              ("llm", "Models and spend"), ("tools", "Plain-code tools and MCP"), ("records", "Records and export")]]
    out, cx = [], x
    for label, (fill, line) in items:
        out.append(f'<rect x="{cx}" y="{y - 11}" width="14" height="14" fill="{fill}" stroke="{line}"/>')
        out.append(text(cx + 20, y, label, size=11.5))
        cx += 20 + len(label) * 7.2 + 30
    out.append(text(cx + 10, y, "◆ owns   ◇ holds   → uses   ⇢ depends on / returns", size=11.5, color=MUTED))
    return "\n".join(out)


# ----------------------------------------------------------------------------------------------------------------
# Sequence diagram: one bug-fix run from the Streamlit page
# ----------------------------------------------------------------------------------------------------------------

LIFELINES = ["Person", "Streamlit + RunSession", "Orchestrator", "Registry", "McpTools → factory-tools",
             "LLM (GLM coder)", "LLM (DeepSeek reviewer)", "SpendGate", "export"]

# (from, to, label, kind) kind: call | reply | self | note | frame-start | frame-else | frame-end
MESSAGES = [
    ("Person", "Streamlit + RunSession", "submit issue + repo", "call"),
    ("Streamlit + RunSession", "Orchestrator", "run(task)  [background thread]", "call"),
    ("Orchestrator", "Orchestrator", "plan_for(task): locate, baseline, patch, check, review", "self"),
    ("Orchestrator", "Registry", "award(announcement) × 5 steps", "call"),
    ("Registry", "Orchestrator", "Bid per step (tool agents win no-model steps)", "reply"),
    ("Orchestrator", "McpTools → factory-tools", "locate(repo, issue)", "call"),
    ("McpTools → factory-tools", "Orchestrator", "Locations (ranked files)", "reply"),
    ("Orchestrator", "McpTools → factory-tools", "run_tests(repo)  # baseline", "call"),
    ("McpTools → factory-tools", "Orchestrator", "TestReport: which tests fail now", "reply"),
    ("loop", "attempt 1..3", "frame-start"),
    ("Orchestrator", "LLM (GLM coder)", "complete(coder, issue + files + feedback)", "call"),
    ("LLM (GLM coder)", "SpendGate", "check(request)  # paid model only", "call"),
    ("SpendGate", "LLM (GLM coder)", "free ARC model: no question", "reply"),
    ("LLM (GLM coder)", "Orchestrator", "unified diff", "reply"),
    ("Orchestrator", "McpTools → factory-tools", "run_tests(repo, patch)  # check", "call"),
    ("McpTools → factory-tools", "Orchestrator", "TestReport with the patch", "reply"),
    ("Orchestrator", "LLM (DeepSeek reviewer)", "complete(reviewer, patch + results)", "call"),
    ("LLM (DeepSeek reviewer)", "Orchestrator", "Review: accept | reject + defects", "reply"),
    ("alt", "reject or tests still fail: feedback to the coder, next attempt", "frame-start"),
    ("Orchestrator", "Orchestrator", "feedback = defects or test output", "self"),
    ("else", "attempt 3 escalates the coder to the paid model", "frame-else"),
    ("LLM (GLM coder)", "SpendGate", "check(gpt-6.1-sol, up to $0.16)", "call"),
    ("SpendGate", "Streamlit + RunSession", "approver(request)  [run thread waits]", "call"),
    ("Streamlit + RunSession", "Person", "approval card", "call"),
    ("Person", "Streamlit + RunSession", "approve / decline (900 s timeout = decline)", "reply"),
    ("Streamlit + RunSession", "SpendGate", "True / False", "reply"),
    ("frame-end",),
    ("frame-end",),
    ("Orchestrator", "Streamlit + RunSession", "RunResult (fixed | stopped)", "reply"),
    ("Streamlit + RunSession", "export", "export_bundle(result)", "call"),
    ("export", "Streamlit + RunSession", "bundle .zip (patch, rerun.py, provenance, cost)", "reply"),
    ("Streamlit + RunSession", "Person", "agent activity, cost, download", "reply"),
]


def sequence_svg() -> str:
    gap, top, left = 215, 70, 40
    xs = {name: left + 90 + i * gap for i, name in enumerate(LIFELINES)}
    width = left + 90 + (len(LIFELINES) - 1) * gap + 130
    parts, y, frames = [], top + 60, []
    for m in MESSAGES:
        kind = m[-1] if len(m) > 1 else m[0]
        if kind == "frame-start":
            frames.append([m[0], m[1], y, []])
            y += 34
        elif kind == "frame-else":
            frames[-1][3].append((y, m[1]))
            y += 34
        elif kind == "frame-end":
            op, guard, y0, elses = frames.pop()
            depth = len(frames)
            x0 = xs["Streamlit + RunSession"] - 95 + depth * 12  # nested frames sit inside their parent
            x1 = xs["SpendGate"] + 95 - depth * 12
            parts.append(f'<rect x="{x0}" y="{y0}" width="{x1 - x0}" height="{y - y0 + 6}" fill="none" '
                         f'stroke="{MUTED}" stroke-width="1.2"/>')
            parts.append(f'<path d="M{x0},{y0} h{len(op) * 8 + 26} v14 l-8,8 h-{len(op) * 8 + 18} z" '
                         f'fill="#f1f3f5" stroke="{MUTED}"/>')
            parts.append(text(x0 + 8, y0 + 16, op, size=12, weight="bold"))
            parts.append(text(x0 + len(op) * 8 + 36, y0 + 16, f"[{guard}]", size=11.5, color=MUTED))
            for ey, eguard in elses:
                parts.append(f'<line x1="{x0}" y1="{ey}" x2="{x1}" y2="{ey}" stroke="{MUTED}" '
                             f'stroke-dasharray="6 4"/>')
                parts.append(text(x0 + 8, ey + 16, f"[{eguard}]", size=11.5, color=MUTED))
            y += 22
        else:
            a, b, label, kind = m
            if kind == "self":
                x = xs[a]
                parts.append(f'<path d="M{x},{y} h40 v18 h-40" fill="none" stroke="{INK}" stroke-width="1.3" '
                             f'marker-end="url(#solid)"/>')
                parts.append(label_bg(x + 46, y + 2, len(label) * 6.9 + 6))
                parts.append(text(x + 48, y + 13, label, size=11.5))
                y += 40
            else:
                x1, x2 = xs[a], xs[b]
                dash = ' stroke-dasharray="6 4"' if kind == "reply" else ""
                marker = "open" if kind == "reply" else "solid"
                parts.append(f'<line x1="{x1}" y1="{y}" x2="{x2}" y2="{y}" stroke="{INK}" stroke-width="1.3"{dash} '
                             f'marker-end="url(#{marker})"/>')
                lx = (x1 + x2) / 2
                w = len(label) * 6.9 + 6
                parts.append(label_bg(lx - w / 2, y - 18, w))
                parts.append(text(lx, y - 6, label, size=11.5, anchor="middle"))
                y += 34
    height = y + 70
    heads = []
    for name, x in xs.items():
        w = max(len(name) * 7.4 + 20, 120)
        heads.append(f'<line x1="{x}" y1="{top + 36}" x2="{x}" y2="{height - 40}" stroke="{MUTED}" '
                     f'stroke-dasharray="3 4"/>')
        for yy in (top, height - 76):
            heads.append(f'<rect x="{x - w / 2}" y="{yy}" width="{w}" height="36" rx="4" fill="#eef6ff" '
                         f'stroke="#4a7fc1" stroke-width="1.5"/>')
            heads.append(text(x, yy + 23, name, size=12, weight="bold", anchor="middle"))
    return svg(width, height, "SIMPLE: sequence diagram, one bug-fix run", heads + parts)


# ----------------------------------------------------------------------------------------------------------------

def label_bg(x, y, w):
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="15" fill="{PAPER}" opacity="0.92"/>'


def text(x, y, s, size=12, weight="normal", color=INK, anchor="start"):
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-family="{FONT}" font-size="{size}" font-weight="{weight}" '
            f'fill="{color}" text-anchor="{anchor}">{escape(s)}</text>')


def svg(width, height, title, parts) -> str:
    defs = f"""<defs>
<marker id="open" viewBox="0 0 12 12" refX="11" refY="6" markerWidth="11" markerHeight="11" orient="auto">
  <path d="M1,1 L11,6 L1,11" fill="none" stroke="{MUTED}" stroke-width="1.4"/></marker>
<marker id="solid" viewBox="0 0 12 12" refX="11" refY="6" markerWidth="10" markerHeight="10" orient="auto">
  <path d="M1,1 L11,6 L1,11 z" fill="{INK}"/></marker>
<marker id="comp" viewBox="0 0 20 12" refX="1" refY="6" markerWidth="18" markerHeight="11" orient="auto">
  <path d="M1,6 L10,1 L19,6 L10,11 z" fill="{MUTED}"/></marker>
<marker id="agg" viewBox="0 0 20 12" refX="1" refY="6" markerWidth="18" markerHeight="11" orient="auto">
  <path d="M1,6 L10,1 L19,6 L10,11 z" fill="{PAPER}" stroke="{MUTED}"/></marker>
</defs>"""
    head = text(40, 34, title, size=18, weight="bold")
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" height="{height:.0f}" '
            f'viewBox="0 0 {width:.0f} {height:.0f}">\n<rect width="100%" height="100%" fill="{PAPER}"/>\n'
            f'{defs}\n{head}\n' + "\n".join(parts) + "\n</svg>\n")


if __name__ == "__main__":
    (HERE / "class-diagram.svg").write_text(class_svg())
    (HERE / "sequence-bug-fix.svg").write_text(sequence_svg())
    print("wrote", HERE / "class-diagram.svg", HERE / "sequence-bug-fix.svg")
