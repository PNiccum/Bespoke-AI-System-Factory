"""Draw ASPIRE's UML class and sequence diagrams as SVG: python docs/uml/render.py

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
    "workflows": ("#e8f8f7", "#2a9d8f"),
    "extension": ("#fdf2f8", "#b8478a"),
    "pipeline": ("#fffbe6", "#a68a00"),
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
                                    "domain = 'software'", "failing_tests: list[str]",
                                    "patch: str  # pr_review"], []),
    "Plan": ("core", "", 500, 620, ["task_type: str", "steps: list[Step]"], ["plan_for(task)  # template"]),
    "Step": ("core", "", 500, 750, ["id: str", "capability: str", "needs_llm: bool", "reason: str"], []),
    "RunResult": ("core", "", 820, 440, ["status: fixed | tested | reviewed |", "        unproven | failed | stopped",
                                          "patch: str", "attempts: int", "check: TestReport", "review: Review",
                                          "report: dict  # score, findings", "stop_reason: str"], []),
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
                       "prove_fix(repo, patch, tests)", "score_tests(ScoreTestsIn)", "inspect_diff(InspectDiffIn)"]),
    "repo": ("tools", "module · code_navigator", 2040, 470, [], ["locate(LocateIn): Locations"]),
    "sandbox": ("tools", "module · test_runner", 2040, 590, ["runner: process | docker | podman | apptainer"],
                ["run_tests(RunTestsIn): TestReport", "prove_fix(...): FixProof", "run_pytest(tmp, args, timeout)"]),
    "TestReport": ("tools", "", 2040, 790, ["patch_applied: bool", "passed, failed, errors", "timed_out: bool",
                                             "output_tail: str"], []),

    "quality": ("tools", "module · mutation_tester, diff_inspector", 2040, 950, [],
                ["score_tests(ScoreTestsIn): MutationReport", "inspect_diff(InspectDiffIn): DiffFacts",
                 "  # plain code: mutate.py, diff_inspect.py"]),

    "workflows": ("workflows", "factory/workflows", 500, 1150, ["TEMPLATES: test_gen, pr_review",
                                                               "WORKFLOWS: task type -> run()"],
                  ["begin(orch, task, steps)", "finish(orch, task, result, bids)"]),
    "test_gen": ("workflows", "task type · write tests", 870, 1150,
                 ["steps: locate, baseline, write,", "  check, score, review", "MIN_SCORE = 0.6  # mutants caught"],
                 ["run(orch, task): RunResult"]),
    "pr_review": ("workflows", "task type · review a PR", 1220, 1150, ["steps: inspect, baseline, check, review"],
                  ["run(orch, task): RunResult"]),
    "PRReview": ("workflows", "", 1620, 1150, ["verdict: approve | request_changes | comment", "summary: str",
                                              "comments: list[Comment]  # changed lines only"], []),

    "planner": ("extension", "module · intake", 60, 1400, ["BLOCKS: 9 building blocks", "gaps.jsonl  # unsupported"],
                ["route(request, repo, llm, registry): Routed", "route_by_rules(request, patch)  # no model",
                 "check_steps(steps): problems"]),
    "bench": ("extension", "module · benchmark", 500, 1400, ["bench/tasks.yaml: 8 tasks, 3 held out",
                                                              "modes: aspire | baseline (one call)"],
              ["run_one(cfg, task, mode, agents): Outcome", "judge(repo, patch)  # plain code"]),
    "extend": ("extension", "module · self-extension, dry run", 960, 1400,
               ["gates: scope, held-out benchmark,", "  review panel, then a person", "archive.jsonl"],
               ["propose(cfg, agent): Result", "check_scope(proposal, incumbent)", "benchmark_gate(before, after)",
                "panel(...)  # other family"]),
    "demo": ("extension", "module · recorded runs", 1420, 1400, ["demos/*.json: task + model replies"],
             ["record(run_dir, out)", "replay(cfg, recording)  # Offline provider", "replay_llm(cfg, run_dir, rec)"]),

    "Project": ("pipeline", "blackboard · projects/<id>/", 60, 1640,
                ["board: Blackboard  # blackboard.jsonl", "  typed, numbered, license-tagged", "task.txt, plan.json"],
                ["create(base, title)", "write_file(rel, text, author)", "board.post(kind, author, data)",
                 "board.for_paid(entries)  # no ARC-only"]),
    "Interview": ("pipeline", "stage 1 · Interviewer (GLM)", 500, 1640,
                  ["MAX_QUESTIONS: 6", "check_plan(plan)  # plain code"],
                  ["start(request)", "reply(text)", "approve()  # person only"]),
    "PlanningLoop": ("pipeline", "stage 2 · three families", 960, 1640,
                     ["critic: DeepSeek  # Plan B", "analyst: gpt-oss  # pros, cons", "editor: GLM  # never votes"],
                     ["run(): Outcome", "decide(votes, problems)  # code"]),
    "KnowledgeLoop": ("pipeline", "stage 3 · knowledge and research", 1420, 1640,
                      ["evaluator: gpt-oss", "orchestrator, librarian: GLM", "verifier: DeepSeek",
                       "MAX_ROUNDS: 3"],
                      ["run(): KnowledgeResult", "research(topic)", "write_page(topic, docs)"]),
    "Wiki": ("pipeline", "shared · workspace/wiki/", 1880, 1640, ["MIN_SOURCES: 2", "pages: WikiPage (.json, .md)"],
             ["covers(topic): WikiPage | None", "save(page)"]),
    "sources": ("pipeline", "module · plain-code researchers", 1420, 1900,
                ["OpenAlex, Europe PMC, NCBI Gene", "Doc(id, text, citation, license)"],
                ["search(query, n): list[Doc]  # no model"]),

    "ProvenanceLog": ("records", "provenance.jsonl", 500, 1000, [], ["record(event)", "add_source(source)"]),
    "ProvenanceEvent": ("records", "", 820, 1000, ["kind: source|model_call|tool_call|approval", "actor: str",
                                                   "inputs, outputs: sha256"], []),
    "export": ("records", "module · export.py", 60, 1000, [], ["export_bundle(result, task, run_dir, out)",
                                                              "  -> fix.patch, rerun.py, workflow.json,",
                                                              "     evidence.json, provenance, cost",
                                                              "workflows.export_workflow_bundle(...)",
                                                              "  -> tests.patch or review.md, rerun.py"]),
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
    ("RunResult", "Review", "comp", "0..1", [(1060, 636), (1060, 847), (1002, 847)]),
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
    ("factory-tools", "quality", "dep", "", [(2040, 420), (2015, 420), (2015, 975), (2040, 975)]),
    ("Orchestrator", "workflows", "assoc", "test_gen, pr_review", [(500, 340), (464, 340), (464, 1200), (500, 1200)]),
    ("workflows", "test_gen", "dep", ""),
    ("workflows", "pr_review", "dep", "", [(640, 1278), (640, 1300), (1340, 1300), (1340, 1244)]),
    ("pr_review", "PRReview", "comp", "1"),
    ("CLI", "planner", "dep", "ask", [(60, 375), (20, 375), (20, 1450), (60, 1450)]),
    ("extend", "bench", "dep", "held-out tasks"),
    ("Interview", "Project", "assoc", "posts to"),
    ("PlanningLoop", "Interview", "dep", "approved Plan A"),
    ("KnowledgeLoop", "PlanningLoop", "dep", "plan.json"),
    ("KnowledgeLoop", "Wiki", "assoc", "reads, writes"),
    ("KnowledgeLoop", "sources", "assoc", "searches"),
]


def box_size(name, stereo, fields, methods):
    rows = [name, f"«{stereo}»" if stereo else ""] + fields + methods
    w = max(len(r) for r in rows) * CHAR + 24
    h = (2 if stereo else 1) * LINE + 10 + max(1, len(fields)) * LINE + 8 + max(1, len(methods)) * LINE + 8
    return max(w, 170), h


def class_svg() -> str:
    geo, parts = {}, []
    for name, (_pkg, stereo, x, y, fields, methods) in CLASSES.items():
        w, h = box_size(name, stereo, fields, methods)
        geo[name] = (x, y, w, h)
    width = max(x + w for x, y, w, h in geo.values()) + 40
    height = max(y + h for _, y, _, h in geo.values()) + 80

    for a, b, kind, label, *route in RELATIONS:
        parts.append(routed(route[0], kind, label) if route else edge(geo[a], geo[b], kind, label))
    for name, (pkg, stereo, _x, _y, fields, methods) in CLASSES.items():
        parts.append(class_box(name, pkg, stereo, fields, methods, *geo[name]))
    legend = legend_svg(40, height - 40)
    return svg(width, height, "ASPIRE: UML class diagram", parts + [legend])


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
        (x1, y1), (x2, y2) = max(zip(points, points[1:], strict=False), key=lambda s: abs(s[1][0] - s[0][0]) + abs(s[1][1] - s[0][1]))
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        w = len(label) * 6.2 + 8
        turn = f' transform="rotate(-90 {mx} {my})"' if x1 == x2 else ""  # vertical lanes: text runs along the line
        out.append(f'<g{turn}><rect x="{mx - w / 2:.1f}" y="{my - 9:.1f}" width="{w:.1f}" height="15" '
                   f'fill="{PAPER}"/>' + text(mx, my + 3, label, size=10.5, color=MUTED, anchor="middle") + "</g>")
    return "\n".join(out)


def legend_svg(x, y):
    items = [(name, PKG[key]) for key, name in
             [("ui", "User interface"), ("core", "Orchestration"), ("registry", "Agent selection"),
              ("llm", "Models and spend"), ("tools", "Plain-code tools and MCP"), ("records", "Records and export"),
              ("workflows", "Task types"), ("extension", "Intake, benchmark, self-extension"),
              ("pipeline", "Multi-agent pipeline")]]
    out, cx = [], x
    for label, (fill, line) in items:
        out.append(f'<rect x="{cx}" y="{y - 11}" width="14" height="14" fill="{fill}" stroke="{line}"/>')
        out.append(text(cx + 20, y, label, size=11.5))
        cx += 20 + len(label) * 7.2 + 30
    out.append(text(cx + 10, y, "◆ owns   ◇ holds   → uses   ⇢ depends on / returns", size=11.5, color=MUTED))
    return "\n".join(out)


# ----------------------------------------------------------------------------------------------------------------
# Sequence diagrams: one run of each task type from the Streamlit page
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


def sequence_svg(title: str, lifelines: list[str], messages: list) -> str:
    gap, top, left = 215, 70, 40
    xs = {name: left + 90 + i * gap for i, name in enumerate(lifelines)}
    width = left + 90 + (len(lifelines) - 1) * gap + 130
    parts, y, frames = [], top + 60, []
    for m in messages:
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
            x0 = xs[lifelines[1]] - 95 + depth * 12  # nested frames sit inside their parent
            x1 = xs[lifelines[-2]] + 95 - depth * 12
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
    return svg(width, height, title, heads + parts)


TEST_GEN_LIFELINES = ["Person", "Streamlit + RunSession", "test_gen.run", "Registry", "McpTools → factory-tools",
                      "LLM (GLM test_writer)", "LLM (DeepSeek reviewer)", "export"]

TEST_GEN = [
    ("Person", "Streamlit + RunSession", "request: write tests for parse_date", "call"),
    ("Streamlit + RunSession", "test_gen.run", "run(task)  [WORKFLOWS['test_gen']]", "call"),
    ("test_gen.run", "Registry", "award × 6 steps", "call"),
    ("Registry", "test_gen.run", "Bids: tool agents for locate, baseline, check, score", "reply"),
    ("test_gen.run", "McpTools → factory-tools", "locate(repo, request)", "call"),
    ("McpTools → factory-tools", "test_gen.run", "target files and functions", "reply"),
    ("test_gen.run", "McpTools → factory-tools", "run_tests(repo)  # baseline", "call"),
    ("McpTools → factory-tools", "test_gen.run", "TestReport before any change", "reply"),
    ("loop", "attempt 1..3; attempt 3 escalates the writer, after approval", "frame-start"),
    ("test_gen.run", "LLM (GLM test_writer)", "complete(code + existing tests + feedback)", "call"),
    ("LLM (GLM test_writer)", "test_gen.run", "diff adding tests", "reply"),
    ("test_gen.run", "test_gen.run", "refuse a diff that touches non-test files", "self"),
    ("test_gen.run", "McpTools → factory-tools", "run_tests(repo, patch)  # check", "call"),
    ("McpTools → factory-tools", "test_gen.run", "new tests pass, nothing newly broken", "reply"),
    ("test_gen.run", "McpTools → factory-tools", "score_tests(files, new tests)  # mutants", "call"),
    ("McpTools → factory-tools", "test_gen.run", "caught k of n planted bugs", "reply"),
    ("alt", "score < 0.6: surviving mutants go back to the writer", "frame-start"),
    ("test_gen.run", "test_gen.run", "feedback = missed mutants; keep the best so far", "self"),
    ("else", "score ≥ 0.6", "frame-else"),
    ("test_gen.run", "LLM (DeepSeek reviewer)", "complete(tests + score + survivors)", "call"),
    ("LLM (DeepSeek reviewer)", "test_gen.run", "accept | reject + defects", "reply"),
    ("frame-end",),
    ("frame-end",),
    ("test_gen.run", "Streamlit + RunSession", "RunResult (tested | unproven: best tests kept)", "reply"),
    ("Streamlit + RunSession", "export", "export_workflow_bundle(result)", "call"),
    ("export", "Streamlit + RunSession", "tests.patch, rerun.py + mutate.py (re-scores, no model)", "reply"),
    ("Streamlit + RunSession", "Person", "score, tests, download", "reply"),
]

PR_REVIEW_LIFELINES = ["Person", "Streamlit + RunSession", "pr_review.run", "Registry", "McpTools → factory-tools",
                       "LLM (DeepSeek pr_reviewer)", "export"]

PR_REVIEW = [
    ("Person", "Streamlit + RunSession", "PR description + diff + repo", "call"),
    ("Streamlit + RunSession", "pr_review.run", "run(task)  [WORKFLOWS['pr_review']]", "call"),
    ("pr_review.run", "Registry", "award × 4 steps", "call"),
    ("Registry", "pr_review.run", "Bids: tool agents for inspect, baseline, check", "reply"),
    ("pr_review.run", "McpTools → factory-tools", "inspect_diff(patch, repo)", "call"),
    ("McpTools → factory-tools", "pr_review.run", "changed lines; rule findings (debug, secrets, syntax)", "reply"),
    ("pr_review.run", "McpTools → factory-tools", "run_tests(repo)  # before", "call"),
    ("McpTools → factory-tools", "pr_review.run", "TestReport before", "reply"),
    ("pr_review.run", "McpTools → factory-tools", "run_tests(repo, patch)  # after", "call"),
    ("McpTools → factory-tools", "pr_review.run", "newly passing / newly failing tests", "reply"),
    ("loop", "attempt 1..3; attempt 3 escalates to Claude, after approval", "frame-start"),
    ("pr_review.run", "LLM (DeepSeek pr_reviewer)", "complete(description + diff + facts + test delta)", "call"),
    ("LLM (DeepSeek pr_reviewer)", "pr_review.run", "PRReview: verdict, summary, comments", "reply"),
    ("pr_review.run", "pr_review.run", "keep only comments on changed lines", "self"),
    ("alt", "comments dropped: send them back with the changed line numbers", "frame-start"),
    ("pr_review.run", "pr_review.run", "feedback = dropped comments", "self"),
    ("frame-end",),
    ("frame-end",),
    ("pr_review.run", "pr_review.run", "blocking rule finding overrides an approve", "self"),
    ("pr_review.run", "Streamlit + RunSession", "RunResult (reviewed)", "reply"),
    ("Streamlit + RunSession", "export", "export_workflow_bundle(result)", "call"),
    ("export", "Streamlit + RunSession", "review.md, review.json, rerun.py + diff_inspect.py", "reply"),
    ("Streamlit + RunSession", "Person", "verdict, comments, download", "reply"),
]


# ----------------------------------------------------------------------------------------------------------------

EXTEND_LIFELINES = ["Person", "factory.extend", "Registry", "bench (run_one, judge)", "LLM (GLM proposer)",
                    "LLM (DeepSeek panel)", "SpendGate", "proposals/"]
EXTEND = [
    ("Person", "factory.extend", "python -m factory.extend --agent coder", "call"),
    ("factory.extend", "Registry", "load(): incumbent coder spec", "call"),
    ("Registry", "factory.extend", "AgentSpec (role, tools, capabilities, prompt)", "reply"),
    ("factory.extend", "bench (run_one, judge)", "development tasks only (5 of 8)", "call"),
    ("bench (run_one, judge)", "factory.extend", "Outcomes: which failed and why", "reply"),
    ("factory.extend", "LLM (GLM proposer)", "propose: spec + dev failures + archive", "call"),
    ("LLM (GLM proposer)", "factory.extend", "Proposal: name, description, prompt", "reply"),
    ("factory.extend", "factory.extend", "gate 1, scope: prompt only, same role/tools, no secret", "self"),
    ("alt", "scope passed", "frame-start"),
    ("factory.extend", "bench (run_one, judge)", "held-out tasks (3), incumbent, then candidate", "call"),
    ("bench (run_one, judge)", "factory.extend", "solved and tokens for each", "reply"),
    ("factory.extend", "factory.extend", "gate 2: solves >= incumbent, tokens <= 1.5x", "self"),
    ("alt", "benchmark passed", "frame-start"),
    ("factory.extend", "LLM (DeepSeek panel)", "gate 3: old spec, new spec, benchmark", "call"),
    ("LLM (DeepSeek panel)", "factory.extend", "accept | reject + concerns", "reply"),
    ("alt", "--paid-panel", "frame-start"),
    ("LLM (DeepSeek panel)", "SpendGate", "check(claude-sonnet-5-5)", "call"),
    ("SpendGate", "Person", "approve the paid reviewer?", "call"),
    ("Person", "SpendGate", "yes / no (no = proposal rejected)", "reply"),
    ("frame-end",),
    ("frame-end",),
    ("frame-end",),
    ("factory.extend", "proposals/", "report.md, spec .yaml, archive.jsonl line", "call"),
    ("factory.extend", "Person", "adopted | awaiting a person | rejected at <gate>", "reply"),
    ("Person", "Person", "adopt: open a pull request (ASPIRE never writes registry/)", "self"),
]


# ---- Activity diagram: the whole pipeline as it runs today ----------------------------------------------------------

ACTOR = {"person": PKG["ui"], "model": PKG["llm"], "code": PKG["tools"], "mixed": PKG["core"]}


def activity_svg() -> str:
    import textwrap
    parts: list[str] = []
    box: dict[str, tuple] = {}  # name -> (cx, cy, w, h)
    lanes: list[tuple] = []

    def anchor(n, side):
        cx, cy, w, h = box[n]
        return {"t": (cx, cy - h / 2), "b": (cx, cy + h / 2), "l": (cx - w / 2, cy), "r": (cx + w / 2, cy)}[side]

    def lines_of(s, w):
        return textwrap.wrap(s, max(8, int((w - 18) / CHAR)))

    def node(name, kind, actor, cx, y, w, s):
        """Place a node whose top edge is at y; return its bottom edge."""
        if kind == "decision":
            w, h = 270, 90
            ls = lines_of(s, 170)
        elif kind in ("start", "end", "conn"):
            w = h = 26
            ls = []
        else:
            ls = lines_of(s, w)
            h = len(ls) * LINE + 16
        cy = y + h / 2
        box[name] = (cx, cy, w, h)
        fill, stroke = ACTOR.get(actor, ("#ffffff", INK))
        if kind == "decision":
            pts = f"{cx},{cy-h/2} {cx+w/2},{cy} {cx},{cy+h/2} {cx-w/2},{cy}"
            parts.append(f'<polygon points="{pts}" fill="#fffdf2" stroke="#a68a00" stroke-width="1.5"/>')
        elif kind == "start":
            parts.append(f'<circle cx="{cx}" cy="{cy}" r="11" fill="{INK}"/>')
        elif kind == "end":
            parts.append(f'<circle cx="{cx}" cy="{cy}" r="12" fill="{PAPER}" stroke="{INK}" stroke-width="1.6"/>'
                         f'<circle cx="{cx}" cy="{cy}" r="7" fill="{INK}"/>')
        elif kind == "conn":
            parts.append(f'<circle cx="{cx}" cy="{cy}" r="12" fill="{PAPER}" stroke="{INK}" stroke-width="1.4"/>')
            parts.append(text(cx, cy + 4, s, size=12, weight="bold", anchor="middle"))
        else:
            parts.append(f'<rect x="{cx-w/2}" y="{y}" width="{w}" height="{h}" rx="9" fill="{fill}" stroke="{stroke}" '
                         f'stroke-width="1.5"/>')
        ty = cy - (len(ls) - 1) * LINE / 2 + 4
        for i, l in enumerate(ls):
            parts.append(text(cx, ty + i * LINE, l, anchor="middle"))
        return y + h

    def edge(pts, label="", dashed=False, lx=None, ly=None):
        d = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        dash = ' stroke-dasharray="6 4"' if dashed else ""
        parts.append(f'<polyline points="{d}" fill="none" stroke="{INK}" stroke-width="1.4"{dash} '
                     f'marker-end="url(#solid)"/>')
        if label:
            (x0, y0), (x1, y1) = pts[0], pts[1]
            if lx is None:
                if abs(y1 - y0) < 1:  # horizontal first segment
                    sgn = 1 if x1 > x0 else -1
                    lx, ly = x0 + sgn * 8, y0 - 6
                    anchor_ = "start" if sgn > 0 else "end"
                else:
                    lx, ly, anchor_ = x0 + 8, y0 + 17, "start"
            else:
                anchor_ = "start"
            w = len(label) * 6.3 + 6
            bx = lx if anchor_ == "start" else lx - w
            parts.append(label_bg(bx - 2, ly - 11, w))
            parts.append(text(lx, ly, label, size=11, color=MUTED, anchor=anchor_))

    def down(a, b, label=""):
        edge([anchor(a, "b"), anchor(b, "t")], label)

    def lane(x0, x1, y0, y1, title, note="", dashed=False):
        dash = ' stroke-dasharray="7 5"' if dashed else ""
        lanes.append(f'<rect x="{x0}" y="{y0}" width="{x1-x0}" height="{y1-y0}" rx="12" fill="#fafbfc" '
                     f'stroke="{"#c98a1b" if dashed else "#b8c0c8"}" stroke-width="1.4"{dash}/>')
        lanes.append(text(x0 + 14, y0 + 20, title, size=13, weight="bold"))
        if note and dashed:
            lanes.append(text(x0 + 14, y0 + 34, note, size=10, color="#9a6a00"))
        elif note:
            lanes.append(text(x1 - 14, y0 + 20, note, size=11, color="#9a6a00", anchor="end"))

    G = 36  # gap for an arrow between two nodes
    L, R = 380, 1250  # column centers
    W = 400
    # ---------------- left column: intake, planning, knowledge ----------------
    y = 62
    y = node("start", "start", "", L, y, 0, "")
    y = node("task", "action", "person", L, y + 30, W, "Person types the task in plain words (Streamlit page or terminal)")
    edge([anchor("start", "b"), anchor("task", "t")])
    y1 = y + 20  # lane 1 starts
    y = node("ask", "action", "model", L, y1 + 50, W,
             "Interviewer (GLM on ARC) asks what it needs: 6 questions at most, 3 a turn")
    y = node("propose", "action", "model", L, y + G, W, "Interviewer proposes Plan A: tasks, each with an acceptance check")
    y = node("pcheck", "action", "code", L, y + G, W,
             "Plain code checks the plan: every task has an acceptance check, unique ids, dependencies point backwards")
    y = node("d_prob", "decision", "", L, y + G, 0, "Plan has problems?")
    y = node("d_appr", "decision", "", L, y + G, 0, "Person approves the plan?")
    y = node("write", "action", "code", L, y + G, W,
             "Approval writes task.txt and plan_a.json; the blackboard (blackboard.jsonl) starts")
    lane1 = (y1 + 36 - 36, y + 14)
    for a, b in [("task", "ask"), ("ask", "propose"), ("propose", "pcheck"), ("pcheck", "d_prob")]:
        down(a, b)
    down("d_prob", "d_appr", "no")
    down("d_appr", "write", "yes")
    # Plan A is rebuilt: loops on the right
    edge([anchor("d_prob", "r"), (640, box["d_prob"][1]), (640, box["propose"][1]), anchor("propose", "r")],
         "yes: back once, problems named")
    edge([anchor("d_appr", "r"), (700, box["d_appr"][1]), (700, box["ask"][1]), anchor("ask", "r")],
         "no: person's changes")
    y2 = lane1[1] + 24
    y = node("crit", "action", "model", L, y2 + 36, W,
             "Plan critic (DeepSeek) lists Plan A's weaknesses and writes an alternative, Plan B")
    y = node("ana", "action", "model", L, y + G, W, "Plan analyst (gpt-oss) writes pros and cons of each plan")
    y = node("edit", "action", "model", L, y + G, W,
             "Editor (GLM) edits both plans and says what it changed for each weakness")
    y = node("recheck", "action", "model", L, y + G, W,
             "Critic re-checks the edited Plan A; every weakness still open counts as a problem")
    y = node("vote", "action", "mixed", L, y + G, W,
             "Critic and analyst vote (the editor does not); code counts: a tie goes to fewer problems, then Plan A")
    y = node("d_win", "decision", "", L, y + G, 0, "Winner passes the plan checks?")
    y = node("plan", "action", "code", L, y + G, W, "plan.json: the refined plan, tasks and subtasks")
    lane2 = (y2, y + 14)
    edge([anchor("write", "b"), anchor("crit", "t")])
    for a, b in [("crit", "ana"), ("ana", "edit"), ("edit", "recheck"), ("recheck", "vote"), ("vote", "d_win")]:
        down(a, b)
    down("d_win", "plan", "yes")
    edge([anchor("d_win", "r"), (640, box["d_win"][1]), (640, box["edit"][1]), anchor("edit", "r")],
         "no: one repair round")
    y3 = lane2[1] + 24
    y = node("need", "action", "model", L, y3 + 36, W, "Knowledge evaluator (gpt-oss) lists the topics the work needs")
    y = node("lib", "action", "code", L, y + G, W,
             "Librarian (plain code) checks the wiki: a topic is covered by a verified page with enough sources")
    y = node("d_miss", "decision", "", L, y + G, 0, "Topics missing?")
    ytop = y + G
    y = node("r1", "action", "mixed", L, ytop, W,
             "Research orchestrator (GLM) writes searches; code connectors search (OpenAlex, PMC, NCBI, ...)")
    y = node("r2", "action", "model", L, y + G, W,
             "Orchestrator approves or rejects each document; up to 3 search rounds a topic")
    y = node("r3", "action", "mixed", L, y + G, W,
             "Librarian (GLM) writes a page, each sentence tagged with a verbatim quote; code drops any quote not in the document")
    y = node("r4", "action", "model", L, y + G, W, "Wiki verifier (DeepSeek) checks each quote supports its sentence")
    y = node("d_ver", "decision", "", L, y + G, 0, "Page verified?")
    y = node("r5", "action", "model", L, y + G, W,
             "Page saved to the shared wiki; evaluator looks again (3 evaluations at most; the last still researches)")
    lane3 = (y3, y + 14)
    edge([anchor("plan", "b"), anchor("need", "t")])
    for a, b in [("need", "lib"), ("lib", "d_miss")]:
        down(a, b)
    down("d_miss", "r1", "yes")
    for a, b in [("r1", "r2"), ("r2", "r3"), ("r3", "r4"), ("r4", "d_ver")]:
        down(a, b)
    down("d_ver", "r5", "yes")
    edge([anchor("d_ver", "r"), (640, box["d_ver"][1]), (640, box["r3"][1]), anchor("r3", "r")],
         "no: one rewrite, then a gap search")
    edge([anchor("r5", "l"), (110, box["r5"][1]), (110, box["d_miss"][1]), anchor("d_miss", "l")], "evaluate again",
         dashed=True)
    yc = lane3[1] + 24
    node("A_out", "conn", "", L, yc, 0, "A")
    edge([anchor("d_miss", "r"), (760, box["d_miss"][1]), (760, box["A_out"][1]), anchor("A_out", "r")],
         "no: knowledge approved", lx=520, ly=box["d_miss"][1] - 8)
    # ---------------- right column: production to delivery or re-design ----------------
    y = 62
    y = node("A_in", "conn", "", R, y, 0, "A")
    ry1 = y + 22
    y = node("head", "action", "model", R, ry1 + 40, W,
             "Head orchestrator (GLM) reads the task, plan and approved wiki pages and fixes the design: language, layout, entry point, run command")
    y = node("sub", "action", "model", R, y + G, W, "Sub-orchestrators (GLM) turn each task into work orders, up to 6 a task")
    y = node("build", "action", "mixed", R, y + G, W,
             "Builders (coder role) return whole files, 3 work orders at a time; code writes them under output/ (60 files at most)")
    y = node("pchk", "action", "code", R, y + G, W,
             "Code writes README.md and run.json, compiles every Python file and runs the tests in the sandbox")
    lane4 = (ry1, y + 14)
    edge([anchor("A_in", "b"), anchor("head", "t")])
    for a, b in [("head", "sub"), ("sub", "build"), ("build", "pchk")]:
        down(a, b)
    ry2 = lane4[1] + 24
    y = node("critic", "action", "model", R, ry2 + 40, W,
             "Project critic (DeepSeek) lists holes and failures, with the plain-code checks in front of it (4 reviews at most)")
    y = node("d_iss", "decision", "", R, y + G, 0, "Blocker or major issues left?")
    y = node("solve", "action", "model", R, y + G, W, "Problem solver (gpt-oss) answers each issue with replacement files")
    y = node("judge", "action", "mixed", R, y + G, W,
             "Critic accepts or declines each fix; code applies only accepted fixes and re-runs the checks")
    y = node("d_dec", "decision", "", R, y + G, 0, "Any fix declined?")
    y = node("d_aft", "decision", "", R, y + G, 0, "Already declined after research?")
    lane5 = (ry2, y + 14)
    edge([anchor("pchk", "b"), anchor("critic", "t")])
    down("critic", "d_iss")
    down("d_iss", "solve", "yes")
    down("solve", "judge")
    down("judge", "d_dec")
    down("d_dec", "d_aft", "yes")
    edge([anchor("d_dec", "r"), (1505, box["d_dec"][1]), (1505, box["critic"][1]), anchor("critic", "r")],
         "no: next review")
    edge([anchor("d_aft", "l"), (1010, box["d_aft"][1]), (1010, box["solve"][1]), anchor("solve", "l")],
         "no: wiki,", lx=1112, ly=box["d_aft"][1] - 8)
    parts.append(label_bg(1018, box["d_aft"][1] + 4, 100))
    parts.append(text(1020, box["d_aft"][1] + 16, "then research,", size=11, color=MUTED))
    parts.append(label_bg(1018, box["d_aft"][1] + 20, 100))
    parts.append(text(1020, box["d_aft"][1] + 32, "then retry", size=11, color=MUTED))
    ry3 = lane5[1] + 24
    y = node("qa1", "action", "code", R, ry3 + 40, W, "Code runs the checks first: every file compiles, the tests pass in the sandbox")
    y = node("qa2", "action", "model", R, y + G, W,
             "Final evaluator (DeepSeek, wrote none of the files) checks each task's acceptance criterion against the files and test results")
    y = node("d_qa", "decision", "", R, y + G, 0, "QA approves?")
    y = node("d_rnd", "decision", "", R, y + G, 0, "First build, and critic did not flag re-design?")
    lane6 = (ry3, y + 14)
    edge([anchor("d_iss", "r"), (1560, box["d_iss"][1]), (1560, box["qa1"][1] - 52), (R + 120, box["qa1"][1] - 52),
          (R + 120, anchor("qa1", "t")[1])], "no: critic accepts", lx=1400, ly=box["d_iss"][1] - 8)
    edge([anchor("d_aft", "r"), (1360, box["d_aft"][1]), (1360, box["qa1"][1] - 40), (R + 60, box["qa1"][1] - 40),
          (R + 60, anchor("qa1", "t")[1])], "yes: flag re-design", lx=1384, ly=box["d_aft"][1] - 8)
    down("qa1", "qa2")
    down("qa2", "d_qa")
    down("d_qa", "d_rnd", "no")
    edge([anchor("d_rnd", "r"), (1535, box["d_rnd"][1]), (1535, box["head"][1]), anchor("head", "r")],
         "yes: rebuild once", lx=1396, ly=box["d_rnd"][1] - 8)
    # ---- re-design
    ry4 = lane6[1] + 24
    y = node("arch", "action", "code", R, ry4 + 40, W,
             "Archive the project as a failure case (a snapshot later fixes are replayed on)")
    y = node("plan_pay", "action", "mixed", R, y + G, W,
             "Ask once for the whole paid plan: Claude Opus and GPT for the audit and the debate, capped at $1")
    y = node("d_pay", "decision", "", R, y + G, 0, "Person approves the paid plan?")
    y = node("aud", "action", "model", R, y + G, W,
             "Auditor 1 traces each failure to a cause on the blackboard; auditor 2 reviews; they settle the list")
    y = node("deb", "action", "model", R, y + G, W,
             "Summarizer (GLM) briefs two experts (Claude and GPT), who debate up to 2 rounds, $0.50 cap")
    y = node("d_cause", "decision", "", R, y + G, 0, "Cause kind (routed in code)")
    yb = y + G + 6
    bw = 200
    xs = (R - 215, R, R + 215)
    b1 = node("b1", "action", "mixed", xs[0], yb, bw,
              "prompt of a non-judge agent: factory.extend candidates, held-out benchmark, panel, replay; applied if every gate passes")
    b2 = node("b2", "action", "mixed", xs[1], yb, bw,
              "missing knowledge: the knowledge loop researches it now, or it is queued")
    b3 = node("b3", "action", "person", xs[2], yb, bw,
              "everything else, and any judge's prompt: waits for the person")
    yr = max(b1, b2, b3) + G
    y = node("report", "action", "code", R, yr, W,
             "redesign/report.md: what was applied, how to undo it, what was researched, what waits")
    lane7 = (ry4, y + 14)
    edge([anchor("d_rnd", "b"), anchor("arch", "t")], "no")
    down("arch", "plan_pay")
    down("plan_pay", "d_pay")
    down("d_pay", "aud", "yes")
    edge([anchor("d_pay", "r"), (1505, box["d_pay"][1]), (1505, box["aud"][1]), anchor("aud", "r")],
         "no: free ARC auditors", lx=1396, ly=box["d_pay"][1] - 8)
    down("aud", "deb")
    down("deb", "d_cause")
    cyy = box["d_cause"][1]
    for n in ("b1", "b2", "b3"):
        cx = box[n][0]
        sx = anchor("d_cause", "b")[0] if n == "b2" else (anchor("d_cause", "l")[0] if n == "b1" else anchor("d_cause", "r")[0])
        sy = anchor("d_cause", "b")[1] if n == "b2" else cyy
        pts = [(sx, sy)] if n == "b2" else [(sx, sy), (cx, sy)]
        pts.append((cx, box[n][1] - box[n][3] / 2))
        edge(pts, "", False)
    for n in ("b1", "b2", "b3"):
        edge([(box[n][0], box[n][1] + box[n][3] / 2), (box[n][0], yr - G / 2), (R, yr - G / 2), anchor("report", "t")]
             if n != "b2" else [(box[n][0], box[n][1] + box[n][3] / 2), anchor("report", "t")])
    y = node("end2", "end", "", R, y + G, 0, "")
    down("report", "end2")
    parts.append(text(R, y + 20, "Stopped for a person's decision, never silently", size=11, color=MUTED, anchor="middle"))
    # delivered: bottom of left column
    node("dl", "action", "code", 780, lane3[1] + 70, 280,
         "Deliverable: deliverable/<project id>.zip with the solution, task.txt, plan.json, knowledge.json, the wiki pages used and the public blackboard as provenance")
    edge([anchor("d_qa", "l"), (1000, box["d_qa"][1]), (1000, box["dl"][1]), anchor("dl", "r")], "yes: approved")
    node("end1", "end", "", 780, box["dl"][1] + box["dl"][3] / 2 + 28, 0, "")
    down("dl", "end1")
    # gates box
    gy = lane3[1] + 120
    gate_lines = ["Rules that hold in every stage",
                  "- Each model call goes through the spend gate. ARC models are free;",
                  "  a paid call that would take a run past $1 waits for the person.",
                  "- Credits: Claude $20, OpenAI $15, with $10 held back for the demo.",
                  "- Paid models see only public blackboard entries; arc_only text stays on ARC.",
                  "- A model never reviews its own work: the reviewer is another family.",
                  "- Every step is a blackboard entry: provenance, replay, audit trail.",
                  "- Code decides coverage, approval and routing; a model's \"approve\" is not enough."]
    gx = 40
    gh = len(gate_lines) * 18 + 20
    parts.append(f'<rect x="{gx-10}" y="{gy-20}" width="540" height="{gh}" rx="10" fill="#fffbe6" stroke="#a68a00" '
                 f'stroke-width="1.4"/>')
    for i, l in enumerate(gate_lines):
        parts.append(text(gx, gy + i * 18, l, size=11, weight="bold" if i == 0 else "normal"))
    # legend
    ly0 = gy + gh + 4
    for i, (k, lab) in enumerate((("person", "person"), ("model", "model step"), ("code", "plain code"),
                                  ("mixed", "model and code"))):
        f, st = ACTOR[k]
        parts.append(f'<rect x="{gx-10+i*150}" y="{ly0}" width="22" height="14" rx="4" fill="{f}" stroke="{st}" stroke-width="1.5"/>')
        parts.append(text(gx + 18 + i * 150, ly0 + 12, lab, size=11))
    parts.append(f'<polygon points="{gx+600},{ly0+7} {gx+612},{ly0} {gx+624},{ly0+7} {gx+612},{ly0+14}" fill="#fffdf2" stroke="#a68a00"/>')
    parts.append(text(gx + 632, ly0 + 12, "decision", size=11))
    height = max(box["end1"][1], box["end2"][1], ly0) + 50
    # lanes (drawn first)
    lane(30, 930, box["task"][1] + box["task"][3] / 2 + 10, lane1[1], "1  Intake: the Interviewer", "built")
    lane(30, 930, y2, lane2[1], "2  Planning loop: Plan A, Plan B and a vote", "built")
    lane(30, 930, y3, lane3[1], "3  Knowledge loop: research and the shared wiki", "built; PR #62 research fixes")
    lane(960, 1590, ry1, lane4[1], "4  Production", "built")
    lane(960, 1590, ry2, lane5[1], "5  Critic loop", "built")
    lane(960, 1590, ry3, lane6[1], "6  Final QA", "subject to change: QA rework in progress", dashed=True)
    lane(960, 1590, ry4, lane7[1], "7  Re-design: improve ASPIRE itself", "built; live runs pending")
    title = "ASPIRE: activity diagram, one run from plain-language task to deliverable or re-design"
    return svg(1620, height, title, lanes + parts)


def designed_svg() -> str:
    """The pipeline exactly as Parker's original plan lays it out (models numbered as in the plan)."""
    import textwrap
    parts: list[str] = []
    lanes: list[str] = []
    box: dict[str, tuple] = {}

    def anchor(n, side):
        cx, cy, w, h = box[n]
        return {"t": (cx, cy - h / 2), "b": (cx, cy + h / 2), "l": (cx - w / 2, cy), "r": (cx + w / 2, cy)}[side]

    def node(name, kind, actor, cx, y, w, s):
        if kind == "decision":
            w, h, ls = 270, 90, textwrap.wrap(s, 24)
        elif kind in ("start", "end", "conn"):
            w = h = 26
            ls = []
        else:
            ls = textwrap.wrap(s, max(8, int((w - 18) / CHAR)))
            h = len(ls) * LINE + 16
        cy = y + h / 2
        box[name] = (cx, cy, w, h)
        fill, stroke = ACTOR.get(actor, ("#ffffff", INK))
        if kind == "decision":
            pts = f"{cx},{cy-h/2} {cx+w/2},{cy} {cx},{cy+h/2} {cx-w/2},{cy}"
            parts.append(f'<polygon points="{pts}" fill="#fffdf2" stroke="#a68a00" stroke-width="1.5"/>')
        elif kind == "start":
            parts.append(f'<circle cx="{cx}" cy="{cy}" r="11" fill="{INK}"/>')
        elif kind == "end":
            parts.append(f'<circle cx="{cx}" cy="{cy}" r="12" fill="{PAPER}" stroke="{INK}" stroke-width="1.6"/>'
                         f'<circle cx="{cx}" cy="{cy}" r="7" fill="{INK}"/>')
        elif kind == "conn":
            parts.append(f'<circle cx="{cx}" cy="{cy}" r="12" fill="{PAPER}" stroke="{INK}" stroke-width="1.4"/>')
            parts.append(text(cx, cy + 4, s, size=12, weight="bold", anchor="middle"))
        else:
            parts.append(f'<rect x="{cx-w/2}" y="{y}" width="{w}" height="{h}" rx="9" fill="{fill}" '
                         f'stroke="{stroke}" stroke-width="1.5"/>')
        ty = cy - (len(ls) - 1) * LINE / 2 + 4
        for i, line in enumerate(ls):
            parts.append(text(cx, ty + i * LINE, line, anchor="middle"))
        return y + h

    def edge(pts, label="", dashed=False, lx=None, ly=None):
        d = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        dash = ' stroke-dasharray="6 4"' if dashed else ""
        parts.append(f'<polyline points="{d}" fill="none" stroke="{INK}" stroke-width="1.4"{dash} '
                     f'marker-end="url(#solid)"/>')
        if label:
            (x0, y0), (x1, y1) = pts[0], pts[1]
            if lx is None:
                if abs(y1 - y0) < 1:
                    sgn = 1 if x1 > x0 else -1
                    lx, ly, an = x0 + sgn * 8, y0 - 6, ("start" if sgn > 0 else "end")
                else:
                    lx, ly, an = x0 + 8, y0 + 17, "start"
            else:
                an = "start"
            w = len(label) * 6.3 + 6
            parts.append(label_bg((lx if an == "start" else lx - w) - 2, ly - 11, w))
            parts.append(text(lx, ly, label, size=11, color=MUTED, anchor=an))

    def down(a, b, label=""):
        edge([anchor(a, "b"), anchor(b, "t")], label)

    def chain(*names):
        for a, b in zip(names, names[1:], strict=False):
            down(a, b)

    def lane(x0, x1, y0, y1, title, dashed=False):
        lanes.append(f'<rect x="{x0}" y="{y0}" width="{x1-x0}" height="{y1-y0}" rx="12" fill="#fafbfc" '
                     f'stroke="#b8c0c8" stroke-width="1.4"/>')
        lanes.append(text(x0 + 14, y0 + 20, title, size=13, weight="bold"))

    G, L, R, W = 36, 380, 1250, 400
    # ================= left column =================
    y = node("start", "start", "", L, 62, 0, "")
    y = node("task", "action", "person", L, y + 30, W, "User submits a task request")
    top1 = y + 20
    y = node("ask", "action", "model", L, top1 + 46, W,
             "Interviewer chatbot (model 1) asks questions, clarifies the request and generates tasks")
    y = node("prelim", "action", "model", L, y + G, W, "Interviewer lays out Plan A and asks for preliminary approval")
    y = node("d_sat", "decision", "", L, y + G, 0, "User satisfied?")
    y = node("folder", "action", "code", L, y + G, W,
             "A project folder is created, with a txt file defining the requested task; the shared blackboard starts")
    end1 = y + 14
    edge([anchor("start", "b"), anchor("task", "t")])
    down("task", "ask")
    chain("ask", "prelim", "d_sat")
    down("d_sat", "folder", "yes")
    edge([anchor("d_sat", "r"), (640, box["d_sat"][1]), (640, box["ask"][1]), anchor("ask", "r")], "no: chatbot loops")
    top2 = end1 + 24
    y = node("p1", "action", "model", L, top2 + 40, W,
             "Project planner 1 (model 2) critiques the plan and tasks, edits the plan and creates an alternative, Plan B")
    y = node("p2", "action", "model", L, y + G, W, "Project planner 2 (model 3) views both plans and writes pros and cons for each")
    y = node("judge", "action", "model", L, y + G, W, "Editor / judge (model 1 or 4) edits each plan using the pros and cons")
    y = node("vote", "action", "model", L, y + G, W, "All three agents vote on which plan moves forward")
    y = node("refined", "action", "mixed", L, y + G, W, "Refined plan with tasks and subtasks")
    end2 = y + 14
    edge([anchor("folder", "b"), anchor("p1", "t")])
    chain("p1", "p2", "judge", "vote", "refined")
    top3 = end2 + 24
    y = node("keval", "action", "model", L, top3 + 40, W,
             "Knowledge evaluator decides which topics and knowledge areas the task needs")
    y = node("lib", "action", "model", L, y + G, W, "Librarian manages the system's wikis and determines which topics are covered")
    y = node("d_cov", "decision", "", L, y + G, 0, "All topics covered?")
    y = node("ro", "action", "model", L, y + G, W, "Research orchestrator delegates a subtask per topic or subtopic to research agents")
    y = node("ra", "action", "mixed", L, y + G, W,
             "Research agents use APIs and web search: scholarly articles (Elsevier ...), NCBI, PMC, parts catalogs")
    y = node("appr", "action", "model", L, y + G, W, "Research orchestrator approves each document or data set")
    y = node("d_enough", "decision", "", L, y + G, 0, "Enough information for a wiki?")
    y = node("sort", "action", "model", L, y + G, W,
             "Librarian sorts the documents into categories and starts the wiki addition / creation loop")
    y = node("ver", "action", "model", L, y + G, W, "Verifier agent checks that the additions are accurate")
    y = node("d_acc", "decision", "", L, y + G, 0, "Additions accurate?")
    y = node("reeval", "action", "model", L, y + G, W,
             "Knowledge evaluator re-checks that the requested information is present, then reviews the new data and the "
             "task for more needs")
    end3 = y + 14
    edge([anchor("refined", "b"), anchor("keval", "t")])
    chain("keval", "lib", "d_cov")
    down("d_cov", "ro", "no")
    chain("ro", "ra", "appr", "d_enough")
    down("d_enough", "sort", "yes")
    chain("sort", "ver", "d_acc")
    down("d_acc", "reeval", "yes")
    edge([anchor("d_enough", "r"), (640, box["d_enough"][1]), (640, box["ra"][1]), anchor("ra", "r")],
         "no: another round")
    edge([anchor("d_acc", "r"), (700, box["d_acc"][1]), (700, box["sort"][1]), anchor("sort", "r")], "no: fix additions")
    edge([anchor("reeval", "l"), (110, box["reeval"][1]), (110, box["d_cov"][1]), anchor("d_cov", "l")],
         "re-evaluate", dashed=True)
    yc = end3 + 24
    node("A_out", "conn", "", L, yc, 0, "A")
    edge([anchor("d_cov", "r"), (780, box["d_cov"][1]), (780, box["A_out"][1]), anchor("A_out", "r")],
         "yes: production starts", lx=520, ly=box["d_cov"][1] - 8)
    # ================= right column =================
    y = node("A_in", "conn", "", R, 62, 0, "A")
    r1 = y + 22
    y = node("head", "action", "model", R, r1 + 40, W,
             "Head orchestrator reviews the knowledge base for project templates (gathered by the knowledge loop)")
    y = node("sub", "action", "model", R, y + G, W, "Head orchestrator delegates tasks to sub-orchestrators, which delegate subtasks")
    y = node("org", "action", "model", R, y + G, W, "Subtask outputs go back up to the orchestrators and are organized together")
    e4 = y + 14
    edge([anchor("A_in", "b"), anchor("head", "t")])
    chain("head", "sub", "org")
    r2 = e4 + 24
    y = node("critic", "action", "model", R, r2 + 40, W,
             "Project critic looks for holes, failures and inefficiencies, ease of use and effectiveness")
    y = node("solve", "action", "model", R, y + G, W,
             "Problem solver 1 (a different model) suggests solutions and returns them to the critic")
    y = node("d_dec", "decision", "", R, y + G, 0, "Critic accepts the solutions?")
    y = node("d_two", "decision", "", R, y + G, 0, "Knowledge loop already reactivated once?")
    y = node("wiki", "action", "model", R, y + G, W,
             "Problem solver checks the wiki for topics that may fix it; if none, the knowledge / research loop runs again")
    e5 = y + 14
    edge([anchor("org", "b"), anchor("critic", "t")])
    chain("critic", "solve", "d_dec")
    down("d_dec", "d_two", "no")
    down("d_two", "wiki", "no")
    edge([anchor("wiki", "l"), (1010, box["wiki"][1]), (1010, box["solve"][1]), anchor("solve", "l")],
         "try again", dashed=False, lx=1018, ly=box["wiki"][1] - 52)
    r3 = e5 + 24
    y = node("qa", "action", "mixed", R, r3 + 40, W,
             "Final evaluator does QA: checks all files and does test runs")
    y = node("d_appr", "decision", "", R, y + G, 0, "Approved?")
    y = node("deliver", "action", "code", R, y + G, W,
             "A folder with the files is produced, potentially delivered through email")
    y = node("end1", "end", "", R, y + G, 0, "")
    e6 = y + 14
    edge([anchor("d_dec", "r"), (1520, box["d_dec"][1]), (1520, box["qa"][1] - 50), (R + 120, box["qa"][1] - 50),
          (R + 120, anchor("qa", "t")[1])], "yes", lx=R + 150, ly=box["d_dec"][1] - 8)
    down("qa", "d_appr")
    down("d_appr", "deliver", "yes")
    down("deliver", "end1")
    edge([anchor("d_appr", "r"), (1575, box["d_appr"][1]), (1575, box["head"][1]), anchor("head", "r")],
         "no: back with the issues", lx=R + 150, ly=box["d_appr"][1] - 8)
    r4 = e6 + 24
    y = node("aud1", "action", "model", R, r4 + 40, W,
             "Auditor 1 evaluates the successes, failures and shortcomings, and what kept the system from a good solution")
    y = node("aud2", "action", "model", R, y + G, W,
             "Auditor 2 (a different model) evaluates the solutions and returns suggestions to auditor 1")
    y = node("aud1b", "action", "model", R, y + G, W, "Auditor 1 returns an evaluation of those suggestions")
    y = node("d_agree", "decision", "", R, y + G, 0, "Auditors agree on problems and fixes?")
    y = node("sum", "action", "model", R, y + G, W,
             "An LLM call summarizes the project progress and transcript, and what the auditors suggested")
    y = node("deb", "action", "mixed", R, y + G, W,
             "Claude and ChatGPT APIs are queried for ways to improve the overall system, and debate each other")
    y = node("d_fix", "decision", "", R, y + G, 0, "Fix decided, or credit cost limit reached?")
    y = node("rep", "action", "person", R, y + G, W, "Report of the fix is created and sent to Parker for approval")
    y = node("end2", "end", "", R, y + G, 0, "")
    e7 = y + 14
    edge([anchor("d_two", "r"), (1545, box["d_two"][1]), (1545, box["aud1"][1]), anchor("aud1", "r")],
         "yes: re-design", lx=R + 150, ly=box["d_two"][1] - 8)
    chain("aud1", "aud2", "aud1b", "d_agree")
    down("d_agree", "sum", "yes")
    chain("sum", "deb", "d_fix")
    down("d_fix", "rep", "yes")
    down("rep", "end2")
    edge([anchor("d_agree", "r"), (1500, box["d_agree"][1]), (1500, box["aud2"][1]), anchor("aud2", "r")],
         "no: another exchange", lx=R + 150, ly=box["d_agree"][1] - 8)
    edge([anchor("d_fix", "r"), (1500, box["d_fix"][1]), (1500, box["deb"][1]), anchor("deb", "r")],
         "no: keep debating", lx=R + 150, ly=box["d_fix"][1] - 8)
    # notes and legend
    gy = end3 + 150
    notes = ["Notes on the plan",
             "- Every agent runs on an ARC model except the Claude and ChatGPT calls",
             "  in the re-design debate, which spend credits against a cost limit.",
             "- Agents share one blackboard from the moment the folder is created.",
             "- Models are numbered as in the plan; a different model checks each",
             "  stage's work (planner 2, problem solver 1, auditor 2, the verifier).",
             "- As built, the editor does not vote and email delivery is left out."]
    parts.append(f'<rect x="30" y="{gy-20}" width="540" height="{len(notes)*18+20}" rx="10" fill="#fffbe6" '
                 f'stroke="#a68a00" stroke-width="1.4"/>')
    for i, line in enumerate(notes):
        parts.append(text(40, gy + i * 18, line, size=11, weight="bold" if i == 0 else "normal"))
    ly0 = gy + len(notes) * 18 + 28
    for i, (k, lab) in enumerate((("person", "person"), ("model", "model step"), ("code", "plain step"),
                                  ("mixed", "model and tools"))):
        f, st = ACTOR[k]
        parts.append(f'<rect x="{30+i*150}" y="{ly0}" width="22" height="14" rx="4" fill="{f}" stroke="{st}" '
                     f'stroke-width="1.5"/>')
        parts.append(text(58 + i * 150, ly0 + 12, lab, size=11))
    height = max(e7, ly0) + 60
    lane(30, 930, top1 - 4 + 0, end1, "Intake: Interviewer loop")
    lane(30, 930, top2, end2, "Planning loop")
    lane(30, 930, top3, end3, "Knowledge evaluation and research loop")
    lane(960, 1590, r1, e4, "Production")
    lane(960, 1590, r2, e5, "Critic loop")
    lane(960, 1590, r3, e6, "Final evaluation")
    lane(960, 1590, r4, e7, "Re-design loop")
    title = "ASPIRE: activity diagram of the original plan (as designed)"
    return svg(1620, height, title, lanes + parts)


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
    (HERE / "class-diagram.svg").write_text(class_svg(), encoding="utf-8")
    diagrams = {"sequence-bug-fix": ("ASPIRE: sequence diagram, one bug-fix run", LIFELINES, MESSAGES),
                "sequence-test-gen": ("ASPIRE: sequence diagram, writing tests", TEST_GEN_LIFELINES, TEST_GEN),
                "sequence-pr-review": ("ASPIRE: sequence diagram, reviewing a pull request", PR_REVIEW_LIFELINES,
                                       PR_REVIEW),
                "sequence-self-extension": ("ASPIRE: sequence diagram, self-extension dry run", EXTEND_LIFELINES,
                                            EXTEND)}
    for name, args in diagrams.items():
        (HERE / f"{name}.svg").write_text(sequence_svg(*args), encoding="utf-8")
    (HERE / "activity-workflow.svg").write_text(activity_svg(), encoding="utf-8")
    (HERE / "activity-designed.svg").write_text(designed_svg(), encoding="utf-8")
    print("wrote", HERE / "class-diagram.svg", *(HERE / f"{n}.svg" for n in diagrams))
