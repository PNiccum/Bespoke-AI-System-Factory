"""The comparison for the MCF7 research task: one ARC model call that answers the lab's questions directly.

    python bench/research/mcf7_release/baseline.py LIST [--counts COUNTS] --out DIR

The protein list is lab data, so this forces the private-data rule (Claude and OpenAI refuse the call) and uses the
free ``agent`` role on ARC. It writes ``baseline_report.md`` and ``baseline_usage.json`` to DIR.

The full counts table (about 3,300 rows) does not fit one call, so with ``--counts`` the call gets what does fit,
computed by code: the log2 fold change of every protein on the list, the number of proteins detected, and the 40
most up and 40 most down proteins that are not on the list. ASPIRE gets the whole table.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
import add_decoys  # noqa: E402

PROMPT = """{request}

Answer every question above for this protein list. Cite a UniProt accession or a PubMed ID for each statement about
a protein, and say "no source" when you have none. Finish with a JSON array, in a ```json block, of claims: objects
with id, protein, statement, direction ("up", "down" or "none") and citations.

## The protein list
{table}
{counts}"""


def read_table(path: str) -> tuple[list[str], list[dict]]:
    text = Path(path).read_text(encoding="utf-8-sig", errors="replace")
    first = text.splitlines()[0] if text else ""
    delim = "\t" if "\t" in first else ";" if first.count(";") > first.count(",") else ","
    reader = csv.DictReader(io.StringIO(text), delimiter=delim)
    header = [(h or "").strip() for h in (reader.fieldnames or [])]
    reader.fieldnames = header
    return [h for h in header if h], [{k: v for k, v in r.items() if k} for r in reader]


def log2_changes(header: list[str], rows: list[dict]) -> tuple[dict, str, str, str]:
    """Per accession: (gene, description, log2((mean treatment + 1) / (mean control + 1))), and the column names."""
    cols = add_decoys.find_columns(header)
    reps = add_decoys.count_columns(header, rows, {c for c in cols.values() if c})
    control = [c for c in reps if "control" in c.lower() or "non" in c.lower()]
    treat = [c for c in reps if c not in control]
    if not (cols["accession"] and control and treat):
        raise SystemExit("counts table: need an accession column and control and treatment replicate columns")

    def mean(r, cs):
        return sum(float(r[c] or 0) for c in cs) / len(cs)

    out = {}
    for r in rows:
        acc = (r.get(cols["accession"]) or "").strip()
        if acc:
            out[acc] = ((r.get(cols["gene"]) or "").strip() if cols["gene"] else "",
                        (r.get(cols["description"]) or "").strip() if cols["description"] else "",
                        math.log2((mean(r, treat) + 1) / (mean(r, control) + 1)))
    return out, cols["accession"], str(len(control)), str(len(treat))


def counts_summary(list_path: str, counts_path: str, extremes: int = 40) -> str:
    lh, lrows = read_table(list_path)
    lacc = add_decoys.find_columns(lh)["accession"]
    on_list = {(r.get(lacc) or "").strip() for r in lrows} if lacc else set()
    ch, crows = read_table(counts_path)
    changes, _, n_c, n_t = log2_changes(ch, crows)
    line = lambda a, v: f"{a}\t{v[0]}\t{v[2]:+.2f}\t{v[1]}"  # noqa: E731
    listed = [(a, changes[a]) for a in on_list if a in changes]
    others = sorted(((a, v) for a, v in changes.items() if a not in on_list), key=lambda x: x[1][2])
    head = "accession\tgene\tlog2 fold change\tdescription"
    return (f"\n## Measured changes (computed by code from the lab's spectral counts)\n"
            f"{len(changes)} proteins were detected; {n_c} control and {n_t} treatment replicates; "
            f"log2 fold change = log2((mean treatment + 1) / (mean control + 1)).\n"
            f"### Proteins on the list ({len(listed)} of {len(on_list)} found in the counts)\n{head}\n"
            + "\n".join(line(a, v) for a, v in sorted(listed, key=lambda x: -x[1][2]))
            + f"\n### The {extremes} most increased proteins not on the list\n{head}\n"
            + "\n".join(line(a, v) for a, v in others[::-1][:extremes])
            + f"\n### The {extremes} most decreased proteins not on the list\n{head}\n"
            + "\n".join(line(a, v) for a, v in others[:extremes]) + "\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("list", help="the lab's protein list (CSV, TSV or text)")
    ap.add_argument("--counts", help="the lab's full spectral-count table (CSV or TSV); summarised by code")
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=str(ROOT / "configs" / "factory.yaml"))
    args = ap.parse_args(argv)
    os.environ["ASPIRE_DATA_PRIVATE"] = "1"  # before the config loads: the list never goes to Claude or OpenAI

    from factory.config import load_config
    from factory.llm import LLM

    cfg = load_config(args.config)
    assert not cfg.data_public
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    table = Path(args.list).read_text(encoding="utf-8-sig", errors="replace")
    llm = LLM(cfg, out, use_cache=False)
    counts = counts_summary(args.list, args.counts) if args.counts else ""
    prompt = PROMPT.format(request=(HERE / "request.md").read_text(encoding="utf-8"), table=table, counts=counts)
    reply = llm.complete("agent", prompt, tag="mcf7-baseline", data="private", max_tokens=32000)
    (out / "baseline_report.md").write_text(reply.text, encoding="utf-8")
    (out / "baseline_usage.json").write_text(json.dumps({
        "model": reply.model, "stop_reason": reply.stop_reason, "seconds": reply.seconds,
        "real_usd": reply.real_usd, "priced_usd": reply.priced_usd,
        "usage": getattr(reply.usage, "__dict__", str(reply.usage))}, indent=1, default=str), encoding="utf-8")
    print(f"{reply.model}: {reply.stop_reason}, {reply.seconds:.0f} s, real ${reply.real_usd:.2f}")
    print(f"wrote {out / 'baseline_report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
