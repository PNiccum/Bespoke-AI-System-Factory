"""The comparison for the MCF7 research task: one ARC model call that answers the lab's questions directly.

    python bench/research/mcf7_release/baseline.py LIST --out DIR

The protein list is lab data, so this forces the private-data rule (Claude and OpenAI refuse the call) and uses the
free ``agent`` role on ARC. It writes ``baseline_report.md`` and ``baseline_usage.json`` to DIR.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

PROMPT = """{request}

Answer every question above for this protein list. Cite a UniProt accession or a PubMed ID for each statement about
a protein, and say "no source" when you have none. Finish with a JSON array, in a ```json block, of claims: objects
with id, protein, statement, direction ("up", "down" or "none") and citations.

## The protein list
{table}"""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("list", help="the lab's protein list (CSV, TSV or text)")
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
    reply = llm.complete("agent", PROMPT.format(request=(HERE / "request.md").read_text(encoding="utf-8"),
                                                table=table), tag="mcf7-baseline", data="private",
                         max_tokens=32000)
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
