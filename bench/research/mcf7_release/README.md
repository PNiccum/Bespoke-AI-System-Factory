# Research task: MCF7 serum starvation and release (open-ended biology)

A research lab's question, run through the whole ASPIRE pipeline as a test of open-ended research rather than a coding
task with known tests. The lab's list of 95 upregulated proteins (`Dataset_for_AI_advanced_agent_091626_iml`) is
unpublished lab data: it stays on the laptop, goes only to ARC models and public databases, and never enters this
repository or reaches Claude or OpenAI. `ASPIRE_DATA_PRIVATE=1` enforces that in code: with it set, the config's
`data_public` is false and the Claude and OpenAI providers refuse every call.

| File | What it is |
|---|---|
| `request.md` | The lab's request in plain words (no list data) |
| `answers.md` | Written answers the Interviewer gets in an unattended run: input format, data rule, deliverable, sources, limits |
| `decoys.csv` | Four planted rows (mouse vimentin, an isoform duplicate, human vimentin, a made-up accession) |
| `add_decoys.py` | Appends the decoys to a copy of a table in its own columns (three-column list, fold-change table, or replicate counts; reads .xlsx and writes CSV) |
| `checklist.md` | A 46-point scoring sheet built from public knowledge of the experiment, not from the list |
| `baseline.py` | The comparison: one ARC call on the 95-protein list plus a code-computed summary of the counts (fold change of each listed protein, the 40 most up and 40 most down others); the full table does not fit one call, ASPIRE gets all of it |

## Run it (laptop, VT VPN, free ARC models)

```powershell
$env:ASPIRE_DATA_PRIVATE = "1"
$T = "bench/research/mcf7_release"
python -m factory interview --request-file $T/request.md --answers-file $T/answers.md --approve
python -m factory plan     --project <project folder it prints>
python -m factory research --project <project folder>
python -m factory build    --project <project folder>
```

Then run what ASPIRE delivered, on a copy of the list with the decoys appended:

```powershell
python $T/add_decoys.py <lab list> list_with_decoys.csv
python $T/add_decoys.py "<lab counts table>.xlsx" counts_with_decoys.csv   # needs openpyxl, or export to CSV first
python <deliverable>/interpret.py list_with_decoys.csv --counts counts_with_decoys.csv --out runs/mcf7-aspire
python $T/baseline.py list_with_decoys.csv --counts counts_with_decoys.csv --out runs/mcf7-baseline
```

## Score it

Code checks first, then people:

1. Stage outcomes: did the Interviewer, planning, research and build each pass, and where did it stop? Calls,
   tokens and time per stage come from the project's run directories.
2. Decoys: how many of the four planted rows each report notices (0 to 4).
3. Citations: run `bench/projects/claim_audit/reference/claimaudit.py` on each `claims.json` with the list as
   `--dataset` and the retrieved records as `--sources`. The grounded fraction is the headline number. The baseline
   has no retrieved sources, so its PMIDs are checked against Europe PMC by hand or with the evidence tool.
4. Enrichment: the numbers in `enrichment.csv` recompute from their counts (`davidfilter.py --recheck` style).
5. Rerun: the delivered pipeline rerun from its cache gives the same tables with no network or model calls.
6. A person scores both reports with `checklist.md` and checks 10 to 20% of the passing claims against the papers.

Results go in the project folder under `evaluation/live/<time>-mcf7-research/` as a summary only: counts, scores
and stage outcomes. The reports themselves stay on the laptop until the lab says derived outputs may be shared.
