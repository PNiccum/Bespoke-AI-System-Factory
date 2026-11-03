# Proteomics test tasks

These tasks come from the Lazar lab's description of how they interpret a protein list (relayed by Parker on
2026-11-03). The worked example is MCF7 cells deprived of serum for 48 h and released for 24 h in serum-rich medium;
the 95 proteins are a selected subset of those that went up. The lab wants to know whether the set shows release
into the cell cycle, which proteins contradict that (arrest, apoptosis), which are drug targets or proposed
targets, and which point to new mechanisms of aberrant ER+ proliferation.

## Data rule

The 95-protein list is lab data. It is not in this repository and never goes to Claude or OpenAI. Every fixture
here is public or synthetic: real UniProt accessions and gene names, invented numbers, and PMIDs in the 9xxxxxxx
range, which do not exist on purpose. Run the tools on the real list on the laptop with ARC models only.

## The five scored tasks

`bench/projects_proteomics.yaml` holds one small tool per workflow step. Each is a project-tier task: the pipeline
builds the tool from a written request and Plan A, then plain code runs acceptance tests that no model sees.

```bash
python -m factory.bench_projects --projects bench/projects_proteomics.yaml
python -m factory.bench_projects --projects bench/projects_proteomics.yaml --only claim_audit --repeat 3
```

| Task | Workflow step | What the hidden tests probe |
|---|---|---|
| `protein_qc` | f.i: is the list plausible | UniProt accession format, isoform duplicates, a mouse entry, an unknown ID, a HER2/HER3 symbol mix-up (Gemini's entity check), a protein MCF7 does not express (vimentin), the order and skip rules |
| `evidence_table` | f.ii to f.iv: retrieve facts | UniProt and Europe PMC response parsing, GO biological process only, unique Reactome pathways, a source URL on every row, one failed protein not stopping the rest, offline replay |
| `david_filter` | f.v: narrow DAVID's chart | inclusive thresholds (at least 5 proteins, fold 1.5, Benjamini 0.05), category filter, sort order, and recomputing fold enrichment, the hypergeometric tail and the EASE score, which catches a row with a misreported fold |
| `target_flags` | question iii: drug targets | approved, clinical, preclinical, literature-only and untargeted tiers from a ChEMBL or Open Targets table (DrugBank downloads are paused), case-insensitive gene matching, and "changed but untargeted" candidates |
| `claim_audit` | question 6: validation | an AI report's claims against the fold changes (direction), the list (membership, aliases), pathway membership, and the cited sources: the citation exists, is well formed, and names the protein as a whole word (CDK10 is not CDK1) |

Each reference solution in `bench/projects/<name>/reference/` passes its tests; `tests/test_bench_projects.py`
checks that, and that an empty delivery fails.

## The live research task (not scored automatically)

The lab's full request is a research task, not a tool. Give it to the Interviewer on the laptop with the real list:

> Interpret this list of 95 proteins upregulated in MCF7 cells 24 h after release from 48 h serum starvation.
> Check the list is plausible, retrieve annotations (GO, KEGG or Reactome, disease, drug targets), cross-reference
> the literature, and report: (i) whether the set shows release into the cell cycle and proliferation; (ii) which
> proteins do not fit (arrest, apoptosis); (iii) which are drug targets or proposed targets, and which have known
> cancer relevance; the most promising candidates for new mechanisms; new hypotheses; gaps in the data and in the
> literature; and experiments to validate them. For literature, search PubMed and Europe PMC for MCF7 cell-cycle
> arrest and release studies. Use NCBI, UniProt, Ensembl, KEGG, Reactome, Human Protein Atlas and WikiPathways.
> Deliver report.md and claims.json (one claim per protein statement, in claim_audit's format).

The deliverable is scored with the tools above, so most of the judging is code:

1. `proteinqc.py` on the input list. Add a few planted decoys to a copy of the list first (a mouse accession, an
   isoform duplicate, vimentin, a made-up accession) and count how many the report notices.
2. `davidfilter.py --recheck` on the DAVID chart the report relied on.
3. `claimaudit.py` on claims.json. The grounded fraction is the headline number; every failing claim goes to a
   person.
4. A person checks a random 10 to 20% of the passing claims against the papers, as Gemini suggested.

## How to validate without going in circles (question 6)

Asking another model whether the first one was right is circular, so the checks above are ordered from code to
people:

- **Code checks first.** IDs, species, expression, directions, pathway membership and citation existence are
  facts a script can check. `claim_audit` turns "trace the provenance" into a pass or fail per claim.
- **Recompute, don't trust.** Enrichment statistics are recomputed from the counts (`david_filter --recheck`).
  Comparing DAVID's top terms with g:Profiler or Enrichr on the same list is the next step: terms both tools find
  are robust; terms only the AI mentions are suspect.
- **Planted errors.** Decoys with a known answer measure whether the pipeline catches problems, which a report
  cannot fake.
- **Rerun.** ASPIRE's exported tools rerun for free, so two runs on the same inputs must give the same tables.
- **A different model family judges what code cannot**, and a person samples the rest.

## Open questions for the lab

- The list's format: UniProt accessions, gene names, or both, and whether fold changes are included.
- Whether they can export the DAVID Functional Annotation Chart by hand, since DAVID's web service needs
  registration.
- Whether derived outputs (pathway tables, the report) may be committed or shared, or must stay on ARC too.
