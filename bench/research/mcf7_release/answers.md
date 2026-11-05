Answers for the Interviewer, given to every round of questions in an unattended run
(`python -m factory interview --answers-file ... --approve`). They are what Parker and the lab would say.

- Input: two files, both CSV or TSV with a header (the lab's Excel sheet is exported to CSV first; `add_decoys.py`
  does that). `LIST`: three columns, `Protein ID` (UniProt accession), `Gene name` (symbol) and `Description` (the
  protein's name, free text): the 95 proteins the lab selected as up after release; no quantities. `COUNTS`: the
  full table, about 3,300 rows, the same first three columns (the Description header may end in a space) plus six
  numeric columns of raw spectral counts: "Control 1/ Nonproliferative", "Control 2/ ...", "Control 3/ ..." and
  "Treatment1/ Proliferative", "Treatment2/ ...", "Treatment3/ ...". Many counts are zero. All 95 list proteins are
  in COUNTS. Read columns by header name (case, spacing and trailing spaces may vary); a header containing "control"
  or "nonproliferative" is a control replicate, "treatment" or "proliferative" otherwise. Human (Homo sapiens). A row
  may lack its gene symbol.
- Data rule: the list is unpublished lab data. It may be sent to public databases as gene names or accessions
  (UniProt, Reactome, Europe PMC, ChEMBL, Open Targets), but never to Claude or OpenAI. Any model step uses ARC models
  only, with the key from ARC_API_KEY. Tests must run offline with recorded responses and invented sample data.
- Deliverable: a command-line pipeline, `python interpret.py LIST --counts COUNTS --out DIR`, that writes:
  - `evidence.csv`: one row per protein with its UniProt function, GO biological process terms, Reactome pathways,
    drug-target status and the source URL of each fact;
  - `changes.csv`: one row per protein in COUNTS: mean control and treatment counts, log2 fold change as
    log2((mean treatment + 1) / (mean control + 1)), a Welch t-test p-value on log2(count + 1) with Benjamini-Hochberg
    adjustment, whether it is in LIST, and a low-count flag (all six counts under 3). With three replicates a side the
    p-values are weak; the report must say so;
  - `enrichment.csv`: pathway and GO terms over-represented in the list, with counts, fold enrichment and an adjusted
    p-value, computed in code;
  - `report.md`: answers to questions 1 to 5, written from the evidence tables;
  - `claims.json`: one claim per statement about a protein, as a list of objects with id, protein, statement,
    direction ("up" or "down" from changes.csv, "none" for a statement that does not concern its level) and citations (like "PMID:12345678" or "UniProt:P06493").
- Where a model is needed: only for reading abstracts and writing report.md and claims.json. Lookups, counts,
  enrichment statistics and drug-target tiers are plain code. Every model-written claim must cite a source that the
  pipeline actually retrieved.
- Enrichment: over-representation of the LIST against the proteins detected in COUNTS (not the whole proteome),
  computed in code and recomputable from the counts; also run it on the proteins up in COUNTS that are not in LIST.
- Use COUNTS to answer question 2 from data: which of the 95 are not up in the counts (log2 fold change at or below 0,
  or all-zero), which proteins fall strongly in the treatment condition, and whether strongly up proteins exist that
  the lab did not select.
- Success: a person at the lab finds the report useful; claims.json passes a citation check (every cited ID was
  retrieved and names the protein); the enrichment numbers recompute from the counts; planted decoys in LIST and COUNTS (a
  mouse accession, an isoform duplicate, a protein MCF7 does not express, a made-up accession) are flagged.
- Sources: UniProt REST, Reactome Analysis Service, Europe PMC REST for literature (search "MCF7 serum starvation
  release", "cell cycle re-entry", and each protein with "breast cancer"), ChEMBL or Open Targets for drug targets.
  DAVID needs registration, so do not depend on it. DrugBank downloads are not available.
- Limits: free ARC models only; no paid API calls. A run on 95 proteins should finish in under two hours and cache
  every response so a rerun makes no network calls.
- Out of scope: raw mass-spec processing, protein quantification, wet-lab work.
