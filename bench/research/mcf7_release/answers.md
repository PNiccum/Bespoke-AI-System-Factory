Answers for the Interviewer, given to every round of questions in an unattended run
(`python -m factory interview --answers-file ... --approve`). They are what Parker and the lab would say.

- Input: one CSV or TSV file with a header. Columns: UniProt accession, gene symbol, and log2 fold change (24 h
  release versus 48 h starvation) when the lab provides it. Some rows may have only one of accession or gene. Human
  (Homo sapiens). About 95 rows, but the pipeline must work for any length.
- Data rule: the list is unpublished lab data. It may be sent to public databases as gene names or accessions
  (UniProt, Reactome, Europe PMC, ChEMBL, Open Targets), but never to Claude or OpenAI. Any model step uses ARC models
  only, with the key from ARC_API_KEY. Tests must run offline with recorded responses and invented sample data.
- Deliverable: a command-line pipeline, `python interpret.py LIST --out DIR`, that writes:
  - `evidence.csv`: one row per protein with its UniProt function, GO biological process terms, Reactome pathways,
    drug-target status and the source URL of each fact;
  - `enrichment.csv`: pathway and GO terms over-represented in the list, with counts, fold enrichment and an adjusted
    p-value, computed in code;
  - `report.md`: answers to questions 1 to 5, written from the evidence tables;
  - `claims.json`: one claim per statement about a protein, as a list of objects with id, protein, statement,
    direction ("up", "down" or "none") and citations (like "PMID:12345678" or "UniProt:P06493").
- Where a model is needed: only for reading abstracts and writing report.md and claims.json. Lookups, counts,
  enrichment statistics and drug-target tiers are plain code. Every model-written claim must cite a source that the
  pipeline actually retrieved.
- Success: a person at the lab finds the report useful; claims.json passes a citation check (every cited ID was
  retrieved and names the protein); the enrichment numbers recompute from the counts; planted decoys in the list (a
  mouse accession, an isoform duplicate, a protein MCF7 does not express, a made-up accession) are flagged.
- Sources: UniProt REST, Reactome Analysis Service, Europe PMC REST for literature (search "MCF7 serum starvation
  release", "cell cycle re-entry", and each protein with "breast cancer"), ChEMBL or Open Targets for drug targets.
  DAVID needs registration, so do not depend on it. DrugBank downloads are not available.
- Limits: free ARC models only; no paid API calls. A run on 95 proteins should finish in under two hours and cache
  every response so a rerun makes no network calls.
- Out of scope: raw mass-spec processing, protein quantification, wet-lab work.
