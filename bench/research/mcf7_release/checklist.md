# What a strong answer covers (scoring checklist)

This checklist is written from public knowledge of serum starvation and release in MCF7 cells, not from the lab's
list, so it says what to look for, not what the answer is. Score each line 0 (missing or wrong), 1 (mentioned) or
2 (argued from the list's own proteins with a valid citation). A person scores it; the code checks in README.md come
first.

## Question 1: release into the cycle (0 to 12)
- Places the 24 h time point correctly: MCF7 re-enter G1 within hours of serum and many cells are in S or G2/M by
  about 24 h, so S-phase and G2/M proteins are expected, not only early G1 ones.
- Looks for DNA replication licensing and S phase (MCM2-7, PCNA, RRM1/2, TYMS, replication factor C, FEN1).
- Looks for G2/M (CCNB1, CDK1, AURKA/B, PLK1, TOP2A, KIF and spindle proteins) and says whether they are present.
- Looks for growth and biosynthesis that serum switches on (ribosome biogenesis, translation, mTORC1 and MYC
  targets, nucleotide and lipid synthesis), and does not count every ribosomal protein as cell-cycle evidence.
- Uses enrichment computed against a sensible background (the detected proteome when known, the genome otherwise)
  and says which.
- Names the E2F and estrogen (ER) target programs, since serum contains estrogens and MCF7 are ER+.

## Question 2: proteins against the expected outcome (0 to 8)
- Checks for CDK inhibitors and arrest markers (CDKN1A/p21, CDKN1B/p27, CDKN2A, GADD45, TP53 targets) and for
  quiescence markers.
- Checks for apoptosis (BAX, BAK1, BID, cleaved-caspase substrates, cytochrome c release, PARP1 cleavage) and
  separates protein level from activity: a caspase going up is not apoptosis by itself.
- Considers stress and starvation carry-over (autophagy, unfolded protein response, heat shock) as a reason for
  proteins that do not fit.
- Says that a list of upregulated proteins alone cannot show what went down, and asks for the down list and the
  fold changes, since the list carries neither.

## Question 3: drug targets and cancer relevance (0 to 8)
- Separates approved drugs, clinical candidates, preclinical or literature-only targets and untargeted proteins,
  with a source per tier (ChEMBL or Open Targets record, trial ID, or PMID).
- Does not confuse a drug that reaches a pathway with a drug that binds the protein.
- Links ER+ context correctly (CDK4/6 inhibitors are standard of care with endocrine therapy; ESR1 itself is the
  target of tamoxifen and fulvestrant).
- Flags cancer relevance with evidence (COSMIC Cancer Gene Census, prognostic data, or papers), not general fame.

## Question 4: candidates and hypotheses (0 to 8)
- Picks a short list (about 3 to 8 proteins) and says why each is promising and why it is not already well known.
- States hypotheses that the data could falsify, and names the experiment (knockdown or inhibitor with proliferation
  and cell-cycle readout, time course, a second ER+ line such as T47D, an ER- control such as MDA-MB-231).
- Says what the data cannot show (one time point, one line, a selected subset, no replicates stated).

## Question 5: follow-up questions (0 to 4)
- Asks targeted next questions (time course, the down list, phospho data, the full detected background, replicates).

## Using the counts (0 to 6)
- Computes the fold changes from the six replicates itself and says the scale and the pseudocount it used.
- Says that three replicates a side give weak p-values and does not call a protein significant on a raw p-value alone.
- Uses the detected proteins as the enrichment background and says so.
- Checks the 95 against the data: names the ones that are not up (or all zero) in the counts, instead of taking the
  selection on trust.
- Reports strongly changed proteins that are not in the 95, including those that went down.
- Does not treat a spectral-count difference as a measured abundance ratio without the caveat.

## Hygiene (each failure is minus 2)
- A claim whose citation does not exist or does not name the protein.
- A protein the report says is on the list but is not, or the reverse.
- A gene symbol mix-up (CDK10 for CDK1, ERBB3 for ERBB2, alias confusion).
- A planted decoy that the report treats as real biology.

Total: 46 points before hygiene.
