# UML

| File | What it shows |
|---|---|
| [activity-workflow.svg](activity-workflow.svg) / `.png` | The whole pipeline as one activity diagram: Interviewer, planning vote, knowledge loop and wiki, production, critic loop, final QA (marked subject to change), re-design, with the spend and approval gates |
| [class-diagram.svg](class-diagram.svg) / `.png` | ASPIRE's classes by package: interface, orchestration, agent selection (contract net), models and spend, plain-code tools and MCP, records and export, task types (test_gen, pr_review) |
| [sequence-bug-fix.svg](sequence-bug-fix.svg) / `.png` | One bug-fix run from the Streamlit page: award, locate, baseline, the patch/check/review loop, escalation with a person's approval, export |
| [sequence-test-gen.svg](sequence-test-gen.svg) / `.png` | Writing tests: locate, baseline, write, check, mutation score, review, with the retry loop |
| [sequence-pr-review.svg](sequence-pr-review.svg) / `.png` | Reviewing a PR: diff inspection, tests before and after, model review kept to changed lines |
| [sequence-self-extension.svg](sequence-self-extension.svg) / `.png` | Self-extension dry run: proposal, scope gate, held-out benchmark, review panel, stop for a person |
| `ASPIRE-UML.pdf` | All five diagrams, one per page, for submission |

The diagrams are data in `render.py`; edit it and run `python docs/uml/render.py` to redraw the SVGs. To rebuild the
PNGs and the PDF: `node docs/uml/to_pdf.mjs` (needs Node and Playwright), or open the SVGs in a browser and print to
PDF.
