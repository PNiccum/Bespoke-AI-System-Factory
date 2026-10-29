# UML

| File | What it shows |
|---|---|
| [class-diagram.svg](class-diagram.svg) / `.png` | SIMPLE's classes by package: interface, orchestration, agent selection (contract net), models and spend, plain-code tools and MCP, records and export |
| [sequence-bug-fix.svg](sequence-bug-fix.svg) / `.png` | One bug-fix run from the Streamlit page: award, locate, baseline, the patch/check/review loop, escalation with a person's approval, export |
| `SIMPLE-UML.pdf` | Both diagrams, one per page, for submission |

The diagrams are data in `render.py`; edit it and run `python docs/uml/render.py` to redraw the SVGs. To rebuild the
PNGs and the PDF: `node docs/uml/to_pdf.mjs` (needs Node and Playwright), or open the SVGs in a browser and print to
PDF.
