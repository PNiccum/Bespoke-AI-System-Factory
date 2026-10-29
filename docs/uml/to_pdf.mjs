// Render the UML SVGs to PNG and one PDF (one diagram per page): node docs/uml/to_pdf.mjs
// Needs Node and Playwright with Chromium. Any browser's "print to PDF" on the SVGs does the same by hand.
import { createRequire } from "module";
import { readFileSync } from "fs";
import { dirname, join } from "path";
import { fileURLToPath } from "url";

const { chromium } = createRequire(import.meta.url)("playwright"); // honors NODE_PATH for a global install
const here = dirname(fileURLToPath(import.meta.url));
const names = ["class-diagram", "sequence-bug-fix", "sequence-test-gen", "sequence-pr-review"];
const browser = await chromium.launch();
const page = await browser.newPage();
const pages = [];
for (const name of names) {
  const svg = readFileSync(join(here, `${name}.svg`), "utf8");
  const [, w, h] = svg.match(/width="(\d+)" height="(\d+)"/).map(Number);
  await page.setViewportSize({ width: w, height: h });
  await page.setContent(`<html><body style="margin:0">${svg}</body></html>`);
  await page.screenshot({ path: join(here, `${name}.png`), fullPage: true });
  pages.push({ svg, w, h });
}
// PDF: each diagram scaled to fit a landscape tabloid page (17 x 11 in).
const body = pages.map(({ svg, w, h }) =>
  `<div style="page-break-after:always;width:16.4in;height:10.4in;display:flex;align-items:center;justify-content:center">
     <div style="width:100%;height:100%">${svg.replace(/width="\d+" height="\d+"/, 'width="100%" height="100%"')}</div></div>`).join("");
await page.setContent(`<html><body style="margin:0">${body}</body></html>`);
await page.pdf({ path: join(here, "SIMPLE-UML.pdf"), width: "17in", height: "11in", margin: { top: "0.3in", bottom: "0.3in", left: "0.3in", right: "0.3in" }, printBackground: true });
await browser.close();
console.log("wrote PNGs and SIMPLE-UML.pdf in", here);
