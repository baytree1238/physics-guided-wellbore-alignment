#!/usr/bin/env python3
"""Static and HTML-render QA for portfolio mathematics."""

from __future__ import annotations

import re
import html as html_module
import hashlib
import json
from pathlib import Path

import nbformat
from nbconvert import HTMLExporter


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "portfolio_notebook_executed.ipynb"
HTML = ROOT / "evidence" / "portfolio_notebook_rendered.html"


def main() -> int:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    markdown_cells = [cell.source for cell in notebook.cells if cell.cell_type == "markdown"]
    markdown = "\n\n".join(markdown_cells)
    if any(token in markdown for token in (r"\[", r"\]", r"\(", r"\)")):
        raise RuntimeError("backslash math delimiters survived notebook normalization")
    for source in markdown_cells:
        if source.count("$$") % 2:
            raise RuntimeError("unbalanced display-math delimiter")
        without_display = re.sub(r"\$\$.+?\$\$", "", source, flags=re.DOTALL)
        if without_display.count("$") % 2:
            raise RuntimeError("unbalanced inline-math delimiter")
    formulas = re.findall(r"\$\$(.+?)\$\$", markdown, flags=re.DOTALL)
    if len(formulas) < 8 or not all(formula.strip() for formula in formulas):
        raise RuntimeError(f"expected at least eight non-empty display formulas, found {len(formulas)}")
    exporter = HTMLExporter(template_name="classic")
    html, _ = exporter.from_notebook_node(notebook)
    if "MathJax" not in html or "MathJax Error" in html or "mjx-merror" in html:
        raise RuntimeError("HTML MathJax audit failed")
    for formula in formulas:
        # Matrix alignment uses ``&``, which nbconvert correctly HTML-escapes.
        # Accept either the literal or escaped TeX source.
        source = formula.strip()
        if source not in html and html_module.escape(source, quote=False) not in html:
            raise RuntimeError("a TeX source block did not survive nbconvert")
    HTML.write_text(html, encoding="utf-8")
    report = {
        "status": "PASS",
        "display_formulas": len(formulas),
        "backslash_delimiters": 0,
        "unbalanced_delimiters": 0,
        "html_sha256": hashlib.sha256(HTML.read_bytes()).hexdigest(),
    }
    (ROOT / "evidence" / "math_render_qa.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Math render QA: PASS ({len(formulas)} display formulas)")
    print(HTML)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
