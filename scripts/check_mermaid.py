"""Render every ```mermaid block of the given Markdown files with mermaid.js in headless Chrome.

    python scripts/check_mermaid.py docs/ARCHITECTURE.md README.md

Exit code 1 if any diagram fails to parse (GitHub would show an error box instead of the diagram).
Needs Playwright with the installed Chrome (the same setup as scripts/ui_e2e.py) and internet access for the
mermaid.js CDN.
"""
from __future__ import annotations

import re
import sys

from playwright.sync_api import sync_playwright

MERMAID = "https://cdn.jsdelivr.net/npm/mermaid@10.9.1/dist/mermaid.min.js"
RENDER = """async (src) => {
  try { await mermaid.render('m' + Math.random().toString(36).slice(2), src); return 'ok'; }
  catch (e) { return String(e.message || e); }
}"""


def main(paths: list[str]) -> int:
    bad = total = 0
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True)
        page = browser.new_page()
        page.set_content("<html><body></body></html>")
        page.add_script_tag(url=MERMAID)
        page.evaluate("mermaid.initialize({startOnLoad: false})")
        for path in paths:
            blocks = re.findall(r"```mermaid\n(.*?)```", open(path, encoding="utf-8").read(), re.S)
            for i, src in enumerate(blocks, 1):
                total += 1
                res = page.evaluate(RENDER, src)
                if res != "ok":
                    bad += 1
                print(f"{path} #{i} ({src.splitlines()[0].strip()}): {'ok' if res == 'ok' else 'ERROR ' + res[:300]}")
        browser.close()
    print(f"{total} diagrams, {bad} errors")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
