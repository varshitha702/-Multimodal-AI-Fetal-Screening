"""Render a text log (e.g. a pytest run) as a terminal-style PNG for reports.

    python scripts/render_log.py docs/evidence/pytest_run.txt docs/screenshots/test_run_pytest.png
"""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

BG, FG, DIM = (24, 26, 33), (220, 223, 228), (140, 146, 160)
GREEN, RED, YELLOW = (80, 200, 120), (240, 90, 90), (230, 190, 80)


def colour(line: str):
    if "PASSED" in line or " passed" in line:
        return GREEN
    if "FAILED" in line or "ERROR" in line or " failed" in line:
        return RED
    if "SKIPPED" in line or "skipped" in line:
        return YELLOW
    if line.startswith(("=", "platform", "rootdir", "configfile", "plugins", "cachedir", "collect")):
        return DIM
    return FG


def main(src: str, dst: str, max_cols: int = 118) -> None:
    lines = [ln.rstrip()[:max_cols] for ln in Path(src).read_text(encoding="utf-8").splitlines()]
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/consola.ttf", 15)
    except OSError:
        font = ImageFont.load_default()
    w = int(font.getlength("M") * max_cols) + 40
    lh = 19
    img = Image.new("RGB", (w, lh * len(lines) + 40), BG)
    draw = ImageDraw.Draw(img)
    for i, ln in enumerate(lines):
        draw.text((20, 20 + i * lh), ln, font=font, fill=colour(ln))
    img.save(dst)
    print(f"Wrote {dst} ({len(lines)} lines)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
