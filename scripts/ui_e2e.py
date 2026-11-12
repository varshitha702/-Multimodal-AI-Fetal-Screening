"""End-to-end UI check + screenshots (frontend on :3000, backend on :8000 must be running).

    python scripts/ui_e2e.py

Drives the installed Google Chrome with Playwright through the real UI and the real model, asserts the
key behaviour, and writes screenshots to docs/screenshots/ui_*.png and docs/screenshots/swagger_docs.png.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd
from playwright.sync_api import expect, sync_playwright

ROOT = Path(__file__).resolve().parents[1]
SHOTS = ROOT / "docs" / "screenshots"
WEB = "http://localhost:3000"
API = "http://localhost:8000"
TRAIN_DIR = ROOT / "data" / "raw" / "HC18" / "training_set"
DEMO_IMAGE = "006_HC.png"  # from the held-out TEST split (reference HC 67.84 mm)
GA_DEMO_IMAGE = "544_HC.png"  # held-out TEST split, reference HC 171.4 mm (within the 14-40 week range)


def main() -> int:
    split = pd.read_csv(ROOT / "data" / "splits" / "hc18_split.csv").set_index("filename")
    assert split.loc[DEMO_IMAGE, "split"] == "test"
    ref_hc = float(split.loc[DEMO_IMAGE, "head circumference (mm)"])
    tmp = ROOT / "outputs" / "e2e_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    corrupt = tmp / "corrupted_scan.png"
    corrupt.write_bytes((TRAIN_DIR / DEMO_IMAGE).read_bytes()[:4000])
    SHOTS.mkdir(parents=True, exist_ok=True)
    checks: list[str] = []

    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True)
        ctx = browser.new_context(viewport={"width": 1440, "height": 1000}, color_scheme="light")
        page = ctx.new_page()
        errors: list[str] = []
        page.on("pageerror", lambda e: errors.append(str(e)))

        # 1. Empty state
        page.goto(WEB)
        expect(page.get_by_test_id("empty-state")).to_be_visible()
        expect(page.get_by_text("not a medical device").first).to_be_visible()
        expect(page.get_by_test_id("api-status")).to_have_text(re.compile("Model ready"), timeout=15000)
        page.screenshot(path=SHOTS / "ui_upload_empty.png")
        checks.append("empty state, disclaimer and 'Model ready' status visible")

        # 2. Upload an HC18 test-split image -> pixel spacing auto-filled from the SHA-256 lookup
        page.get_by_test_id("file-input").set_input_files(TRAIN_DIR / DEMO_IMAGE)
        expect(page.get_by_text("HC18 auto-filled")).to_be_visible()
        px = page.locator("#pixel-size").input_value()
        assert abs(float(px) - float(split.loc[DEMO_IMAGE, "pixel size(mm)"])) < 1e-6, px
        page.screenshot(path=SHOTS / "ui_upload_ready.png")
        checks.append(f"pixel spacing auto-filled ({px} mm/px)")

        # 3. Analyse -> results
        page.get_by_role("button", name="Analyse image").click()
        expect(page.get_by_test_id("results")).to_be_visible(timeout=30000)
        hc_text = page.get_by_test_id("metric-head-circumference").inner_text()
        hc = float(re.search(r"[\d.]+", hc_text).group())
        assert abs(hc - ref_hc) < 5, (hc, ref_hc)
        page.screenshot(path=SHOTS / "ui_results.png", full_page=True)
        checks.append(f"results shown: HC {hc} mm vs reference {ref_hc} mm")

        # 4. Printable report (print media emulation)
        page.emulate_media(media="print")
        expect(page.get_by_test_id("print-report")).to_be_visible()
        page.screenshot(path=SHOTS / "ui_report_print.png", full_page=True)
        page.pdf(path=str(SHOTS / "ui_report_example.pdf"), format="A4", print_background=True)
        page.emulate_media(media="screen")
        checks.append("print report renders (screenshot + PDF)")

        # 5. Dark mode
        page.get_by_role("button", name="Toggle light/dark theme").click()
        expect(page.locator("html")).to_have_class(re.compile("dark"))
        page.screenshot(path=SHOTS / "ui_results_dark.png", full_page=True)
        checks.append("dark theme toggles")

        # 6. Error state: corrupted file -> API 400 shown to the user
        page.get_by_role("button", name="Remove image").click()
        page.get_by_test_id("file-input").set_input_files(corrupt)
        page.get_by_role("button", name="Analyse image").click()
        expect(page.get_by_text("Could not analyse the image")).to_be_visible(timeout=15000)
        expect(page.get_by_text("not a valid image or is corrupted")).to_be_visible()
        page.screenshot(path=SHOTS / "ui_error_corrupted.png")
        checks.append("corrupted upload shows the API's 400 message")

        # 7. Client-side rejection of a wrong file type
        page.get_by_role("button", name="Remove image").click()
        txt = tmp / "notes.txt"
        txt.write_text("not an image")
        page.get_by_test_id("file-input").set_input_files(txt)
        expect(page.get_by_text("is not a PNG or JPEG image")).to_be_visible()
        checks.append("wrong file type rejected in the browser")

        # 8. Model page
        page.goto(f"{WEB}/model")
        expect(page.get_by_test_id("model-info")).to_be_visible(timeout=15000)
        page.screenshot(path=SHOTS / "ui_model_page.png", full_page=True)
        checks.append("model page shows held-out test metrics")

        # 9. Mobile layout
        mobile = browser.new_context(viewport={"width": 390, "height": 844}, color_scheme="light")
        mpage = mobile.new_page()
        mpage.goto(WEB)
        mpage.get_by_test_id("file-input").set_input_files(TRAIN_DIR / DEMO_IMAGE)
        mpage.get_by_role("button", name="Analyse image").click()
        expect(mpage.get_by_test_id("results")).to_be_visible(timeout=30000)
        overflow = mpage.evaluate("document.documentElement.scrollWidth > window.innerWidth")
        assert not overflow, "horizontal scroll on mobile"
        mpage.screenshot(path=SHOTS / "ui_mobile.png", full_page=True)
        checks.append("mobile (390 px): results render, no horizontal scroll")

        # 10. Gestational age -> INTERGROWTH-21st centile (HC18 has no GA; 20+0 is an example user input)
        gpage = ctx.new_page()
        gpage.on("pageerror", lambda e: errors.append(str(e)))
        gpage.goto(WEB)
        gpage.evaluate("localStorage.setItem('theme', 'light')")  # E05 left the shared context in dark mode
        gpage.reload()
        gpage.locator("#ga-weeks").fill("12")
        gpage.get_by_test_id("file-input").set_input_files(TRAIN_DIR / GA_DEMO_IMAGE)
        expect(gpage.get_by_text("Enter whole weeks and 0–6 days")).to_be_visible()
        expect(gpage.get_by_role("button", name="Analyse image")).to_be_disabled()
        checks.append("GA outside 14-40 weeks rejected in the form (button disabled)")
        gpage.locator("#ga-weeks").fill("20")
        gpage.locator("#ga-days").fill("0")
        gpage.get_by_role("button", name="Analyse image").click()
        expect(gpage.get_by_test_id("growth-panel")).to_be_visible(timeout=30000)
        expect(gpage.get_by_test_id("centile-chart").locator("svg path").first).to_be_visible(timeout=15000)
        centile = gpage.get_by_test_id("metric-centile").inner_text()
        gpage.screenshot(path=SHOTS / "ui_results_centile.png", full_page=True)
        gpage.emulate_media(media="print")
        gpage.screenshot(path=SHOTS / "ui_report_centile_print.png", full_page=True)
        gpage.emulate_media(media="screen")
        checks.append(f"GA 20+0 on {GA_DEMO_IMAGE}: centile card ({centile}) and INTERGROWTH-21st chart shown")

        # 11. Swagger docs
        page.goto(f"{API}/docs")
        expect(page.get_by_text("/api/v1/predict")).to_be_visible(timeout=15000)
        page.screenshot(path=SHOTS / "swagger_docs.png", full_page=True)
        checks.append("Swagger UI lists the endpoints")

        assert not errors, f"page errors: {errors}"
        browser.close()

    for f in tmp.iterdir():
        f.unlink()
    tmp.rmdir()
    for i, c in enumerate(checks, 1):
        print(f"E{i:02d} PASS  {c}")
    print(f"{len(checks)} UI checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
