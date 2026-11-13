"""End-to-end UI check + screenshots (frontend on :3000, backend on :8000 must be running).

    python scripts/ui_e2e.py

Drives the installed Google Chrome with Playwright through the real multi-page UI (Overview, Scan, CTG, Report,
Evidence) and the real models, asserts the key behaviour, and writes screenshots to docs/screenshots/ui_*.png
and docs/screenshots/swagger_docs.png. Navigation between workspace pages uses the in-app links, so the shared
analysis session (scan -> CTG -> report) is exercised the way a user would.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd
from playwright.sync_api import Page, expect, sync_playwright

ROOT = Path(__file__).resolve().parents[1]
SHOTS = ROOT / "docs" / "screenshots"
WEB = "http://localhost:3000"
API = "http://localhost:8000"
TRAIN_DIR = ROOT / "data" / "raw" / "HC18" / "training_set"
DEMO_IMAGE = "006_HC.png"  # held-out TEST split, reference HC 67.84 mm; early gestation -> plane gate warns
GA_DEMO_IMAGE = "544_HC.png"  # held-out TEST split, reference HC 171.4 mm (within the 14-40 week range)
PLANES_DIR = ROOT / "data" / "raw" / "FETAL_PLANES_DB" / "Images"
LOW_RELIABILITY_IMAGE = "794_HC.png"  # held-out TEST split; passes the plane gate, TTA flags it (HC SD 14 mm)


def wrong_plane_image() -> Path:
    """A FETAL_PLANES_DB official-TEST femur image that the gate rejects (most confident one)."""
    df = pd.read_csv(ROOT / "outputs" / "plane_test_predictions.csv")
    df = df[(df.plane == "Fetal femur") & (df.pred_plane == "Fetal femur") & ~df.passed]
    return PLANES_DIR / f"{df.sort_values('plane_conf').iloc[-1].image}.png"


def nav(page: Page, label: str) -> None:
    """Client-side navigation through the header (keeps the in-memory session)."""
    page.get_by_role("navigation", name="Main").get_by_role("link", name=re.compile(rf"{label}$")).click()


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
        ctx = browser.new_context(viewport={"width": 1440, "height": 1000})
        page = ctx.new_page()
        errors: list[str] = []
        page.on("pageerror", lambda e: errors.append(str(e)))

        # 1. Landing page
        page.goto(WEB)
        expect(page.get_by_test_id("hero-scan")).to_be_visible()
        expect(page.get_by_text("not a medical device").first).to_be_visible()
        expect(page.get_by_test_id("api-status")).to_have_text(re.compile("Model ready"), timeout=15000)
        expect(page.get_by_test_id("pipeline")).to_be_visible()
        page.wait_for_timeout(4500)  # let the hero replay reach its finished state
        page.screenshot(path=SHOTS / "ui_landing.png")
        page.screenshot(path=SHOTS / "ui_landing_full.png", full_page=True)
        checks.append("landing page: live-scan hero, pipeline, disclaimer and 'Model ready' status")

        # 2. Into the workstation; empty state; upload an HC18 test image -> pixel spacing auto-filled
        page.get_by_role("link", name="Open the workstation").click()
        expect(page.get_by_test_id("empty-state")).to_be_visible()
        page.screenshot(path=SHOTS / "ui_upload_empty.png")
        page.get_by_test_id("file-input").set_input_files(TRAIN_DIR / DEMO_IMAGE)
        expect(page.get_by_text("HC18 auto-filled")).to_be_visible()
        px = page.locator("#pixel-size").input_value()
        assert abs(float(px) - float(split.loc[DEMO_IMAGE, "pixel size(mm)"])) < 1e-6, px
        page.screenshot(path=SHOTS / "ui_upload_ready.png")
        checks.append(f"workstation empty state; pixel spacing auto-filled ({px} mm/px)")

        # 3. Plane gate warns (early-gestation image) -> HC withheld; 4. override -> HC reported, warning kept
        page.get_by_role("button", name="Analyse image").click()
        expect(page.get_by_test_id("results")).to_be_visible(timeout=60000)
        badge = page.get_by_test_id("plane-check")
        expect(badge).to_have_attribute("data-status", "warn")
        expect(page.get_by_test_id("metric-head-circumference")).to_have_text("—")
        expect(page.get_by_text("withheld: plane check")).to_be_visible()
        expect(page.get_by_test_id("stage-rail").locator("li").first).to_have_attribute("data-state", "warn")
        checks.append(f"{DEMO_IMAGE}: plane check WARN, stage rail marks the plane stage, HC withheld")
        page.get_by_role("button", name="Measure anyway (override)").click()
        expect(badge).to_contain_text("Measured anyway", timeout=60000)
        hc = float(re.search(r"[\d.]+", page.get_by_test_id("metric-head-circumference").inner_text()).group())
        assert abs(hc - ref_hc) < 5, (hc, ref_hc)
        page.screenshot(path=SHOTS / "ui_results.png", full_page=True)
        badge.screenshot(path=SHOTS / "ui_plane_check_override.png")
        checks.append(f"override: HC {hc} mm vs reference {ref_hc} mm, warning kept")

        # 5. Printable report
        page.emulate_media(media="print")
        expect(page.get_by_test_id("print-report")).to_be_visible()
        page.screenshot(path=SHOTS / "ui_report_print.png", full_page=True)
        page.pdf(path=str(SHOTS / "ui_report_example.pdf"), format="A4", print_background=True)
        page.emulate_media(media="screen")
        checks.append("print report renders (screenshot + PDF)")

        # 6. Theme: dark console by default, light report-paper theme on toggle
        expect(page.locator("html")).to_have_class(re.compile("dark"))
        page.get_by_role("button", name="Toggle light/dark theme").click()
        expect(page.locator("html")).not_to_have_class(re.compile("dark"))
        page.screenshot(path=SHOTS / "ui_results_light.png", full_page=True)
        page.get_by_role("button", name="Toggle light/dark theme").click()
        expect(page.locator("html")).to_have_class(re.compile("dark"))
        checks.append("dark console theme by default; light theme toggles and back")

        # 7. Error state: corrupted file -> API 400; 8. wrong type rejected in the browser
        page.get_by_role("button", name="Remove image").click()
        page.get_by_test_id("file-input").set_input_files(corrupt)
        page.get_by_role("button", name="Analyse image").click()
        expect(page.get_by_text("Could not analyse the image")).to_be_visible(timeout=15000)
        expect(page.get_by_text("not a valid image or is corrupted")).to_be_visible()
        page.screenshot(path=SHOTS / "ui_error_corrupted.png")
        checks.append("corrupted upload shows the API's 400 message")
        page.get_by_role("button", name="Remove image").click()
        txt = tmp / "notes.txt"
        txt.write_text("not an image")
        page.get_by_test_id("file-input").set_input_files(txt)
        expect(page.get_by_text("is not a PNG or JPEG image")).to_be_visible()
        checks.append("wrong file type rejected in the browser")

        # 9. Evidence page (tabs)
        nav(page, "Evidence")
        expect(page.get_by_test_id("model-info")).to_be_visible(timeout=15000)
        expect(page.get_by_test_id("metric-test-dice")).to_have_text(re.compile(r"0\.9"), timeout=15000)
        page.screenshot(path=SHOTS / "ui_evidence.png", full_page=True)
        page.get_by_role("tab", name=re.compile("Plane gate")).click()
        expect(page.get_by_test_id("plane-gate-info")).to_be_visible()
        expect(page.get_by_test_id("metric-plane-accuracy")).to_have_text(re.compile("%"))
        page.screenshot(path=SHOTS / "ui_evidence_plane.png", full_page=True)
        page.get_by_role("tab", name=re.compile("CTG")).click()
        expect(page.get_by_test_id("ctg-info")).to_be_visible()
        checks.append("evidence locker: live segmentation, plane-gate and CTG metrics with figures")

        # 10. Mobile layout
        mobile = browser.new_context(viewport={"width": 390, "height": 844})
        mpage = mobile.new_page()
        mpage.goto(WEB)
        assert not mpage.evaluate("document.documentElement.scrollWidth > window.innerWidth"), "landing overflows"
        mpage.screenshot(path=SHOTS / "ui_mobile_landing.png")
        mpage.goto(f"{WEB}/scan")
        mpage.get_by_test_id("file-input").set_input_files(TRAIN_DIR / GA_DEMO_IMAGE)
        mpage.get_by_role("button", name="Analyse image").click()
        expect(mpage.get_by_test_id("results")).to_be_visible(timeout=60000)
        assert not mpage.evaluate("document.documentElement.scrollWidth > window.innerWidth"), "scan overflows"
        mpage.screenshot(path=SHOTS / "ui_mobile.png", full_page=True)
        checks.append("mobile (390 px): landing and scan results render, no horizontal scroll")

        # 11-14. Fresh session: GA validation, PASS + centile, HC +- SD, uncertainty map
        gpage = ctx.new_page()
        gpage.on("pageerror", lambda e: errors.append(str(e)))
        gpage.goto(f"{WEB}/scan")
        gpage.locator("#ga-weeks").fill("12")
        gpage.get_by_test_id("file-input").set_input_files(TRAIN_DIR / GA_DEMO_IMAGE)
        expect(gpage.get_by_text("Enter whole weeks and 0–6 days")).to_be_visible()
        expect(gpage.get_by_role("button", name="Analyse image")).to_be_disabled()
        checks.append("GA outside 14-40 weeks rejected in the form (button disabled)")
        gpage.locator("#ga-weeks").fill("20")
        gpage.locator("#ga-days").fill("0")
        gpage.get_by_role("button", name="Analyse image").click()
        expect(gpage.get_by_test_id("growth-panel")).to_be_visible(timeout=60000)
        expect(gpage.get_by_test_id("centile-chart").locator("svg path").first).to_be_visible(timeout=15000)
        centile = gpage.get_by_test_id("metric-centile").inner_text()
        expect(gpage.get_by_test_id("plane-check")).to_have_attribute("data-status", "pass")
        gpage.get_by_test_id("plane-check").screenshot(path=SHOTS / "ui_plane_check_pass.png")
        gpage.screenshot(path=SHOTS / "ui_results_centile.png", full_page=True)
        checks.append(f"GA 20+0 on {GA_DEMO_IMAGE}: plane check PASS, centile card ({centile}) and chart shown")
        hc_pm = gpage.get_by_test_id("metric-head-circumference").inner_text()
        assert "±" in hc_pm, hc_pm
        expect(gpage.get_by_test_id("metric-reliability")).to_have_text("OK")
        checks.append(f"{GA_DEMO_IMAGE}: HC shown as HC ± TTA SD ({hc_pm.strip()}), reliability OK")
        gpage.get_by_role("button", name="Uncertainty map").click()
        expect(gpage.get_by_alt_text(re.compile("Uncertainty map"))).to_be_visible()
        gpage.screenshot(path=SHOTS / "ui_uncertainty_map.png")
        checks.append("viewport tab switches to the TTA uncertainty map")

        # 15-18. Report and CTG through the shared session (client-side navigation)
        nav(gpage, "Report")
        summary = gpage.get_by_test_id("screening-summary")
        expect(summary).to_have_attribute("data-level", "Routine", timeout=15000)
        expect(summary).to_contain_text("CTG (no CTG features submitted)")
        checks.append("report from ultrasound alone: Routine, CTG listed as not assessed")
        nav(gpage, "CTG")
        ctg = gpage.get_by_test_id("ctg-panel")
        ctg.get_by_role("button", name="Normal", exact=True).click()
        ctg.get_by_role("button", name="Classify CTG").click()
        expect(gpage.get_by_test_id("ctg-result")).to_contain_text("NORMAL", timeout=15000)
        nav(gpage, "Report")
        expect(summary).to_have_attribute("data-level", "Routine", timeout=15000)
        expect(summary).not_to_contain_text("Not assessed")
        checks.append("Normal CTG example (held-out test) -> NORMAL; report Routine with every input assessed")
        nav(gpage, "CTG")
        ctg.get_by_role("button", name="Pathological", exact=True).click()
        ctg.get_by_role("button", name="Classify CTG").click()
        result = gpage.get_by_test_id("ctg-result")
        expect(result).to_contain_text("PATHOLOGICAL", timeout=15000)
        expect(result).to_contain_text("Top factors")
        expect(gpage.get_by_test_id("shap-chart")).to_be_visible()
        gpage.wait_for_timeout(900)
        result.screenshot(path=SHOTS / "ui_ctg_result.png")
        gpage.screenshot(path=SHOTS / "ui_ctg_page.png", full_page=True)
        nav(gpage, "Report")
        expect(summary).to_have_attribute("data-level", "Urgent review", timeout=15000)
        expect(summary).to_contain_text("R1")
        expect(gpage.get_by_text("different patients").first).to_be_visible()
        expect(gpage.get_by_test_id("rule-ladder").locator("li[data-fired=true]").first).to_contain_text("R1")
        expect(gpage.get_by_test_id("session-tray")).to_contain_text("Urgent review")
        summary.screenshot(path=SHOTS / "ui_screening_summary_urgent.png")
        gpage.screenshot(path=SHOTS / "ui_report_page.png", full_page=True)
        nav(gpage, "Scan")
        gpage.emulate_media(media="print")
        expect(gpage.get_by_test_id("print-summary")).to_contain_text("Urgent review")
        gpage.screenshot(path=SHOTS / "ui_report_screening_print.png", full_page=True)
        gpage.emulate_media(media="screen")
        checks.append("Pathological CTG -> PATHOLOGICAL with SHAP chart; report escalates to Urgent review (R1 lit "
                      "in the decision trace, session tray updated); printed report includes it")
        nav(gpage, "CTG")
        gpage.locator("#ctg-LB").fill("400")
        expect(ctg.get_by_text("Missing or out-of-range: LB")).to_be_visible()
        expect(ctg.get_by_role("button", name="Classify CTG")).to_be_disabled()
        checks.append("out-of-range CTG input (LB 400 bpm) rejected in the form")

        # 19. Low-reliability case from the held-out test split
        upage = ctx.new_page()
        upage.on("pageerror", lambda e: errors.append(str(e)))
        upage.goto(f"{WEB}/scan")
        upage.get_by_test_id("file-input").set_input_files(TRAIN_DIR / LOW_RELIABILITY_IMAGE)
        expect(upage.get_by_text("HC18 auto-filled")).to_be_visible()
        upage.get_by_role("button", name="Analyse image").click()
        expect(upage.get_by_test_id("metric-reliability")).to_have_text("Low: re-scan", timeout=60000)
        expect(upage.get_by_label("Findings").get_by_text("Low reliability, re-scan recommended")).to_be_visible()
        upage.get_by_role("button", name="Uncertainty map").click()
        upage.screenshot(path=SHOTS / "ui_low_reliability.png", full_page=True)
        checks.append(f"{LOW_RELIABILITY_IMAGE}: 'Low: re-scan' reliability and quality warning shown")

        # 20. Plane gate on a different plane: a FETAL_PLANES_DB test femur image -> WARN, HC withheld
        femur = wrong_plane_image()
        fpage = ctx.new_page()
        fpage.on("pageerror", lambda e: errors.append(str(e)))
        fpage.goto(f"{WEB}/scan")
        fpage.get_by_test_id("file-input").set_input_files(femur)
        fpage.locator("#pixel-size").fill("0.1")
        fpage.get_by_role("button", name="Analyse image").click()
        fbadge = fpage.get_by_test_id("plane-check")
        expect(fbadge).to_have_attribute("data-status", "warn", timeout=60000)
        expect(fbadge).to_contain_text("please re-acquire")
        expect(fpage.get_by_test_id("metric-head-circumference")).to_have_text("—")
        expect(fpage.get_by_text("withheld: plane check")).to_be_visible()
        fpage.screenshot(path=SHOTS / "ui_plane_check_warn.png", full_page=True)
        checks.append(f"femur image {femur.name}: plane check WARN ('femur'), HC withheld")

        # 21. Swagger docs
        page.goto(f"{API}/docs")
        expect(page.get_by_text("/api/v1/predict").first).to_be_visible(timeout=15000)
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
