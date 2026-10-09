"""Phase 8b: screenshots of the LOCAL demo with Playwright (Chromium), saved to docs/screenshots/.

The demo must be running: uv run python app/app.py  (http://127.0.0.1:7860)

Usage:
    uv run python scripts/screenshot_demo.py
    uv run python scripts/screenshot_demo.py --static-preview OUT.png   # render hub/static/index.html with figures
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
URL = "http://127.0.0.1:7860/"
OUT_DIR = ROOT / "docs/screenshots"
CASES = [  # (file name, text, star choice as shown in the radio group)
    ("01_balance_lost_no_star.png", "saldo saya hilang setelah top up", "blank"),
    ("02_praise_with_1_star.png", "aplikasinya bagus banget, makasih", "1"),
    ("03_bintang5_tapi_complaint_5_star.png", "bintang 5 deh, tapi kenapa transfer saya gagal terus", "5"),
    ("04_english_no_star.png", "worst app ever", "blank"),
]
VIEWPORT = {"width": 1280, "height": 900}


def demo_screenshots() -> None:
    """One full-page screenshot per case in "Analyze a review", plus the Findings tab."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport=VIEWPORT)
        for name, text, star in CASES:
            page.goto(URL, wait_until="networkidle")
            page.get_by_label("Review text (Indonesian)").fill(text)
            page.get_by_role("radio", name=star, exact=True).check()
            page.get_by_role("button", name="Analyze").click()
            page.get_by_text("Because", exact=False).first.wait_for(timeout=60_000)
            page.wait_for_timeout(500)  # let the score bars finish animating
            page.screenshot(path=str(OUT_DIR / name), full_page=True)
            print("saved", OUT_DIR / name)
        page.goto(URL, wait_until="networkidle")
        page.get_by_role("tab", name="Findings").click()
        page.wait_for_function("() => { const im = Array.from(document.images).filter(i => i.offsetParent);"
                               " return im.length >= 5 && im.every(i => i.complete && i.naturalWidth > 0); }",
                               timeout=60_000)
        page.wait_for_timeout(500)
        path = OUT_DIR / "05_findings_tab.png"
        page.screenshot(path=str(path), full_page=True)
        print("saved", path)
        browser.close()


def static_preview(out: Path) -> None:
    """Render the static findings page with its figures (as it will be published) to a PNG."""
    with tempfile.TemporaryDirectory() as tmp:
        site = Path(tmp)
        shutil.copy2(ROOT / "hub/static/index.html", site / "index.html")
        (site / "figures").mkdir()
        for f in (ROOT / "reports/figures").glob("phase7_*.png"):
            shutil.copy2(f, site / "figures" / f.name)
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(viewport={"width": 1100, "height": 900})
            page.goto((site / "index.html").as_uri(), wait_until="load")
            page.evaluate("() => Promise.all(Array.from(document.images).map(i => (i.loading = 'eager', i.decode().catch(() => {}))))")
            page.screenshot(path=str(out), full_page=True)
            browser.close()
    print("saved", out)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--static-preview", type=Path, default=None)
    a = ap.parse_args()
    static_preview(a.static_preview) if a.static_preview else demo_screenshots()


if __name__ == "__main__":
    main()
