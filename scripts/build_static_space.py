"""Phase 8b: build the static findings page (hub/static/index.html) from the project documents.

Sources, so the page cannot drift from them:
    reports/phase7_findings.md      findings, secondary note, limitations
    hub/model/README.md             "Known failure modes" section
    app/config.yaml                 figure captions (Findings tab of the demo)
    reports/figures/phase7_*.png    figures (copied next to index.html when publishing)

Usage:
    uv run python scripts/build_static_space.py
"""

from __future__ import annotations

import html
import re
import sys
from pathlib import Path

import yaml
from markdown_it import MarkdownIt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "hub/static/index.html"
MODEL_URL = "https://huggingface.co/k08e24bryant/indobert-ewallet-complaints"
GITHUB_URL = "https://github.com/k08e24bryant/ewallet-review-analysis"
FIGURES = ["phase7_weekly_complaint_share.png", "phase7_spike_daily.png", "phase7_spike_topics.png",
           "phase7_hidden_complaints.png", "phase7_topic_trends.png", "phase7_versions.png"]
EXTRA_CAPTIONS = {"phase7_versions.png": "Secondary: complaint share by installed app version. DANA v2.145's high "
                                         "share is mostly spike timing (30% outside spike days)."}

CSS = """
:root { --bg:#fcfcfb; --surface:#ffffff; --text:#0b0b0b; --muted:#52514e; --line:#e4e3df; --accent:#2a78d6; --note:#fff6d5; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#141413; --surface:#1a1a19; --text:#f2f1ec; --muted:#c3c2b7; --line:#34332f; --accent:#3987e5; --note:#3a3317; }
  figure img { background:#fcfcfb; }
}
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--text); font:16px/1.6 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif; }
main { max-width:980px; margin:0 auto; padding:32px 16px 64px; }
h1 { font-size:1.9rem; line-height:1.2; margin:0 0 8px; }
h2 { font-size:1.35rem; margin:48px 0 12px; padding-top:8px; border-top:1px solid var(--line); }
h3 { font-size:1.05rem; margin:24px 0 8px; }
p, li { color:var(--text); }
.lede { color:var(--muted); margin:0 0 16px; }
.note { background:var(--note); border-radius:8px; padding:12px 16px; margin:16px 0; }
.links { display:flex; flex-wrap:wrap; gap:8px 20px; margin:12px 0 0; padding:0; list-style:none; }
a { color:var(--accent); }
code { font-family:ui-monospace,Consolas,monospace; font-size:.9em; background:var(--line); padding:1px 5px; border-radius:4px; }
table { border-collapse:collapse; width:100%; margin:12px 0; font-size:.92rem; display:block; overflow-x:auto; }
th, td { border-bottom:1px solid var(--line); padding:6px 10px; text-align:left; vertical-align:top; }
th { color:var(--muted); font-weight:600; }
figure { margin:24px 0; background:var(--surface); border:1px solid var(--line); border-radius:10px; padding:12px; }
figure img { width:100%; height:auto; display:block; border-radius:6px; }
figcaption { color:var(--muted); font-size:.92rem; margin-top:8px; }
nav.toc { color:var(--muted); font-size:.95rem; }
nav.toc a { margin-right:14px; }
footer { color:var(--muted); font-size:.85rem; margin-top:48px; }
"""


def section(md: str, heading: str) -> str:
    """Body of a '## heading' section (up to the next '## ')."""
    m = re.search(rf"(?ms)^## {re.escape(heading)}\s*\n(.*?)(?=^## |\Z)", md)
    if not m:
        raise ValueError(f"section {heading!r} not found")
    return m.group(1).strip()


def link_figures(md: str) -> str:
    """`phase7_x.png` code spans -> links to the figure anchors on this page."""
    return re.sub(r"`(phase7_[a-z_]+)\.png`", r"[\1.png](#\1)", md)


def main() -> None:
    """Render the page."""
    sys.stdout.reconfigure(encoding="utf-8")
    mdi = MarkdownIt("commonmark").enable("table")
    findings_md = (ROOT / "reports/phase7_findings.md").read_text(encoding="utf-8")
    card_md = (ROOT / "hub/model/README.md").read_text(encoding="utf-8")
    cfg = yaml.safe_load((ROOT / "app/config.yaml").read_text(encoding="utf-8"))
    captions = {name: cap for name, cap in cfg["findings"]["items"]} | EXTRA_CAPTIONS

    intro = findings_md.split("## Findings")[0].split("\n", 1)[1].strip()  # text under the H1
    findings = section(findings_md, "Findings")
    limitations = section(findings_md, "Limitations")
    failures = section(card_md, "Known failure modes")

    figs = []
    for name in FIGURES:
        stem = name[:-4]
        cap = html.escape(captions[name])
        figs.append(f'<figure id="{stem}"><img src="figures/{name}" alt="{cap}" loading="lazy">'
                    f"<figcaption><strong>{name}</strong> — {cap}</figcaption></figure>")

    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>E-wallet Complaint Findings</title>
<meta name="description" content="Complaint trends in 201,860 Indonesian Google Play reviews of GoPay, OVO, DANA and ShopeePay (Jul–Oct 2026).">
<style>{CSS}</style>
</head>
<body>
<main>
<h1>What do users complain about in Indonesian e-wallet apps?</h1>
<p class="lede">Complaint trends in 201,860 Google Play reviews of GoPay, OVO, DANA and ShopeePay, Jul 1 – Oct 3, 2026.</p>
<div class="note"><strong>Live demo coming soon;</strong> run it locally with <code>uv run python app/app.py</code>.</div>
<ul class="links">
<li><a href="{MODEL_URL}">Model on Hugging Face</a></li>
<li><a href="{GITHUB_URL}">Code on GitHub</a></li>
</ul>
<nav class="toc"><p><a href="#findings">Findings</a><a href="#figures">Figures</a><a href="#limitations">Limitations</a><a href="#failure-modes">Known failure modes</a></p></nav>

<h2 id="findings">Findings</h2>
{mdi.render(intro)}
{mdi.render(link_figures(findings))}

<h2 id="figures">Figures</h2>
{''.join(figs)}

<h2 id="limitations">Limitations</h2>
{mdi.render(limitations)}

<h2 id="failure-modes">Known failure modes of the model</h2>
{mdi.render(failures)}

<footer>Generated from <code>reports/phase7_findings.md</code> and the model card by
<code>scripts/build_static_space.py</code>. Review data is not published; figures show aggregates only.</footer>
</main>
</body>
</html>
"""
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(page, encoding="utf-8")
    print(f"wrote {OUT} ({len(page) / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
