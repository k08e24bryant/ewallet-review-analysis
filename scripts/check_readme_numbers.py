"""Phase 9: cross-check every number in README.md against reports/*.json.

Each claim is recomputed from its JSON source, formatted as the README writes it, and must appear
verbatim. Afterwards every remaining number in the README prose (code blocks and links excluded)
that no claim covers is listed, so nothing slips through unchecked.

Usage:
    uv run python scripts/check_readme_numbers.py
"""

from __future__ import annotations

import json
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.topics_final import wilson  # noqa: E402

R = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in (ROOT / "reports").glob("*.json")}
P, S = "indobert/spec_3class/none|star_1_2_or_model", "reference/star_1_2"


def f3(x: float) -> str:
    return f"{x:.3f}"


def pct1(x: float) -> str:
    return f"{100 * x:.1f}"


def ci3(c: list[float]) -> str:
    return f"[{c[0]:.3f}, {c[1]:.3f}]"


def ci_pct1(c: list[float]) -> str:
    return f"[{100 * c[0]:.1f}, {100 * c[1]:.1f}]"


def claims() -> list[tuple[str, str, str]]:
    """(label, JSON source, expected README string)."""
    g, t = R["phase5_gold"], R["phase7_trends"]
    gp, gs = g["results"][P]["reweighted"], g["results"][S]["reweighted"]
    d = g["differences"][0]["reweighted"]
    out = [
        ("rows", "phase2_eda.rows", f"{R['phase2_eda']['rows']:,}"),
        ("PRIMARY F1", "phase5_gold", f"**{f3(gp['bin_neg_f1']['value'])}** {ci3(gp['bin_neg_f1']['ci95'])}"),
        ("star F1", "phase5_gold", f"**{f3(gs['bin_neg_f1']['value'])}** {ci3(gs['bin_neg_f1']['ci95'])}"),
        ("F1 headline (rounded)", "phase5_gold", f"F1 {f3(gp['bin_neg_f1']['value'])}** on 600"),
        ("F1 diff", "phase5_gold", f"(+{f3(d['bin_neg_f1']['value'])} [+{f3(d['bin_neg_f1']['ci95'][0])}, +{f3(d['bin_neg_f1']['ci95'][1])}])"),
        ("recall PRIMARY vs star", "phase5_gold", f"**{f3(gp['bin_neg_recall']['value'])}** vs **{f3(gs['bin_neg_recall']['value'])}**"),
        ("precision", "phase5_gold", f"precision {f3(gp['bin_neg_precision']['value'])} vs {f3(gs['bin_neg_precision']['value'])}"),
        ("precision diff", "phase5_gold", f"(difference −{f3(-d['bin_neg_precision']['value'])} [−{f3(-d['bin_neg_precision']['ci95'][0])}, +{f3(d['bin_neg_precision']['ci95'][1])}])"),
        ("gold size", "phase5_gold (bootstrap strata)", "600"),
    ]
    m = R["phase3_gold"]["rating_text_mismatch"]["all"]["pos_stars_labeled_negative"]
    out.append(("4-5 star complaints", "phase3_gold.rating_text_mismatch", f"**{pct1(m['reweighted'])}%** {ci_pct1(m['reweighted_ci95'])}"))
    ia = g["intra_annotator"]
    out += [("kappa 3-class", "phase5_gold.intra_annotator", f"**{ia['three_class']['kappa']:.2f}** (3 classes)"),
            ("kappa binary", "phase5_gold.intra_annotator", f"**{ia['binary_complaint']['kappa']:.2f}** (complaint vs not)"),
            ("relabel n", "phase5_gold.intra_annotator", f"{ia['n']} blind relabels")]
    dev = R["phase5_dev_selection"]["dev_validation"]["final_rows"]
    out.append(("dev size", "phase5_dev_selection", f"dev {dev}"))
    out.append(("dev size (eval design)", "phase5_dev_selection", f"dev set of {dev} reviews"))
    ov = t["overall_complaint_share_complete_weeks"]
    for app, name in (("ovo", "OVO"), ("dana", "DANA"), ("gopay", "GoPay"), ("shopeepay", "ShopeePay")):
        out.append((f"{name} share", "phase7_trends.overall", f"{name} {pct1(ov[app]['share'])}% {ci_pct1(ov[app]['ci95'])}"))
    out.append(("OVO share (summary)", "phase7_trends.overall", f"OVO {pct1(ov['ovo']['share'])}%, ShopeePay"))
    out.append(("ShopeePay share (summary)", "phase7_trends.overall", f"ShopeePay {pct1(ov['shopeepay']['share'])}% of reviews"))
    for s in t["spikes"]:
        pk = s["peak_day"]
        out.append((f"spike {pk['date']}", "phase7_trends.spikes.peak_day",
                    f"{pk['n']:,}" + (" reviews, " if pk["date"] in ("2026-07-20",) else "; " if pk["date"] != "2026-07-28" else "; ")
                    + f"{pct1(pk['complaint_share']['share'])}%"))
    dana_base = next(s for s in t["spikes"] if s["app"] == "dana")["volume"]["baseline_median_per_day"]
    out.append(("DANA median reviews/day", "phase7_trends.spikes.volume", f"about {dana_base:.0f} reviews a day"))
    jul20 = next(s for s in t["spikes"] if s["week"] == "2026-07-20")
    rp = next(x for x in jul20["topics_fitted_docs"] if x["topic"] == "Reward points cannot be redeemed")
    out.append(("Jul 20 reward points pp", "phase7_trends.spikes.topics", f"{rp['diff_pp']:.1f} percentage points [{rp['diff_ci95_pp'][0]:.1f}, {rp['diff_ci95_pp'][1]:.1f}]"))
    wk = [w for w in t["weekly"] if w["app"] == "shopeepay"]
    w21 = next(w for w in wk if w["week"] == "2026-09-21")
    w28 = next(w for w in wk if w["week"] == "2026-09-28")
    med = statistics.median(w["share"] for w in wk if not w["partial"] and w["week"] <= "2026-09-07")
    out += [("ShopeePay Sep 21", "phase7_trends.weekly", f"{pct1(w21['share'])}% [{pct1(w21['lo'])}, {pct1(w21['hi'])}]"),
            ("ShopeePay Sep 28 (partial)", "phase7_trends.weekly", f"{pct1(w28['share'])}% [{pct1(w28['lo'])}, {pct1(w28['hi'])}]"),
            ("ShopeePay median Jul 6-Sep 7", "phase7_trends.weekly (median of complete weeks)", f"median of {pct1(med)}%")]
    tt = t["topic_trends"]["weekly"]
    for app, topic, lead in (("ovo", "Money lost, missing or taken without consent", ""), ("dana", "DANA Cicil not usable", ""),
                             ("shopeepay", "Loan or paylater rejected, frozen or costly", ""), ("gopay", "Loan or paylater rejected, frozen or costly", "")):
        rows = [r for r in tt if r["app"] == app and r["topic"] == topic]
        k, n = sum(r["k"] for r in rows), sum(r["n"] for r in rows)
        lo, hi = wilson(k, n, 0.95)
        out.append((f"{app} {topic[:20]}", "phase7_trends.topic_trends (pooled complete weeks)", f"{pct1(k / n)}% [{pct1(lo)}, {pct1(hi)}]"))
    fr = R["phase6c_v3_apply"]["fresh_fit_check_results"]
    out += [("topic fit (fitted docs)", "phase6c_v3_apply.fresh_fit_check_results.decision",
             f"{100 * fr['decision']['value']:.0f}% [{100 * fr['decision']['ci95'][0]:.0f}, {100 * fr['decision']['ci95'][1]:.0f}]"),
            ("topic fit (specific)", "phase6c_v3_apply.fresh_fit_check_results", f"({100 * fr['broad_vs_specific_fitted']['specific']['yes']['share']:.0f}% for specific topics)"),
            ("old fit-check yes rate", "phase6c_v3_apply.fresh_fit_check_results.decision.baseline", f"old {100 * fr['decision']['baseline']:.0f}% \"yes\" rate")]
    return out


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    ok = bad = 0
    print("== claims (recomputed from reports/*.json) ==")
    for label, src, expected in claims():
        found = expected in readme
        ok += found
        bad += not found
        print(f"   {'OK  ' if found else 'MISS'} {label:<34} {expected!r:<58} <- {src}")
    # Every other number in the prose (code blocks and URLs removed)
    prose = re.sub(r"```.*?```", " ", readme, flags=re.S)
    prose = re.sub(r"\(https?://[^)]+\)|https?://\S+|\[[^\]]*\]\((?!#)[^)]*\)", " ", prose)
    covered = " ".join(e for _, _, e in claims())
    allowed = {"1", "2", "3", "4", "5", "6", "8", "9", "10", "11", "13", "20", "21", "25", "27", "28", "30", "50", "2026",
               "0.02", "37", "3.11", "4–5", "1–2"}  # dates, ranks, star groups, the pre-set 0.02 rule, Python, step numbers
    nums = re.findall(r"(?<![\w.])[−+-]?\d[\d,]*(?:\.\d+)?%?", prose)
    leftover = sorted({n for n in nums if n.strip("%").lstrip("−+-") not in allowed and n.strip("%") not in covered})
    print(f"\nclaims verified: {ok}, mismatches: {bad}")
    print("numbers in prose not covered by a claim (review by hand):", leftover or "none")


if __name__ == "__main__":
    main()
