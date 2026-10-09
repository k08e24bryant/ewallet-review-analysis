"""Phase 8a diagnostic (report only, no retraining): does rating talk in the text ("bintang 5")
flip the model's prediction? Also records the demo test cases verbatim for the model card.

Each base complaint is scored without a rating phrase, with "bintang 5 deh, tapi ..." and, as a
control, with "bintang 1 deh, ...". CPU, same preprocessing as the demo.

Usage:
    uv run python scripts/diagnose_rating_talk.py
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
REPORT = ROOT / "reports/phase8_demo.json"

BASES = [  # written for this test, not real reviews
    "kenapa transfer saya gagal terus",
    "saldo saya hilang setelah top up",
    "aplikasinya sering error waktu mau bayar qris",
    "kode otp tidak pernah masuk ke nomor saya",
    "pengajuan paylater saya ditolak terus padahal bayar tepat waktu",
]
VARIANTS = {"none": "{b}", "bintang 5": "bintang 5 deh, tapi {b}", "bintang 1": "bintang 1 deh, {b}",
            "control: oke deh, tapi": "oke deh, tapi {b}"}  # same words as the 5-star variant, no rating
DEMO_CASES = [("aplikasinya bagus banget, makasih", 1), ("bintang 5 deh, tapi kenapa transfer saya gagal terus", 5),
              ("worst app ever", None), ("😡😡😡", None)]


def main() -> None:
    """Score the variants and the demo cases; print and store the results."""
    sys.stdout.reconfigure(encoding="utf-8")
    from demo_core import Analyzer, load_config

    a = Analyzer(load_config(ROOT / "app/config.yaml"))

    def score(text: str) -> dict[str, float]:
        p = a.classify([a.prep.model_text(a.prep.clean(text))])[0]
        return {l: round(float(v), 4) for l, v in zip(a.labels, p)}

    rows = []
    for b in BASES:
        for name, tpl in VARIANTS.items():
            t = tpl.format(b=b)
            s = score(t)
            rows.append({"base": b, "variant": name, "text": t, "scores": s, "label": max(s, key=s.get)})
    summary = {}
    for name in VARIANTS:
        r = [x for x in rows if x["variant"] == name]
        summary[name] = {"predicted_negative": sum(x["label"] == "negative" for x in r), "n": len(r),
                         "mean_negative_score": round(sum(x["scores"]["negative"] for x in r) / len(r), 3)}
    cases = []
    for text, star in DEMO_CASES:
        r = a.analyze(text, star)
        cases.append({"text": text, "star": star, "scores": {k: round(v, 4) for k, v in r.probs.items()},
                      "model_label": r.model_label, "is_complaint": r.is_complaint, "flagged_by": r.flagged_by,
                      "topic": r.topic_name, "topic_note": r.topic_note, "language_warning": r.language_warning})
    rep = json.loads(REPORT.read_text(encoding="utf-8")) if REPORT.exists() else {}
    rep["diagnostic_rating_talk"] = {"note": "report only; the model is fixed (evaluated on gold once)",
                                     "summary": summary, "rows": rows}
    rep["demo_test_cases"] = cases
    REPORT.write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")

    print("== rating talk in the text ==")
    for x in rows:
        s = x["scores"]
        print(f"   {x['variant']:<10} neg {s['negative']:.3f} neu {s['neutral']:.3f} pos {s['positive']:.3f} -> {x['label']:<8} | {x['text']}")
    print("summary:", json.dumps(summary))
    print("\n== demo test cases (current demo logic) ==")
    for c in cases:
        print(json.dumps(c, ensure_ascii=False))


if __name__ == "__main__":
    main()
