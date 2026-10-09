"""Phase 8b checks: the same six inputs through (a) local vs hub-loaded models in-process, and
(b) the local demo server vs the live Space via gradio_client. Report only.

Usage:
    uv run python scripts/check_space.py --hub-local
    uv run python scripts/check_space.py --remote k08e24bryant/ewallet-complaint-analyzer
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/phase8_demo.json"
INPUTS = [("saldo saya hilang setelah top up", "blank"), ("aplikasinya bagus banget, makasih", "1"),
          ("bintang 5 deh, tapi kenapa transfer saya gagal terus", "5"), ("worst app ever", "blank"),
          ("jelek", "blank"), ("", "blank")]


def save(key: str, value) -> None:
    rep = json.loads(REPORT.read_text(encoding="utf-8")) if REPORT.exists() else {}
    rep.setdefault("hub_checks", {})[key] = value
    REPORT.write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")


def hub_local() -> None:
    """Analyzer with local models vs the same code loading from the Hub (CPU)."""
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    sys.path.insert(0, str(ROOT / "app"))
    from demo_core import Analyzer, load_config

    base = load_config(ROOT / "app/config.yaml")
    hub_cfg = {**base, "models": {**base["models"], "source": "hub"}}
    t0 = time.perf_counter()
    hub = Analyzer(hub_cfg)
    hub_load = time.perf_counter() - t0
    loc = Analyzer(base)
    rows = []
    for text, star in INPUTS:
        if not text:
            continue  # empty input is handled in app.py, not in the Analyzer
        s = None if star == "blank" else int(star)
        a, b = loc.analyze(text, s), hub.analyze(text, s)
        diff = max(abs(a.probs[k] - b.probs[k]) for k in a.probs)
        same = (a.is_complaint, a.topic_name, a.topic_note, a.language_warning) == (b.is_complaint, b.topic_name, b.topic_note, b.language_warning)
        rows.append({"text": text, "star": star, "max_prob_diff": diff, "same_outputs": same,
                     "is_complaint": b.is_complaint, "topic": b.topic_name})
        print(f"   {'OK ' if same and diff < 1e-6 else 'DIFF'} max|Δp|={diff:.1e}  {text!r} [{star}] -> complaint={b.is_complaint} topic={b.topic_name}")
    save("hub_vs_local_in_process", {"hub_load_seconds_incl_download": round(hub_load, 1), "rows": rows})


def call(client, text: str, star: str) -> dict:
    verdict, probs, topic = client.predict(text, star, api_name="/analyze")
    conf = {d["label"]: round(d["confidence"], 4) for d in (probs or {}).get("confidences", [])}
    return {"verdict": verdict, "scores": conf, "topic": topic}


def remote(space: str) -> None:
    """Call the live Space and the local server with the same inputs; compare outputs."""
    from gradio_client import Client
    from huggingface_hub import HfApi

    api = HfApi()
    t0 = time.perf_counter()
    while True:
        stage = api.get_space_runtime(space).stage
        if stage == "RUNNING":
            break
        if stage in ("BUILD_ERROR", "RUNTIME_ERROR", "CONFIG_ERROR", "NO_APP_FILE"):
            raise SystemExit(f"Space stage {stage}")
        time.sleep(5)
    t1 = time.perf_counter()
    live = Client(space, verbose=False)
    first = call(live, *INPUTS[0])
    first_s = time.perf_counter() - t1
    local = Client("http://127.0.0.1:7860/", verbose=False)
    rows = []
    for text, star in INPUTS:
        t = time.perf_counter()
        r = call(live, text, star)
        ms = 1000 * (time.perf_counter() - t)
        l = call(local, text, star)
        dmax = max((abs(r["scores"].get(k, 0) - v) for k, v in l["scores"].items()), default=0.0)
        same = r["verdict"] == l["verdict"] and r["topic"] == l["topic"]
        rows.append({"text": text, "star": star, "same_text_outputs": same, "max_score_diff": dmax, "space_ms": round(ms),
                     "space": r, "local": l})
        print(f"   {'OK ' if same and dmax <= 1e-3 else 'DIFF'} {ms:6.0f} ms  max|Δscore|={dmax:.4f}  {text!r} [{star}]")
        print(f"        {' / '.join(x for x in r['verdict'].splitlines() if x.strip())[:150]}")
        print(f"        {' / '.join(x for x in r['topic'].splitlines() if x.strip())[:150]}")
    save("live_space_vs_local_server", {"space": space, "wait_until_running_s": round(t1 - t0, 1),
                                        "first_request_s": round(first_s, 1), "rows": rows})


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--hub-local", action="store_true")
    g.add_argument("--remote", metavar="SPACE_ID")
    a = p.parse_args()
    hub_local() if a.hub_local else remote(a.remote)


if __name__ == "__main__":
    main()
