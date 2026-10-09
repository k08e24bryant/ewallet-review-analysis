"""Phase 8a: CPU-only latency and peak RAM of the demo, simulating HF Spaces CPU basic (2 vCPU).

CUDA is hidden before torch loads and torch uses 2 threads. Peak RAM = maximum resident set size
of this process, sampled every 20 ms from start-up through all requests.

Usage:
    uv run python scripts/benchmark_demo.py
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # before torch is imported anywhere ("" is not enough on Windows)

import json
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
REPORT = ROOT / "reports/phase8_demo.json"
N_REVIEWS, SEED, THREADS = 50, 42, 2


class PeakRSS(threading.Thread):
    """Samples the process RSS until stopped; keeps the maximum."""

    def __init__(self) -> None:
        super().__init__(daemon=True)
        self.proc, self.peak, self.stop = psutil.Process(), 0, threading.Event()

    def run(self) -> None:
        while not self.stop.is_set():
            self.peak = max(self.peak, self.proc.memory_info().rss)
            time.sleep(0.02)


def main() -> None:
    """Load the demo on CPU, time analyze() on real and example inputs, record peak RAM."""
    sys.stdout.reconfigure(encoding="utf-8")
    mon = PeakRSS()
    mon.start()
    import torch

    torch.set_num_threads(THREADS)
    assert not torch.cuda.is_available(), "CUDA must be hidden"
    from demo_core import Analyzer, load_config

    cfg = load_config(ROOT / "app/config.yaml")
    t0 = time.perf_counter()
    analyzer = Analyzer(cfg)
    load_s = time.perf_counter() - t0
    rss_after_load = psutil.Process().memory_info().rss

    rv = pd.read_parquet(ROOT / "data/processed/reviews_clean.parquet", columns=["content", "score"])
    sample = rv.sample(n=N_REVIEWS, random_state=SEED)
    inputs = [(c, int(s)) for c, s in zip(sample["content"], sample["score"])]
    inputs += [(t, None if s == "blank" else int(s)) for t, s in cfg["examples"]]
    analyzer.analyze("pemanasan", None)  # warm-up (first call builds kernels)
    rows = []
    for text, star in inputs:
        t = time.perf_counter()
        r = analyzer.analyze(text, star)
        rows.append({"ms": 1000 * (time.perf_counter() - t), "with_topic": r.topic_id is not None})
    mon.stop.set()
    mon.join()
    df = pd.DataFrame(rows)

    def stats(x: pd.Series) -> dict[str, float]:
        return {"n": int(len(x)), "median_ms": round(float(x.median()), 1), "p95_ms": round(float(np.percentile(x, 95)), 1),
                "max_ms": round(float(x.max()), 1)}

    res = {
        "setup": {"device": "cpu (CUDA hidden)", "torch_threads": THREADS, "simulates": "HF Spaces CPU basic (2 vCPU, 16 GB)",
                  "requests": len(inputs), "inputs": f"{N_REVIEWS} random reviews (with their star) + {len(cfg['examples'])} demo examples",
                  "torch": torch.__version__},
        "load_seconds": round(load_s, 1),
        "latency_all": stats(df["ms"]),
        "latency_with_topic": stats(df.loc[df["with_topic"], "ms"]),
        "latency_without_topic": stats(df.loc[~df["with_topic"], "ms"]),
        "rss_after_load_mb": round(rss_after_load / 2**20),
        "peak_rss_mb": round(mon.peak / 2**20),
    }
    rep = json.loads(REPORT.read_text(encoding="utf-8")) if REPORT.exists() else {}
    rep["cpu_benchmark"] = res
    REPORT.write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
