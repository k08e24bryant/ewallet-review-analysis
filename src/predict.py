"""Phase 6a step 1: PRIMARY complaint predictions for every review.

PRIMARY = IndoBERT spec_3class/none + "star 1-2 OR model" (fixed in Phase 5c,
evaluated once on gold in Phase 5d). Each unique model_text is scored once
(fp32, argmax, as in Phase 5d) and mapped back to all rows. No labels are read.

Output (data/processed/predictions.parquet), one row per review:
    reviewId, app, score, in_model_pool, pred_label, p_negative, p_neutral,
    p_positive, is_complaint, flagged_by (star+model | star rule | model | -)

Usage:
    uv run python -m src.predict --config configs/predict.yaml --smoke
    uv run python -m src.predict --config configs/predict.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger("predict")


def load_config(path: Path) -> dict[str, Any]:
    """Load a YAML config."""
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def predict_proba(model_dir: Path, texts: list[str], max_length: int, batch_size: int, device: str) -> tuple[np.ndarray, list[str]]:
    """Softmax probabilities (fp32) for texts, scored in length-sorted batches; returns (probs, label names)."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device).eval()
    labels = [model.config.id2label[i] for i in range(model.config.num_labels)]
    lengths = np.array([len(t) for t in texts])
    order = np.argsort(lengths, kind="stable")
    probs = np.zeros((len(texts), len(labels)), dtype=np.float32)
    t0 = time.time()
    with torch.no_grad():
        for b, i in enumerate(range(0, len(texts), batch_size)):
            idx = order[i : i + batch_size]
            enc = tok([texts[j] for j in idx], truncation=True, max_length=max_length, padding=True, return_tensors="pt")
            logits = model(**{k: v.to(device) for k, v in enc.items()}).logits.float()
            probs[idx] = torch.softmax(logits, dim=-1).cpu().numpy()
            if b % 200 == 0:
                logger.info("  %d / %d texts (%.0fs)", i + len(idx), len(texts), time.time() - t0)
    return probs, labels


def share_table(df: pd.DataFrame) -> dict[str, Any]:
    """Complaint share per app and overall, for all rows and for unique texts (in_model_pool)."""
    out = {}
    for name, part in (("all_rows", df), ("unique_texts", df[df["in_model_pool"]])):
        g = part.groupby("app")["is_complaint"].agg(["size", "sum", "mean"])
        out[name] = {
            **{a: {"rows": int(r["size"]), "complaints": int(r["sum"]), "share": round(float(r["mean"]), 4)} for a, r in g.iterrows()},
            "all": {"rows": int(len(part)), "complaints": int(part["is_complaint"].sum()),
                    "share": round(float(part["is_complaint"].mean()), 4)},
        }
    return out


def main() -> None:
    """CLI entry point."""
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path("configs/predict.yaml"))
    parser.add_argument("--smoke", action="store_true", help="random subset of rows, outputs under models/")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    for noisy in ("httpx", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    import torch

    from src.dev_select import predict_indobert

    cfg = load_config(args.config)
    seed = cfg["seed"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_path, report_path = Path(cfg["output_path"]), Path(cfg["report_path"])
    cols = ["reviewId", "app", "score", "in_model_pool", cfg["text_col"]]
    df = pd.read_parquet(cfg["input_path"], columns=cols)
    if args.smoke:
        df = df.sample(n=cfg["smoke"]["rows"], random_state=seed)
        out_path = Path(cfg["smoke"]["out_dir"]) / "predictions.parquet"
        report_path = Path(cfg["smoke"]["out_dir"]) / "report.json"
    logger.info("seed=%d smoke=%s device=%s rows=%d model=%s", seed, args.smoke, device, len(df), cfg["model_dir"])

    texts = df[cfg["text_col"]].drop_duplicates().tolist()
    t0 = time.time()
    probs, labels = predict_proba(Path(cfg["model_dir"]), texts, cfg["max_length"], cfg["batch_size"], device)
    secs = time.time() - t0
    logger.info("scored %d unique texts in %.0fs", len(texts), secs)

    # Consistency: same texts in the Phase 5d order and batch size must give the same labels
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(texts), size=min(cfg["consistency_check_n"], len(texts)), replace=False)
    ref = predict_indobert(Path(cfg["model_dir"]), [texts[i] for i in pick], device, 64, cfg["max_length"])
    agree = float(np.mean(np.array(labels, dtype=object)[probs[pick].argmax(1)] == ref))
    logger.info("consistency vs Phase 5d batching: %.4f label agreement on %d texts", agree, len(pick))

    by_text = pd.DataFrame(probs, columns=[f"p_{l}" for l in labels]).assign(**{cfg["text_col"]: texts})
    by_text["pred_label"] = np.array(labels, dtype=object)[probs.argmax(1)]
    out = df.merge(by_text, on=cfg["text_col"], how="left", validate="many_to_one")
    assert out["pred_label"].notna().all() and len(out) == len(df)
    star = out["score"] <= cfg["star_negative_max"]
    model_neg = out["pred_label"] == "negative"
    out["is_complaint"] = star | model_neg
    out["flagged_by"] = np.select([star & model_neg, star, model_neg], ["star+model", "star rule", "model"], "-")
    out = out[["reviewId", "app", "score", "in_model_pool", "pred_label", *[f"p_{l}" for l in labels],
               "is_complaint", "flagged_by"]]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)

    report = {
        "seed": seed, "model_id": cfg["model_id"], "model_dir": cfg["model_dir"], "device": device,
        "rows": int(len(out)), "unique_texts_scored": len(texts), "seconds": round(secs, 1),
        "consistency_vs_phase5d_batching": {"n": int(len(pick)), "label_agreement": agree},
        "complaint_share": share_table(out),
        "flagged_by_all_rows": out["flagged_by"].value_counts().to_dict(),
        "pred_label_all_rows": out["pred_label"].value_counts().to_dict(),
        "note": ("Predicted shares, not validated rates. On gold (Phase 5d) PRIMARY had reweighted precision "
                 "0.973 and recall 0.876, so predicted shares likely understate true complaint shares."),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("wrote %s and %s", out_path, report_path)

    print("\n== complaint share (PRIMARY) ==")
    for name, t in report["complaint_share"].items():
        print(name)
        for app, v in t.items():
            print(f"   {app:<10} {v['complaints']:>7} / {v['rows']:>7} = {v['share']:.1%}")
    print("flagged_by (all rows):", report["flagged_by_all_rows"])
    print("consistency:", report["consistency_vs_phase5d_batching"])


if __name__ == "__main__":
    main()
