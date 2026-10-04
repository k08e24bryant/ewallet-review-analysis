"""Phase 4: TF-IDF (word 1-2 + char_wb 3-5) + Logistic Regression baseline.

Tuning uses VAL macro-F1 only; the gold set is read only after the final
model is fixed, for evaluation.

Usage:
    uv run python -m src.train_baseline --config configs/baseline.yaml
    uv run python -m src.train_baseline --config configs/baseline.yaml --smoke
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
import yaml
from joblib import Parallel, delayed
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import FeatureUnion, Pipeline

from src import evaluate as ev

logger = logging.getLogger("baseline")

SMOKE = {"train_rows": 2000, "val_rows": 500, "C": [1.0], "min_df": [2], "bootstrap_n": 50, "out": "models/baseline_smoke"}


def load_config(path: Path) -> dict[str, Any]:
    """Load the baseline YAML config."""
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_features(cfg: dict[str, Any], min_df: int) -> FeatureUnion:
    """Word + char TF-IDF vectorizers combined side by side."""
    parts = []
    for name, p in cfg["features"].items():
        parts.append((name, TfidfVectorizer(
            analyzer=p["analyzer"], ngram_range=tuple(p["ngram_range"]), sublinear_tf=p["sublinear_tf"],
            lowercase=p["lowercase"], min_df=min_df, dtype=np.float32,
        )))
    return FeatureUnion(parts)


def make_clf(cfg: dict[str, Any], C: float) -> LogisticRegression:
    """Logistic regression with balanced class weights."""
    m = cfg["model"]
    return LogisticRegression(
        C=C, class_weight=m["class_weight"], max_iter=m["max_iter"], solver=m["solver"], random_state=cfg["seed"]
    )


def _fit_one(cfg: dict[str, Any], C: float, Xtr, ytr, Xva, yva) -> tuple[float, LogisticRegression, float, int]:
    t = time.time()
    clf = make_clf(cfg, C).fit(Xtr, ytr)
    f1 = f1_score(yva, clf.predict(Xva), average="macro")
    return f1, clf, time.time() - t, int(clf.n_iter_.max())


def tune(cfg: dict[str, Any], train: pd.DataFrame, val: pd.DataFrame) -> tuple[Pipeline, list[dict[str, Any]], dict[str, Any]]:
    """Grid over min_df (refit vectorizers) x C (parallel); pick the best VAL macro-F1."""
    text, target = cfg["text_col"], cfg["target_col"]
    results, best = [], None
    for min_df in cfg["grid"]["min_df"]:
        t = time.time()
        feats = make_features(cfg, min_df).fit(train[text])
        Xtr, Xva = feats.transform(train[text]), feats.transform(val[text])
        logger.info("min_df=%d: %d features (%.0fs)", min_df, Xtr.shape[1], time.time() - t)
        fits = Parallel(n_jobs=min(len(cfg["grid"]["C"]), 8))(
            delayed(_fit_one)(cfg, C, Xtr, train[target], Xva, val[target]) for C in cfg["grid"]["C"]
        )
        for C, (f1, clf, secs, n_iter) in zip(cfg["grid"]["C"], fits):
            row = {"min_df": min_df, "C": C, "val_macro_f1": float(f1), "n_features": int(Xtr.shape[1]),
                   "fit_seconds": round(secs, 1), "n_iter": n_iter}
            results.append(row)
            logger.info("  C=%-5g val macro-F1=%.4f (%ss, %d iter)", C, f1, row["fit_seconds"], n_iter)
            # Ties -> fewer features (higher min_df) then stronger regularization (lower C).
            if best is None or f1 > best[0] + 1e-9:
                best = (f1, row, feats, clf)
    f1, row, feats, clf = best
    if row["n_iter"] >= cfg["model"]["max_iter"]:
        logger.warning("best model hit max_iter=%d; consider raising it", cfg["model"]["max_iter"])
    return Pipeline([("features", feats), ("clf", clf)]), results, row


def top_features(pipe: Pipeline, n: int) -> dict[str, dict[str, list[list]]]:
    """Top-n positive and negative coefficients per class (feature names prefixed word__/char__)."""
    names = pipe.named_steps["features"].get_feature_names_out()
    clf = pipe.named_steps["clf"]
    out = {}
    for i, label in enumerate(clf.classes_):
        coef = clf.coef_[i]
        order = np.argsort(coef)
        out[label] = {
            "positive": [[names[j], round(float(coef[j]), 3)] for j in order[::-1][:n]],
            "negative": [[names[j], round(float(coef[j]), 3)] for j in order[:n]],
        }
    return out


def main() -> None:
    """CLI entry point."""
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.yaml"))
    parser.add_argument("--smoke", action="store_true", help="tiny subsample, 1 grid point, outputs under models/baseline_smoke")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    cfg = load_config(args.config)
    ecfg = ev.load_config(Path(cfg["eval"]["config"]))
    seed = cfg["seed"]
    np.random.seed(seed)
    splits = Path(cfg["splits_dir"])
    train = pd.read_parquet(splits / "train.parquet")
    val = pd.read_parquet(splits / "val.parquet")

    out_model = Path(cfg["model_dir"])
    report_path, errors_path = Path(cfg["eval"]["report_path"]), Path(cfg["eval"]["errors_path"])
    tuning_path = Path(cfg["tuning_report_path"])
    fig_dir = Path(ecfg["figures_dir"])
    if args.smoke:
        train = train.sample(n=SMOKE["train_rows"], random_state=seed)
        val = val.sample(n=SMOKE["val_rows"], random_state=seed)
        cfg["grid"] = {"C": SMOKE["C"], "min_df": SMOKE["min_df"]}
        ecfg["bootstrap"]["n"] = SMOKE["bootstrap_n"]
        out_model = fig_dir = Path(SMOKE["out"])
        report_path, errors_path = out_model / "report.json", out_model / "errors.csv"
        tuning_path = out_model / "tuning.json"
    logger.info("seed=%d smoke=%s train=%d val=%d grid=%s", seed, args.smoke, len(train), len(val), cfg["grid"])

    t0 = time.time()
    pipe, results, best = tune(cfg, train, val)
    logger.info("best: %s (tuning %.0fs)", best, time.time() - t0)
    out_model.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, out_model / "pipeline.joblib")
    meta = {
        "seed": seed, "best": best, "grid": cfg["grid"], "train_rows": len(train), "val_rows": len(val),
        "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__},
    }
    (out_model / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    tuning_path.parent.mkdir(parents=True, exist_ok=True)
    tuning_path.write_text(json.dumps({"seed": seed, "selection": "val macro-F1", "results": results, "best": best}, indent=2), encoding="utf-8")

    # ---- gold evaluation (model is fixed from here on)
    gold = ev.load_gold(ecfg)
    majority = train[cfg["target_col"]].mode()[0]
    preds = {
        "majority": np.full(len(gold), majority, dtype=object),
        "star_rating": gold["weak_label"].to_numpy(),
        "tfidf_logreg": pipe.predict(gold[cfg["text_col"]]),
    }
    logger.info("evaluating on gold (bootstrap n=%d)", ecfg["bootstrap"]["n"])
    rep = ev.evaluate_models(gold, preds, ecfg, compare=("tfidf_logreg", "star_rating"))
    names = {"majority": f"majority ({majority})", "star_rating": "star rating", "tfidf_logreg": "TF-IDF + LogReg"}

    val_pred = pipe.predict(val[cfg["text_col"]])
    val_f1 = f1_score(val[cfg["target_col"]], val_pred, average=None, labels=cfg["labels"])
    errors = ev.sample_errors(gold, preds["tfidf_logreg"], cfg["eval"]["n_errors"], seed)
    errors[["gold_id", "text_clean", "gold_label", "predicted", "score", "notes"]].to_csv(errors_path, index=False, encoding="utf-8-sig")
    feats = top_features(pipe, cfg["eval"]["top_features"])

    report = {
        "seed": seed,
        "weighting": "raw = gold as sampled; reweighted = weighted to in_model_pool stratum sizes (quote for population claims)",
        "bootstrap": ecfg["bootstrap"],
        "majority_class": majority,
        "selected_hyperparameters": {"C": best["C"], "min_df": best["min_df"]},
        "val": {"macro_f1": best["val_macro_f1"], "per_class_f1": dict(zip(cfg["labels"], map(float, val_f1)))},
        "gold": rep,
        "error_types": errors["error_type"].value_counts().to_dict(),
        "all_error_type_counts": pd.Series(
            [f"{g}->{p}" for g, p in zip(gold["gold_label"], preds["tfidf_logreg"]) if g != p]
        ).value_counts().to_dict(),
        "top_features": feats,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    fig_dir.mkdir(parents=True, exist_ok=True)
    ev.plot_binary_comparison(
        rep, names, fig_dir / cfg["eval"]["figures"]["comparison"],
        "Complaint detection on gold: text model vs star rating (95% bootstrap CI)",
    )
    ev.plot_confusion(rep["confusion_matrices"]["tfidf_logreg"], fig_dir / cfg["eval"]["figures"]["confusion"],
                      "TF-IDF + LogReg on gold (3-class)")
    logger.info("wrote %s, %s, %s, figures in %s (total %.0fs)", out_model, report_path, errors_path, fig_dir, time.time() - t0)

    print("\n== tuning (val macro-F1) ==")
    print(pd.DataFrame(results).pivot(index="min_df", columns="C", values="val_macro_f1").round(4).to_string())
    print(f"selected: C={best['C']} min_df={best['min_df']}  val per-class F1:",
          {k: round(v, 3) for k, v in report["val"]["per_class_f1"].items()})
    print("\n== gold: reweighted value [95% CI] | raw value ==")
    print(ev.format_table(rep, names))
    d = rep["difference"]
    for mode in ("reweighted", "raw"):
        x = d[mode]["bin_neg_f1"]
        print(f"TF-IDF minus star rating, binary neg F1 ({mode}): {x['value']:+.3f} "
              f"[{x['ci95'][0]:+.3f}, {x['ci95'][1]:+.3f}], P(diff>0)={x['share_of_resamples_above_0']:.3f}")
    print("\nerror types (all gold errors):", report["all_error_type_counts"])
    def show(name: str) -> str:
        kind, gram = name.split("__", 1)
        return f"{kind[0]}:{gram!r}"

    for label, f in feats.items():
        print(f"\n[{label}] +", ", ".join(show(n) for n, _ in f["positive"]))
        print(f"[{label}] -", ", ".join(show(n) for n, _ in f["negative"]))


if __name__ == "__main__":
    main()
