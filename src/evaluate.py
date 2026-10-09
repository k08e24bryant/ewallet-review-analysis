"""Gold-set evaluation shared by the baseline (Phase 4) and IndoBERT (Phase 5).

Every metric is computed twice, as in Phase 3b:

    raw         on the 600 gold rows as sampled (50 per app x weak_label)
    reweighted  rows weighted by in_model_pool stratum size / 50, estimating
                performance on the pool of unique review texts

CIs come from a stratified bootstrap (resampling within app x weak_label strata).
The bootstrap is paired: all models are scored on the same resamples, so the
difference between two models gets its own CI.

Usage (evaluate a saved baseline pipeline on gold):
    uv run python -m src.evaluate --split gold
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.gold_report import GRID, SEQ_BLUE, SURFACE, TEXT_PRIMARY, TEXT_SECONDARY

# Categorical slots 1-3 of the reference palette (validated as a set)
SERIES3 = ["#2a78d6", "#eb6834", "#1baf7a"]


def load_config(path: Path) -> dict[str, Any]:
    """Load a YAML config."""
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_gold(cfg: dict[str, Any]) -> pd.DataFrame:
    """Gold labels joined with their text and the population weight (pool stratum size / gold stratum size)."""
    gold = pd.read_parquet(cfg["gold_path"])
    pool = pd.read_parquet(cfg["pool_path"])
    text = pool.set_index("reviewId")[["model_text", "text_clean"]]
    gold = gold.join(text, on="reviewId")
    if gold["model_text"].isna().any():
        raise ValueError("some gold reviewIds are missing from the pool")
    sizes = pool.groupby(["app", "weak_label"]).size().rename("pool_n")
    gold = gold.join(sizes, on=["app", "weak_label"])
    gold["weight"] = gold["pool_n"] / gold.groupby(["app", "weak_label"])["reviewId"].transform("size")
    return gold.reset_index(drop=True)


# ---------------------------------------------------------------- metrics

def _cm(yt: np.ndarray, yp: np.ndarray, w: np.ndarray, k: int) -> np.ndarray:
    return np.bincount(yt * k + yp, weights=w, minlength=k * k).reshape(k, k)


def _prf(cm: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tp = np.diag(cm)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(cm.sum(0) > 0, tp / cm.sum(0), 0.0)
        r = np.where(cm.sum(1) > 0, tp / cm.sum(1), 0.0)
        f = np.where(p + r > 0, 2 * p * r / (p + r), 0.0)
    return p, r, f


def metric_set(
    yt: np.ndarray, yp: np.ndarray, w: np.ndarray, labels: list[str], apps: np.ndarray, app_names: list[str],
    neg_idx: int,
) -> dict[str, float]:
    """Flat dict of all metrics for one model on one (possibly resampled) sample."""
    k = len(labels)
    out: dict[str, float] = {}
    p, r, f = _prf(_cm(yt, yp, w, k))
    out["macro_f1"] = float(f.mean())
    for i, l in enumerate(labels):
        out[f"{l}_precision"], out[f"{l}_recall"], out[f"{l}_f1"] = float(p[i]), float(r[i]), float(f[i])

    def binary(mask: np.ndarray) -> tuple[float, float, float]:
        bt, bp = (yt[mask] == neg_idx).astype(int), (yp[mask] == neg_idx).astype(int)
        bp_, br_, bf_ = _prf(_cm(bt, bp, w[mask], 2))
        return float(bp_[1]), float(br_[1]), float(bf_[1])

    all_mask = np.ones(len(yt), dtype=bool)
    out["bin_neg_precision"], out["bin_neg_recall"], out["bin_neg_f1"] = binary(all_mask)
    for a in app_names:
        out[f"bin_neg_f1_{a}"] = binary(apps == a)[2]
    return out


def evaluate_models(gold: pd.DataFrame, preds: dict[str, np.ndarray], cfg: dict[str, Any], compare: tuple[str, str]) -> dict[str, Any]:
    """Raw + reweighted metrics with stratified, paired bootstrap CIs for each model and one pairwise difference."""
    labels = cfg["labels"]
    idx = {l: i for i, l in enumerate(labels)}
    yt = gold["gold_label"].map(idx).to_numpy()
    apps = gold["app"].to_numpy()
    app_names = sorted(set(apps))
    neg = idx[cfg["binary_positive"]]
    w_raw = np.ones(len(gold))
    w_rew = gold["weight"].to_numpy(dtype=float)
    yp = {name: pd.Series(p).map(idx).to_numpy() for name, p in preds.items()}
    if any(np.isnan(v.astype(float)).any() for v in yp.values()):
        raise ValueError("predictions contain labels outside the label set")

    def all_metrics(rows: np.ndarray) -> dict[str, dict[str, dict[str, float]]]:
        return {
            mode: {
                name: metric_set(yt[rows], p[rows], w[rows], labels, apps[rows], app_names, neg)
                for name, p in yp.items()
            }
            for mode, w in (("raw", w_raw), ("reweighted", w_rew))
        }

    point = all_metrics(np.arange(len(gold)))

    bs = cfg["bootstrap"]
    rng = np.random.default_rng(cfg["seed"])
    strata = [g.index.to_numpy() for _, g in gold.groupby(["app", "weak_label"])]
    a, b = compare
    samples: dict[str, list[float]] = {}
    for _ in range(bs["n"]):
        rows = np.concatenate([rng.choice(s, size=len(s), replace=True) for s in strata])
        m = all_metrics(rows)
        for mode, models in m.items():
            for name, metrics in models.items():
                for key, v in metrics.items():
                    samples.setdefault(f"{mode}|{name}|{key}", []).append(v)
            for key in ("macro_f1", "bin_neg_f1"):
                samples.setdefault(f"{mode}|diff|{key}", []).append(models[a][key] - models[b][key])
    q = [(1 - bs["ci"]) / 2 * 100, (1 + bs["ci"]) / 2 * 100]
    ci = {k: [float(x) for x in np.percentile(v, q)] for k, v in samples.items()}

    report: dict[str, Any] = {"models": {}, "difference": {}}
    for mode in ("raw", "reweighted"):
        for name, metrics in point[mode].items():
            report["models"].setdefault(name, {})[mode] = {
                key: {"value": v, "ci95": ci[f"{mode}|{name}|{key}"]} for key, v in metrics.items()
            }
        report["difference"][mode] = {
            "models": f"{a} minus {b}",
            **{
                key: {
                    "value": point[mode][a][key] - point[mode][b][key],
                    "ci95": ci[f"{mode}|diff|{key}"],
                    "share_of_resamples_above_0": float(np.mean(np.array(samples[f"{mode}|diff|{key}"]) > 0)),
                }
                for key in ("macro_f1", "bin_neg_f1")
            },
        }
    report["confusion_matrices"] = {
        name: {
            "rows": "gold", "cols": "predicted", "labels": labels,
            "raw_counts": _cm(yt, p, w_raw, len(labels)).astype(int).tolist(),
            "reweighted_row_share": _row_share(_cm(yt, p, w_rew, len(labels))).round(4).tolist(),
        }
        for name, p in yp.items()
    }
    return report


def _row_share(cm: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.nan_to_num(cm / cm.sum(axis=1, keepdims=True))


# ---------------------------------------------------------------- errors

def sample_errors(gold: pd.DataFrame, pred: np.ndarray, n: int, seed: int, is_error: np.ndarray | None = None) -> pd.DataFrame:
    """Up to ``n`` misclassified gold rows, spread evenly over (gold -> predicted) error types.

    ``is_error`` overrides the default (gold_label != predicted), e.g. for binary
    predictions scored against 3-class gold labels.
    """
    df = gold.assign(predicted=pred)
    wrong = df[df["gold_label"] != df["predicted"] if is_error is None else is_error].copy()
    wrong["error_type"] = wrong["gold_label"] + "->" + wrong["predicted"]
    groups = {t: g.sample(frac=1, random_state=seed) for t, g in wrong.groupby("error_type")}
    picked: list[pd.DataFrame] = []
    remaining = n
    # Round-robin so rare error types are represented; large types fill the rest.
    while remaining > 0 and any(len(g) for g in groups.values()):
        for t in sorted(groups, key=lambda t: len(groups[t])):
            if remaining == 0 or groups[t].empty:
                continue
            picked.append(groups[t].iloc[:1])
            groups[t] = groups[t].iloc[1:]
            remaining -= 1
    out = pd.concat(picked).sort_values(["error_type", "gold_id"])
    out["notes"] = out["notes"].map(lambda ts: ",".join(ts))
    return out


# ---------------------------------------------------------------- figures

def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=10)


def plot_binary_comparison(report: dict[str, Any], model_names: dict[str, str], path: Path, title: str) -> None:
    """Grouped bars: reweighted complaint-detection F1 (negative vs rest) per app and overall, with CIs."""
    import matplotlib.pyplot as plt

    first = next(iter(report["models"].values()))["reweighted"]
    apps = sorted(k.removeprefix("bin_neg_f1_") for k in first if k.startswith("bin_neg_f1_"))
    groups = apps + ["all"]
    x = np.arange(len(groups))
    n = len(model_names)
    width = 0.8 / n
    fig, ax = plt.subplots(figsize=(10, 4.8), facecolor=SURFACE)
    _style(ax)
    for s, ((key, label), color) in enumerate(zip(model_names.items(), SERIES3)):
        m = report["models"][key]["reweighted"]
        keys = [f"bin_neg_f1_{a}" for a in apps] + ["bin_neg_f1"]
        vals = np.array([m[k]["value"] for k in keys])
        lo = np.array([m[k]["ci95"][0] for k in keys])
        hi = np.array([m[k]["ci95"][1] for k in keys])
        pos = x + (s - (n - 1) / 2) * width
        ax.bar(pos, vals, width * 0.92, color=color, label=label, edgecolor=SURFACE, linewidth=2, zorder=2)
        ax.errorbar(pos, vals, yerr=[vals - lo, hi - vals], fmt="none", ecolor=TEXT_SECONDARY,
                    elinewidth=1, capsize=2.5, zorder=3)
        for xi, v, h in zip(pos, vals, hi):
            ax.text(xi, h + 0.015, f"{v:.2f}", ha="center", va="bottom", color=TEXT_PRIMARY, fontsize=8)
    ax.set_xticks(x, [g if g != "all" else "all apps" for g in groups])
    ax.set_ylim(0, 1.1)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_ylabel("F1 for 'negative' (reweighted)", color=TEXT_SECONDARY)
    ax.set_title(title, color=TEXT_PRIMARY, fontsize=12, loc="left")
    ax.legend(frameon=False, labelcolor=TEXT_PRIMARY, loc="upper left", ncol=n)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_confusion(cm_entry: dict[str, Any], path: Path, title: str) -> None:
    """Two panels for one model on gold: raw counts and reweighted row shares."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    labels = cm_entry["labels"]
    raw = np.array(cm_entry["raw_counts"])
    raw_row = _row_share(raw.astype(float))
    w_row = np.array(cm_entry["reweighted_row_share"])
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), facecolor=SURFACE)
    panels = [
        (axes[0], raw_row, lambda i, j: f"{raw[i, j]}\n{raw_row[i, j]:.0%}", "Gold set as sampled (n=600)\ncount and row share"),
        (axes[1], w_row, lambda i, j: f"{w_row[i, j]:.0%}", "Reweighted to the model pool\nrow share (estimate)"),
    ]
    for ax, shade, text, sub in panels:
        ax.imshow(shade, cmap=cmap, vmin=0, vmax=1)
        for i in range(len(labels)):
            for j in range(len(labels)):
                ink = "white" if shade[i, j] >= 0.55 else TEXT_PRIMARY
                ax.text(j, i, text(i, j), ha="center", va="center", color=ink, fontsize=11)
        ax.set_xticks(range(len(labels)), labels)
        ax.set_yticks(range(len(labels)), labels)
        ax.set_xlabel("Predicted", color=TEXT_SECONDARY)
        ax.set_ylabel("Gold label", color=TEXT_SECONDARY)
        ax.set_title(sub, color=TEXT_PRIMARY, fontsize=11, loc="left")
        ax.tick_params(colors=TEXT_SECONDARY, length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_xticks(np.arange(-0.5, len(labels)), minor=True)
        ax.set_yticks(np.arange(-0.5, len(labels)), minor=True)
        ax.grid(which="minor", color=SURFACE, linewidth=2)
        ax.tick_params(which="minor", length=0)
    fig.suptitle(title, x=0.02, ha="left", color=TEXT_PRIMARY, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def format_table(report: dict[str, Any], model_names: dict[str, str]) -> str:
    """Comparison table (reweighted value [CI] | raw value) for the headline metrics."""
    first = next(iter(report["models"].values()))["reweighted"]
    keys = ["macro_f1", "negative_f1", "neutral_f1", "positive_f1", "bin_neg_f1", "bin_neg_precision", "bin_neg_recall"]
    keys += sorted(k for k in first if k.startswith("bin_neg_f1_"))
    lines = [f"{'metric':<22}" + "".join(f"{label:>34}" for label in model_names.values())]
    for k in keys:
        cells = []
        for name in model_names:
            r = report["models"][name]["reweighted"][k]
            raw = report["models"][name]["raw"][k]["value"]
            cells.append(f"{r['value']:.3f} [{r['ci95'][0]:.2f},{r['ci95'][1]:.2f}] | {raw:.3f}")
        lines.append(f"{k:<22}" + "".join(f"{c:>34}" for c in cells))
    return "\n".join(lines)


def main() -> None:
    """Evaluate the saved baseline pipeline on gold (no training)."""
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=["gold"], default="gold")
    parser.add_argument("--config", type=Path, default=Path("configs/evaluate.yaml"))
    parser.add_argument("--baseline-config", type=Path, default=Path("configs/baseline.yaml"))
    args = parser.parse_args()

    import joblib

    cfg = load_config(args.config)
    bcfg = load_config(args.baseline_config)
    gold = load_gold(cfg)
    pipe = joblib.load(Path(bcfg["model_dir"]) / "pipeline.joblib")
    preds = {"star_rating": gold["weak_label"].to_numpy(), "tfidf_logreg": pipe.predict(gold[bcfg["text_col"]])}
    rep = evaluate_models(gold, preds, cfg, compare=("tfidf_logreg", "star_rating"))
    print(format_table(rep, {"star_rating": "star rating", "tfidf_logreg": "TF-IDF + LogReg"}))
    print(json.dumps(rep["difference"], indent=2))


if __name__ == "__main__":
    main()
