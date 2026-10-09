"""Phase 5d: evaluate the fixed models on gold ONCE (no selection or tuning after this).

Options (configs/gold_eval.yaml): star rating alone, Phase 4 TF-IDF (spec),
IndoBERT spec_3class/balanced, PRIMARY (IndoBERT spec_3class/none + "star 1-2
OR model", fixed in Phase 5c), and TF-IDF binary_3star_negative + rule (comparison).

Same method as Phase 4: raw and reweighted (pool stratum size / gold stratum
size), stratified bootstrap paired across options. Also: the 4-5 star complaint
breakdown ("does IndoBERT copy stars?"), PRIMARY error export, intra-annotator
agreement on the blind relabel set, and a W&B summary run.

Usage:
    uv run python -m src.gold_eval --smoke     # whole pipeline on DEV labels; gold not read
    uv run python -m src.gold_eval             # the single gold run
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src import dev_select as ds
from src import evaluate as ev
from src.gold import parse_tags, read_sheet
from src.gold_report import GRID, SERIES, SURFACE, TEXT_PRIMARY, TEXT_SECONDARY

logger = logging.getLogger("gold_eval")


def load_config(path: Path) -> dict[str, Any]:
    """Load a YAML config."""
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _json_default(o: Any) -> Any:
    return o.item() if hasattr(o, "item") else str(o)


def as_tags(v: Any) -> list[str]:
    """Notes as a list of tags (gold stores arrays, dev stores comma-separated strings)."""
    if isinstance(v, (list, tuple, np.ndarray)):
        return [str(t) for t in v]
    return parse_tags(v)


# ---------------------------------------------------------------- data

def load_frame(cfg: dict[str, Any], smoke: bool) -> pd.DataFrame:
    """Gold (or, in smoke mode, dev) rows with gold_label, text, notes tags, and weight."""
    if smoke:
        dev, _ = ds.load_dev(load_config(Path(cfg["dev_select_config"])))
        df = dev.rename(columns={"label": "gold_label", "dev_id": "gold_id"})
    else:
        df = ev.load_gold(load_config(Path(cfg["evaluate_config"])))
    df = df.copy()
    df["notes"] = df["notes"].map(as_tags)
    return df.reset_index(drop=True)


def predict_all(df: pd.DataFrame, cfg: dict[str, Any], dcfg: dict[str, Any], device: str) -> tuple[dict, dict]:
    """Return ({option_id: option dict for ds.evaluate_options}, {model_id: raw label predictions})."""
    import joblib

    icfg = load_config(Path(dcfg["indobert"]["config"]))
    bcfg = load_config(Path(dcfg["baseline"]["config"]))
    star_neg = df["score"].to_numpy() <= dcfg["star_negative_max"]
    raw: dict[str, np.ndarray] = {}

    def model_pred(mid: str) -> np.ndarray:
        if mid in raw:
            return raw[mid]
        kind, scheme, *rest = mid.split("/")
        if kind == "indobert":
            path = Path(icfg["output_root"]) / scheme / rest[0] / "final"
            logger.info("predicting with %s", path)
            raw[mid] = ds.predict_indobert(path, df[dcfg["text_col"]].tolist(), device,
                                           dcfg["indobert"]["batch_size"], icfg["max_length"])
        else:
            d = Path(bcfg["model_dir"]) if scheme == "spec_3class" else Path(bcfg["scheme_model_dir"]) / scheme
            raw[mid] = joblib.load(d / "pipeline.joblib").predict(df[bcfg["text_col"]]).astype(object)
        return raw[mid]

    options: dict[str, dict[str, Any]] = {}
    for oid, meta in cfg["options"].items():
        if oid == "reference/star_1_2":
            pred, pred3 = star_neg, df["weak_label"].to_numpy(dtype=object)  # weak label as a 3-class predictor
        else:
            mid, rule = oid.split("|")
            p = model_pred(mid)
            pred = (p == "negative") | (star_neg if rule == "star_1_2_or_model" else False)
            pred3 = p if (rule == "model_only" and "spec_3class" in mid) else None
        options[oid] = {**meta, "candidate": False, "pred": np.asarray(pred, dtype=bool), "pred3": pred3}
    return options, raw


# ---------------------------------------------------------------- analyses

def summarize(point: dict, samples: list[dict], options: dict, q: list[float]) -> dict[str, Any]:
    """{option: {mode: {metric: {value, ci95}}}} from point estimates and bootstrap samples."""
    out = {}
    for oid in options:
        out[oid] = {
            mode: {k: {"value": v, "ci95": [float(x) for x in np.nanpercentile([s[mode][oid][k] for s in samples], q)]}
                   for k, v in point[mode][oid].items()}
            for mode in ("raw", "reweighted")
        }
    return out


def differences(point: dict, samples: list[dict], cfg: dict[str, Any], q: list[float]) -> list[dict[str, Any]]:
    """Paired bootstrap differences (first minus second) for the configured pairs and metrics."""
    out = []
    for a, b in cfg["differences"]:
        entry: dict[str, Any] = {"pair": f"{a} minus {b}"}
        for mode in ("raw", "reweighted"):
            entry[mode] = {}
            for k in cfg["diff_metrics"]:
                d = np.array([s[mode][a][k] - s[mode][b][k] for s in samples])
                entry[mode][k] = {"value": point[mode][a][k] - point[mode][b][k],
                                  "ci95": [float(x) for x in np.percentile(d, q)],
                                  "share_of_resamples_above_0": float(np.mean(d > 0))}
        out.append(entry)
    return out


def confusion(df: pd.DataFrame, options: dict, labels: list[str]) -> dict[str, Any]:
    """3-class matrices for 3-class options, 2x2 complaint matrices for every option."""
    w = df["weight"].to_numpy(dtype=float)
    out: dict[str, Any] = {}
    for oid, o in options.items():
        entry = {}
        if o["pred3"] is not None:
            idx = {l: i for i, l in enumerate(labels)}
            yt, yp = df["gold_label"].map(idx).to_numpy(), pd.Series(o["pred3"]).map(idx).to_numpy()
            entry["3class"] = {
                "rows": "gold", "cols": "predicted", "labels": labels,
                "raw_counts": ev._cm(yt, yp, np.ones(len(df)), 3).astype(int).tolist(),
                "reweighted_row_share": ev._row_share(ev._cm(yt, yp, w, 3)).round(4).tolist(),
            }
        yt = (df["gold_label"] == "negative").to_numpy().astype(int)
        yp = o["pred"].astype(int)
        entry["binary"] = {
            "rows": "gold", "cols": "predicted", "labels": ["non_negative", "negative"],
            "raw_counts": ev._cm(yt, yp, np.ones(len(df)), 2).astype(int).tolist(),
            "reweighted_row_share": ev._row_share(ev._cm(yt, yp, w, 2)).round(4).tolist(),
        }
        out[oid] = entry
    return out


def star45_breakdown(df: pd.DataFrame, options: dict, cfg: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Every gold 4-5 star complaint with caught/missed per option, plus counts and overlaps."""
    m = ((df["score"] >= 4) & (df["gold_label"] == "negative")).to_numpy()
    rows = df.loc[m, ["gold_id", "app", "score", "text_clean", "notes", "weight"]].copy()
    rows["notes"] = rows["notes"].map(",".join)
    for oid, o in options.items():
        rows[o["role"]] = np.where(o["pred"][m], "caught", "missed")
    w = rows.pop("weight").to_numpy()
    roles = [o["role"] for o in options.values()]
    caught = {r: {"raw": int((rows[r] == "caught").sum()),
                  "reweighted_share": float((w * (rows[r] == "caught")).sum() / w.sum())} for r in roles}
    p, c = rows["primary"] == "caught", rows["comparison"] == "caught"
    cells = {"both": p & c, "primary_only": p & ~c, "comparison_only": ~p & c, "neither": ~p & ~c}
    overlap = {k: {"raw": int(v.sum()), "reweighted_share": float((w * v).sum() / w.sum())} for k, v in cells.items()}
    by_app = rows.groupby("app")[["primary", "comparison"]].agg(lambda s: int((s == "caught").sum()))
    by_app["n"] = rows.groupby("app").size()
    tags = {}
    for r in ("primary", "comparison"):
        for status in ("caught", "missed"):
            t = pd.Series([x for n in rows.loc[rows[r] == status, "notes"] for x in n.split(",") if x], dtype=object)
            tags[f"{r}_{status}"] = t.value_counts().to_dict()
    rows = rows.sort_values(["primary", "comparison", "app", "gold_id"])
    summary = {"n_rows_raw": int(m.sum()), "caught_per_role": caught,
               "overlap_primary_vs_comparison": overlap, "per_app_caught": by_app.to_dict("index"),
               "notes_tags": tags, "csv": cfg["star45_path"]}
    return rows, summary


def export_errors(df: pd.DataFrame, options: dict, raw: dict, cfg: dict[str, Any], dcfg: dict[str, Any]) -> tuple[pd.DataFrame, dict]:
    """PRIMARY complaint errors (binary vs gold), sampled evenly over gold -> predicted types."""
    o = options[cfg["primary"]]
    mid = cfg["primary"].split("|")[0]
    pred_lab = np.where(o["pred"], "negative", "non_negative")
    is_err = (df["gold_label"] == "negative").to_numpy() != o["pred"]
    star = df["score"].to_numpy() <= dcfg["star_negative_max"]
    model_neg = raw[mid] == "negative"
    flag = np.select([star & model_neg, star, model_neg], ["star+model", "star rule", "model"], "-")
    frame = df.assign(model_3class=raw[mid], flagged_by=flag)
    errs = ev.sample_errors(frame, pred_lab, cfg["errors"]["n"], cfg["seed"], is_error=is_err)
    all_types = (frame.loc[is_err, "gold_label"] + "->" + pd.Series(pred_lab)[is_err]).value_counts().to_dict()
    cols = ["gold_id", "app", "score", "error_type", "gold_label", "predicted", "model_3class", "flagged_by",
            "text_clean", "notes"]
    return errs[cols], {"all_error_type_counts": all_types, "exported_error_types": errs["error_type"].value_counts().to_dict(),
                        "flagged_by_among_false_alarms": pd.Series(flag[is_err & o["pred"]]).value_counts().to_dict()}


def load_relabel(gcfg: dict[str, Any], pool: pd.DataFrame, drop_blank: bool = False) -> tuple[pd.DataFrame, list[str]]:
    """Validate the blind relabel workbook against its key (no gold labels read here).

    Blank labels fail validation unless ``drop_blank``; then those rows are dropped
    and returned as the second element.
    """
    rl = gcfg["relabel"]
    key = pd.read_parquet(rl["key_path"])
    sheet = read_sheet(Path(rl["xlsx_path"]), "relabel")
    errors = []
    if sheet["gold_id"].tolist() != key["blind_id"].tolist():
        errors.append("blind ids differ from relabel_key order")
    else:
        expected = key["reviewId"].map(pool.set_index("reviewId")["text_clean"])
        changed = [b for b, a, e in zip(sheet["gold_id"], sheet["text_clean"], expected) if a != e]
        if changed:
            errors.append(f"texts differ from export: {changed[:10]}")
    allowed = set(gcfg["labels"]) | {gcfg["invalid_label"]}
    blank = sheet.loc[sheet["label"].isna(), "gold_id"].tolist()
    if blank and not drop_blank:
        errors.append(f"blank labels: {blank}")
    bad = sheet[sheet["label"].notna() & ~sheet["label"].isin(allowed)]
    if len(bad):
        errors.append(f"unknown labels: {bad[['gold_id', 'label']].values.tolist()}")
    bad_tags = [(b, t) for b, n in zip(sheet["gold_id"], sheet["notes"]) for t in parse_tags(n) if t not in gcfg["note_tags"]]
    if bad_tags:
        errors.append(f"unknown note tags: {bad_tags}")
    if errors:
        raise ValueError("relabel workbook failed validation: " + "; ".join(errors))
    out = key.assign(relabel=sheet["label"].to_numpy(), relabel_notes=sheet["notes"].map(lambda v: ",".join(parse_tags(v))).to_numpy())
    out = out[out["relabel"].notna()].reset_index(drop=True)
    return out.join(pool.set_index("reviewId")["text_clean"], on="reviewId"), blank


def intra_annotator(rel: pd.DataFrame, gold_label: pd.Series, n_boot: int, seed: int, ci: float) -> dict[str, Any]:
    """Cohen's kappa + raw agreement, first (gold) vs blind second label; 3-class and complaint-binary."""
    from sklearn.metrics import cohen_kappa_score

    df = rel.assign(gold_label=rel["reviewId"].map(gold_label).to_numpy())
    if df["gold_label"].isna().any():
        raise ValueError("relabel rows missing from the gold labels")
    a, b = df["gold_label"].to_numpy(), df["relabel"].to_numpy()
    labs = sorted(set(a) | set(b))
    q = [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100]
    rng = np.random.default_rng(seed)
    idx = [rng.integers(0, len(df), len(df)) for _ in range(n_boot)]

    def stats(x: np.ndarray, y: np.ndarray, labels: list[str]) -> dict[str, Any]:
        k = float(cohen_kappa_score(x, y, labels=labels))
        boot_k = [cohen_kappa_score(x[i], y[i], labels=labels) if len(set(x[i]) | set(y[i])) > 1 else np.nan for i in idx]
        boot_a = [float(np.mean(x[i] == y[i])) for i in idx]
        return {"kappa": k, "kappa_ci95": [float(v) for v in np.nanpercentile(boot_k, q)],
                "agreement": float(np.mean(x == y)), "agreement_ci95": [float(v) for v in np.percentile(boot_a, q)]}

    bin_a = np.where(a == "negative", "negative", "non_negative")
    bin_b = np.where(b == "negative", "negative", "non_negative")
    dis = df[a != b].sort_values("blind_id")
    return {
        "n": int(len(df)), "sample": "random rows of the final gold set (50 drawn), relabeled blind (no score, no first label)",
        "three_class": stats(a, b, labs),
        "binary_complaint": stats(bin_a, bin_b, ["negative", "non_negative"]),
        "confusion": {"rows": "gold (first)", "cols": "relabel (second)", "labels": labs,
                      "counts": pd.crosstab(pd.Categorical(a, labs), pd.Categorical(b, labs), dropna=False).to_numpy().tolist()},
        "disagreements": dis[["blind_id", "gold_id", "app", "weak_label", "gold_label", "relabel", "relabel_notes", "text_clean"]]
        .to_dict("records"),
    }


# ---------------------------------------------------------------- figures

def plot_panels(results: dict, cfg: dict, panels: list[tuple[str, str]], ncols: int, path: Path, title: str, subtitle: str) -> None:
    """Small multiples: one panel per metric, one row per option; reweighted value with 95% CI."""
    import matplotlib.pyplot as plt

    opts = list(cfg["options"])
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols + 2.6, 0.42 * len(opts) * nrows + 1.2 * nrows + 0.9),
                             facecolor=SURFACE, sharey=True, squeeze=False)
    for ax, (key, name) in zip(axes.flat, panels):
        ax.set_facecolor(SURFACE)
        for yi, oid in enumerate(opts):
            m = results[oid]["reweighted"].get(key)
            if m is None:
                continue
            v, (lo, hi) = m["value"], m["ci95"]
            ref = cfg["options"][oid]["role"] == "reference"
            c = TEXT_SECONDARY if ref else SERIES[0]
            ax.plot([lo, hi], [yi, yi], color=c, linewidth=2, alpha=0.55, solid_capstyle="round", zorder=2)
            ax.plot(v, yi, marker="D" if ref else "o", markersize=8, color=c, markeredgecolor=SURFACE,
                    markeredgewidth=2, zorder=3)
            if oid == cfg["primary"]:
                ax.plot(v, yi, marker="o", markersize=15, markerfacecolor="none", markeredgecolor=TEXT_PRIMARY,
                        markeredgewidth=1.5, zorder=4)
                ax.text(v, yi - 0.38, f"{v:.2f}", ha="center", va="bottom", color=TEXT_PRIMARY, fontsize=9, fontweight="bold")
        ax.set_xlim(-0.03, 1.05)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_ylim(len(opts) - 0.5, -0.7)
        ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(colors=TEXT_SECONDARY, labelsize=9, length=0)
        ax.set_title(name, color=TEXT_PRIMARY, fontsize=10, loc="left")
    for ax in axes.flat[len(panels):]:
        ax.set_visible(False)
    axes[0, 0].set_yticks(range(len(opts)), [cfg["options"][o]["label"] for o in opts])
    fig.text(0.01, 0.99, title, color=TEXT_PRIMARY, fontsize=12, va="top")
    fig.text(0.01, 0.99 - 0.3 / fig.get_figheight(), subtitle, color=TEXT_SECONDARY, fontsize=9, va="top")
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.55 / fig.get_figheight()))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------- W&B

def log_wandb(report: dict[str, Any], dev_report: dict[str, Any], cfg: dict[str, Any], smoke: bool) -> str | None:
    """One summary run: dev + gold table for the five options, PRIMARY headline numbers, intra-annotator kappa."""
    import wandb

    from src.train_indobert import wandb_logged_in

    icfg = load_config(Path(load_config(Path(cfg["dev_select_config"]))["indobert"]["config"]))
    if not smoke and not wandb_logged_in():
        logger.warning("W&B not logged in; skipping summary run")
        return None
    metrics = ["bin_neg_f1", "bin_neg_precision", "bin_neg_recall", "recall_star_1-2", "recall_star_3", "recall_star_4-5",
               "false_alarm_star_4-5"]
    cols = ["split", "option", "role"] + [f"{m}_rew" for m in metrics] + ["bin_neg_f1_rew_lo", "bin_neg_f1_rew_hi", "bin_neg_f1_raw"]
    table = wandb.Table(columns=cols)
    for split, opts in (("dev", dev_report["options"]), ("gold", report["results"])):
        for oid, meta in cfg["options"].items():
            r = opts[oid]
            table.add_data(split, oid, meta["role"],
                           *[r["reweighted"][m]["value"] if m in r["reweighted"] else None for m in metrics],
                           *r["reweighted"]["bin_neg_f1"]["ci95"], r["raw"]["bin_neg_f1"]["value"])
    run = wandb.init(project=icfg["wandb"]["project"], entity=icfg["wandb"]["entity"], name=cfg["wandb"]["name"],
                     job_type=cfg["wandb"]["job_type"], tags=["phase5", "gold", "summary"],
                     config={"seed": cfg["seed"], "primary": cfg["primary"], "bootstrap": report["bootstrap"],
                             "gold_commit": report["provenance"]["git_head"]},
                     mode="disabled" if smoke else None)
    p = report["results"][cfg["primary"]]["reweighted"]
    ia = report["intra_annotator"]
    run.log({"phase5_dev_vs_gold": table})
    run.summary.update({
        "gold/primary_bin_neg_f1": p["bin_neg_f1"]["value"], "gold/primary_bin_neg_precision": p["bin_neg_precision"]["value"],
        "gold/primary_bin_neg_recall": p["bin_neg_recall"]["value"],
        **{f"gold/diff_{d['pair'].split(' minus ')[1]}_bin_neg_f1": d["reweighted"]["bin_neg_f1"]["value"] for d in report["differences"]},
        **({"intra_annotator/kappa_3class": ia["three_class"]["kappa"],
            "intra_annotator/kappa_binary": ia["binary_complaint"]["kappa"]} if ia else {}),
    })
    url = None if smoke else run.url
    run.finish()
    return url


# ---------------------------------------------------------------- main

def print_report(report: dict[str, Any], cfg: dict[str, Any]) -> None:
    """Console tables."""
    res = report["results"]
    print("\n== complaint detection (negative vs rest): reweighted value [95% CI] | raw ==")
    cols = ["bin_neg_f1", "bin_neg_precision", "bin_neg_recall", "recall_star_1-2", "recall_star_3", "recall_star_4-5",
            "false_alarm_star_4-5"]
    print(f"{'option':<44}" + "".join(f"{c:>24}" for c in cols))
    for oid, meta in cfg["options"].items():
        r = res[oid]
        cells = [f"{r['reweighted'][c]['value']:.3f} [{r['reweighted'][c]['ci95'][0]:.2f},{r['reweighted'][c]['ci95'][1]:.2f}]|{r['raw'][c]['value']:.2f}"
                 for c in cols]
        print(f"{meta['label']:<44}" + "".join(f"{c:>24}" for c in cells))
    apps = sorted(k.removeprefix("bin_neg_f1_") for k in res[cfg["primary"]]["reweighted"] if k.startswith("bin_neg_f1_"))
    print("\n== per-app complaint F1 (reweighted [CI]) ==")
    for oid, meta in cfg["options"].items():
        print(f"{meta['label']:<44}" + "  ".join(
            f"{a}: {res[oid]['reweighted'][f'bin_neg_f1_{a}']['value']:.3f} [{res[oid]['reweighted'][f'bin_neg_f1_{a}']['ci95'][0]:.2f},"
            f"{res[oid]['reweighted'][f'bin_neg_f1_{a}']['ci95'][1]:.2f}]" for a in apps))
    print("\n== 3-class (reweighted [CI] | raw) ==")
    for oid, meta in cfg["options"].items():
        r = res[oid]["reweighted"]
        if "macro_f1_3class" in r:
            print(f"{meta['label']:<44}" + "  ".join(
                f"{k.removesuffix('_3class')}: {r[k]['value']:.3f} [{r[k]['ci95'][0]:.2f},{r[k]['ci95'][1]:.2f}]|{res[oid]['raw'][k]['value']:.2f}"
                for k in ("macro_f1_3class", "negative_f1_3class", "neutral_f1_3class", "positive_f1_3class")))
    print("\n== paired differences (reweighted [CI], P(>0)) ==")
    for d in report["differences"]:
        print(d["pair"])
        for k, v in d["reweighted"].items():
            print(f"   {k:<22} {v['value']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]  P(>0)={v['share_of_resamples_above_0']:.3f}"
                  f"   raw {d['raw'][k]['value']:+.3f}")
    print("\n== dev vs gold, reweighted complaint F1 ==")
    for oid, v in report["dev_vs_gold"].items():
        print(f"{cfg['options'][oid]['label']:<44} dev {v['dev']:.3f}  gold {v['gold']:.3f}  ({v['gold'] - v['dev']:+.3f})")
    s = report["star45_complaints"]
    print(f"\n== gold 4-5 star complaints (n={s['n_rows_raw']}) caught per option (raw | reweighted share) ==")
    for r, c in s["caught_per_role"].items():
        print(f"   {r:<16} {c['raw']:>3} | {c['reweighted_share']:.1%}")
    print("   PRIMARY vs comparison:", {k: v["raw"] for k, v in s["overlap_primary_vs_comparison"].items()})
    print("   per app:", s["per_app_caught"])
    e = report["errors"]
    print("\n== PRIMARY errors ==", e["all_error_type_counts"], "| false alarms flagged by:", e["flagged_by_among_false_alarms"])
    ia = report["intra_annotator"]
    if ia:
        print(f"\n== intra-annotator (n={ia['n']}) ==")
        for k in ("three_class", "binary_complaint"):
            x = ia[k]
            print(f"   {k:<17} kappa {x['kappa']:.3f} [{x['kappa_ci95'][0]:.2f}, {x['kappa_ci95'][1]:.2f}]  "
                  f"agreement {x['agreement']:.1%} [{x['agreement_ci95'][0]:.0%}, {x['agreement_ci95'][1]:.0%}]")
        print("   confusion (rows gold, cols relabel):", ia["confusion"]["labels"], ia["confusion"]["counts"])
        for d in ia["disagreements"]:
            print(f"   {d['blind_id']} {d['gold_id']} {d['app']:<9} gold={d['gold_label']:<8} relabel={d['relabel']:<8} {d['text_clean'][:90]!r}")


def main() -> None:  # noqa: PLR0915 - linear evaluation script
    """CLI entry point."""
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path("configs/gold_eval.yaml"))
    parser.add_argument("--smoke", action="store_true", help="run everything on DEV labels; gold is not read")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    for noisy in ("httpx", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    import torch

    cfg = load_config(args.config)
    dcfg = load_config(Path(cfg["dev_select_config"]))
    ecfg = load_config(Path(cfg["evaluate_config"]))
    gcfg = load_config(Path(cfg["gold_config"]))
    seed = cfg["seed"]
    np.random.seed(seed)
    out_report, out_errors, out_star45 = Path(cfg["report_path"]), Path(cfg["errors"]["path"]), Path(cfg["star45_path"])
    fig_dir = Path(cfg["figures_dir"])
    boot = dict(ecfg["bootstrap"])
    if args.smoke:
        boot["n"] = cfg["smoke"]["bootstrap_n"]
        out = Path(cfg["smoke"]["out_dir"])
        out_report, out_errors, out_star45, fig_dir = out / "report.json", out / "errors.csv", out / "star45.csv", out
    elif out_report.exists():
        raise FileExistsError(f"{out_report} exists: gold is evaluated once. Delete it only if you mean to re-run.")
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_report.parent.mkdir(parents=True, exist_ok=True)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("seed=%d smoke=%s device=%s bootstrap=%s", seed, args.smoke, device, boot)

    df = load_frame(cfg, args.smoke)
    logger.info("%s rows: %d", "dev (smoke)" if args.smoke else "gold", len(df))
    options, raw = predict_all(df, cfg, dcfg, device)

    mcfg = {"labels": ecfg["labels"], "per_app_excluded": cfg["per_app_excluded"], "seed": seed, "bootstrap": boot}
    logger.info("bootstrap n=%d over %d options", boot["n"], len(options))
    point, samples = ds.evaluate_options(df.assign(label=df["gold_label"]), options, mcfg)
    q = [(1 - boot["ci"]) / 2 * 100, (1 + boot["ci"]) / 2 * 100]
    results = summarize(point, samples, options, q)

    star45_rows, star45 = star45_breakdown(df, options, cfg)
    errs, err_info = export_errors(df, options, raw, cfg, dcfg)
    star45_rows.to_csv(out_star45, index=False, encoding="utf-8-sig")
    errs.to_csv(out_errors, index=False, encoding="utf-8-sig")

    pool = pd.read_parquet(ecfg["pool_path"])
    rel, rel_blank = load_relabel(gcfg, pool, drop_blank=args.smoke or cfg.get("relabel_drop_blank", False))
    if rel_blank:
        logger.warning("relabel rows without a label, excluded from agreement: %s", rel_blank)
    if args.smoke:
        # Exercise the code on fake "first labels" (relabel with a few flipped); gold is not read.
        fake = rel.set_index("reviewId")["relabel"].copy()
        fake.iloc[::7] = "neutral"
        ia = intra_annotator(rel, fake, 200, seed, boot["ci"])
    else:
        ia = intra_annotator(rel, df.set_index("reviewId")["gold_label"], boot["n"], seed, boot["ci"])
    ia["blank_relabel_rows_excluded"] = rel_blank

    dev_report = json.loads(Path(cfg["dev_report"]).read_text(encoding="utf-8"))
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    report = {
        "seed": seed,
        "split": "dev (SMOKE: pipeline check, not results)" if args.smoke else "gold (evaluated once)",
        "provenance": {"run_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "git_head": head,
                       "primary_fixed_before_gold": cfg["primary"], "device": device},
        "weighting": ("raw = gold rows as sampled (50 per app x weak_label); reweighted = each row weighted by "
                      "in_model_pool stratum size / 50, estimating the pool of unique review texts. Quote reweighted."),
        "target": "complaint = gold_label negative vs neutral/positive; 3-class metrics where the option is 3-class",
        "bootstrap": {**boot, "method": "stratified by app x weak_label, paired across options (as Phase 4)"},
        "options": cfg["options"],
        "rules": dcfg["rules"],
        "val_scores_weak_labels": {oid: dev_report["models"].get(oid.split("|")[0], {}).get("val") for oid in cfg["options"]},
        "results": results,
        "differences": differences(point, samples, cfg, q),
        "dev_vs_gold": {oid: {"dev": dev_report["options"][oid]["reweighted"]["bin_neg_f1"]["value"],
                              "dev_ci95": dev_report["options"][oid]["reweighted"]["bin_neg_f1"]["ci95"],
                              "gold": results[oid]["reweighted"]["bin_neg_f1"]["value"],
                              "gold_ci95": results[oid]["reweighted"]["bin_neg_f1"]["ci95"]} for oid in cfg["options"]},
        "confusion_matrices": confusion(df, options, ecfg["labels"]),
        "star45_complaints": star45,
        "errors": {**err_info, "csv": str(out_errors)},
        "intra_annotator": ia,
    }

    n_label = f"n={len(df)}"
    plot_panels(results, cfg, [("bin_neg_f1", "Complaint F1"), ("bin_neg_precision", "Precision"), ("bin_neg_recall", "Recall"),
                               ("recall_star_1-2", "Recall, 1-2★ reviews"), ("recall_star_3", "Recall, 3★ reviews"),
                               ("recall_star_4-5", "Recall, 4-5★ reviews")], 3,
                fig_dir / cfg["figures"]["complaint"], "Phase 5d gold: complaint detection (negative vs rest)",
                f"Gold {n_label}, reweighted to the model pool; dots = estimate, lines = 95% stratified bootstrap CI; "
                "ring = PRIMARY (fixed before gold).")
    apps = sorted(k.removeprefix("bin_neg_f1_") for k in results[cfg["primary"]]["reweighted"] if k.startswith("bin_neg_f1_"))
    plot_panels(results, cfg, [(f"bin_neg_f1_{a}", a) for a in apps], len(apps), fig_dir / cfg["figures"]["per_app"],
                "Phase 5d gold: complaint F1 per app", f"Gold {n_label} (50 per app x weak label), reweighted; 95% CI.")
    spec = "indobert/spec_3class/balanced|model_only"
    ev.plot_confusion(report["confusion_matrices"][spec]["3class"], fig_dir / cfg["figures"]["confusion"],
                      "IndoBERT spec_3class · balanced on gold (3-class)")

    out_report.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    try:
        report["wandb_url"] = log_wandb(report, dev_report, cfg, args.smoke)
    except Exception as exc:  # noqa: BLE001 - results are saved; W&B is a mirror
        logger.warning("W&B summary failed: %s", exc)
        report["wandb_url"] = f"failed: {exc}"
    out_report.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    logger.info("wrote %s, %s, %s, figures in %s", out_report, out_errors, out_star45, fig_dir)
    print_report(report, cfg)


if __name__ == "__main__":
    main()
