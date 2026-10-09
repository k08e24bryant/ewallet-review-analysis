"""Phase 5c: score every model on DEV (never gold) and select model + decision rule.

Steps:
    1. validate the dev workbook (same checks as gold); drop invalid rows
    2. predict dev with each IndoBERT run and each TF-IDF scheme
    3. binary complaint metrics (negative vs rest) for every model x rule, plus
       star-rating reference rows; raw and reweighted, stratified bootstrap CIs
    4. apply the selection rule fixed in configs/dev_select.yaml

Weighting: each dev row is weighted by in_model_pool stratum size / dev stratum
size (app x weak_label), estimating performance on the pool of unique review texts.

Usage:
    uv run python -m src.dev_select --config configs/dev_select.yaml
    uv run python -m src.dev_select --config configs/dev_select.yaml --smoke --device cpu
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.gold import parse_tags, read_sheet
from src.gold_report import GRID, SERIES, SURFACE, TEXT_PRIMARY, TEXT_SECONDARY

logger = logging.getLogger("dev_select")

STAR_GROUPS = {"1-2": (1, 2), "3": (3, 3), "4-5": (4, 5)}


def load_config(path: Path) -> dict[str, Any]:
    """Load a YAML config."""
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------- dev set

def load_dev(cfg: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Validate the dev workbook against its key, drop invalid rows, add text and weights.

    Raises SystemExit with every problem listed if validation fails.
    """
    dcfg = load_config(Path(cfg["dev_config"]))
    gcfg = load_config(Path(cfg["gold_config"]))
    labels, invalid = cfg["labels"], gcfg["invalid_label"]
    key = pd.read_parquet(dcfg["key_path"])
    pool = pd.read_parquet(cfg["pool_path"])
    sheet = read_sheet(Path(dcfg["xlsx_path"]), dcfg["sheet"]).rename(columns={"gold_id": "dev_id"})
    errors: list[str] = []

    new = key[key["source"] == "new"].sort_values("dev_id").reset_index(drop=True)
    if sheet["dev_id"].tolist() != new["dev_id"].tolist():
        missing = sorted(set(new["dev_id"]) - set(sheet["dev_id"]))
        extra = sorted(set(sheet["dev_id"]) - set(new["dev_id"]))
        errors.append(f"dev_id order/content differs from key (missing={missing[:10]}, extra={extra[:10]})")
    else:
        expected = new["reviewId"].map(pool.set_index("reviewId")["text_clean"])
        changed = [d for d, a, b in zip(sheet["dev_id"], sheet["text_clean"], expected) if a != b]
        if changed:
            errors.append(f"{len(changed)} texts differ from the exported text: {changed[:10]}")
    blank = sheet[sheet["label"].isna()]
    if len(blank):
        errors.append(f"{len(blank)} rows without a label: {blank['dev_id'].tolist()}")
    bad = sheet[sheet["label"].notna() & ~sheet["label"].isin(labels + [invalid])]
    if len(bad):
        errors.append(f"unknown labels: {bad[['dev_id', 'label']].values.tolist()}")
    bad_tags = [(d, t) for d, n in zip(sheet["dev_id"], sheet["notes"]) for t in parse_tags(n) if t not in gcfg["note_tags"]]
    if bad_tags:
        errors.append(f"unknown note tags: {bad_tags[:20]}")
    reused = key[key["source"] != "new"]
    bad_reused = reused[~reused["label"].isin(labels)]
    if len(bad_reused):
        errors.append(f"reused rows without a valid label: {bad_reused['dev_id'].tolist()}")
    if not key["reviewId"].is_unique:
        errors.append("duplicate reviewIds in the dev key")
    if not key["reviewId"].isin(pool["reviewId"]).all():
        errors.append("dev reviewIds missing from in_model_pool")

    splits = Path(cfg["splits_dir"])
    others = {
        "train": set(pd.read_parquet(splits / "train.parquet")["reviewId"]),
        "val": set(pd.read_parquet(splits / "val.parquet")["reviewId"]),
        "gold": set(pd.read_parquet(gcfg["output_path"])["reviewId"]),  # ids only; gold labels are not read
    }
    overlap = {name: len(ids & set(key["reviewId"])) for name, ids in others.items()}
    if any(overlap.values()):
        errors.append(f"dev overlaps other sets: {overlap}")

    if errors:
        print("\nDEV VALIDATION FAILED")
        for e in errors:
            print(" -", e)
        raise SystemExit(1)

    lab = sheet.set_index("dev_id")
    dev = key.copy()
    is_new = dev["source"] == "new"
    dev.loc[is_new, "label"] = dev.loc[is_new, "dev_id"].map(lab["label"])
    dev["notes"] = dev["dev_id"].map(lab["notes"]).where(is_new, None)
    dropped = dev[dev["label"] == invalid]
    dev = dev[dev["label"] != invalid].copy()

    dev = dev.join(pool.set_index("reviewId")[[cfg["text_col"], "text_clean"]], on="reviewId")
    sizes = pool.groupby(["app", "weak_label"]).size().rename("pool_n")
    dev = dev.join(sizes, on=["app", "weak_label"])
    dev["dev_n"] = dev.groupby(["app", "weak_label"])["reviewId"].transform("size")
    dev["weight"] = dev["pool_n"] / dev["dev_n"]
    dev = dev.reset_index(drop=True)

    target = load_config(Path(cfg["dev_config"]))["per_stratum"]
    strata = {}
    for (app, wl), g in key.groupby(["app", "weak_label"]):
        d = dropped[(dropped["app"] == app) & (dropped["weak_label"] == wl)]
        kept = dev[(dev["app"] == app) & (dev["weak_label"] == wl)]
        strata[f"{app}|{wl}"] = {
            "target": target, "rows": int(len(g)), "reused": int((g["source"] != "new").sum()),
            "new": int((g["source"] == "new").sum()), "invalid_dropped": int(len(d)), "final": int(len(kept)),
            "pool_n": int(sizes[(app, wl)]), "weight": round(float(sizes[(app, wl)] / len(kept)), 2),
            "label_counts": {l: int((kept["label"] == l).sum()) for l in labels},
        }
    info = {
        "checks": {"labels": "all valid", "texts_match_export": True, "note_tags": "all known",
                   "overlap_train_val_gold": overlap},
        "rows_in_key": int(len(key)), "invalid_dropped": dropped["dev_id"].tolist(), "final_rows": int(len(dev)),
        "note_tags": pd.Series([t for n in dev["notes"] for t in parse_tags(n)], dtype="object").value_counts().to_dict(),
        "strata": strata,
    }
    return dev, info


# ---------------------------------------------------------------- predictions

def predict_indobert(model_dir: Path, texts: list[str], device: str, batch_size: int, max_length: int) -> np.ndarray:
    """Argmax label names for texts (fp32, no gradient)."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device).eval()
    out: list[str] = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            enc = tok(texts[i : i + batch_size], truncation=True, max_length=max_length, padding=True, return_tensors="pt")
            logits = model(**{k: v.to(device) for k, v in enc.items()}).logits
            out += [model.config.id2label[j] for j in logits.argmax(-1).tolist()]
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return np.array(out, dtype=object)


def collect_predictions(dev: pd.DataFrame, cfg: dict[str, Any], smoke: bool, device: str) -> dict[str, dict[str, Any]]:
    """{model_id: {kind, scheme, weights, pred, val}} for every IndoBERT run and TF-IDF scheme."""
    import joblib

    texts = dev[cfg["text_col"]].tolist()
    models: dict[str, dict[str, Any]] = {}
    icfg = load_config(Path(cfg["indobert"]["config"]))
    root = Path(cfg["smoke"]["indobert_root"] if smoke else icfg["output_root"])
    for scheme, weights in cfg["indobert"]["runs"]:
        run_dir = root / scheme / weights
        if not (run_dir / "final").exists():
            if smoke:
                continue
            raise FileNotFoundError(f"{run_dir / 'final'} missing; train it first")
        info = json.loads((run_dir / "run_info.json").read_text(encoding="utf-8"))
        logger.info("predicting dev with IndoBERT %s/%s", scheme, weights)
        models[f"indobert/{scheme}/{weights}"] = {
            "kind": "indobert", "scheme": scheme, "weights": weights, "path": str(run_dir / "final"),
            "pred": predict_indobert(run_dir / "final", texts, device, cfg["indobert"]["batch_size"], icfg["max_length"]),
            "val": {"macro_f1": info["best_metric"], "negative_f1": info["val_metrics"].get("eval_negative_f1"),
                    "optimizer_steps": info["optimizer_steps"], "train_rows": info["train_rows"],
                    "wandb_url": info.get("wandb_url")},
        }

    bcfg = load_config(Path(cfg["baseline"]["config"]))
    for scheme in cfg["baseline"]["schemes"]:
        d = Path(bcfg["model_dir"]) if scheme == "spec_3class" else Path(bcfg["scheme_model_dir"]) / scheme
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        pipe = joblib.load(d / "pipeline.joblib")
        models[f"tfidf/{scheme}"] = {
            "kind": "tfidf", "scheme": scheme, "weights": bcfg["model"]["class_weight"], "path": str(d),
            "pred": pipe.predict(dev[bcfg["text_col"]]).astype(object),
            "val": {"macro_f1": meta["best"]["val_macro_f1"], "C": meta["best"]["C"], "min_df": meta["best"]["min_df"]},
        }
    return models


def build_options(dev: pd.DataFrame, models: dict[str, dict[str, Any]], cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Binary complaint predictions for every model x rule, plus star-rating reference rows."""
    star_neg = dev["score"].to_numpy() <= cfg["star_negative_max"]
    options: dict[str, dict[str, Any]] = {}
    for mid, m in models.items():
        neg = m["pred"] == "negative"
        for rule, p in (("model_only", neg), ("star_1_2_or_model", neg | star_neg)):
            options[f"{mid}|{rule}"] = {
                "model": mid, "kind": m["kind"], "scheme": m["scheme"], "weights": m["weights"], "rule": rule,
                "candidate": m["kind"] == cfg["selection"]["candidates"], "pred": p,
                "pred3": m["pred"] if (m["scheme"] == "spec_3class" and rule == "model_only") else None,
            }
    options["reference/star_1_2"] = {"model": "star rating", "kind": "reference", "scheme": None, "weights": None,
                                     "rule": "star 1-2", "candidate": False, "pred": star_neg, "pred3": None}
    return options


# ---------------------------------------------------------------- metrics

def _prf(t: np.ndarray, p: np.ndarray, w: np.ndarray) -> tuple[float, float, float]:
    tp, pp, ap = (w * (t & p)).sum(), (w * p).sum(), (w * t).sum()
    prec = tp / pp if pp > 0 else 0.0
    rec = tp / ap if ap > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
    return float(prec), float(rec), float(f1)


def metric_set(
    y: np.ndarray, p: np.ndarray, w: np.ndarray, score: np.ndarray, app: np.ndarray, apps: list[str],
    y3: np.ndarray | None = None, p3: np.ndarray | None = None, labels: list[str] | None = None,
) -> dict[str, float]:
    """Binary complaint metrics (negative = positive class) for one option on one (resampled) sample."""
    out: dict[str, float] = {}
    out["bin_neg_precision"], out["bin_neg_recall"], out["bin_neg_f1"] = _prf(y, p, w)
    out["pred_neg_share"] = float((w * p).sum() / w.sum())
    for g, (lo, hi) in STAR_GROUPS.items():
        m = (score >= lo) & (score <= hi)
        out[f"recall_star_{g}"] = _prf(y[m], p[m], w[m])[1]
    for a in apps:
        m = app == a
        out[f"bin_neg_f1_{a}"] = _prf(y[m], p[m], w[m])[2]
    if p3 is not None:
        f1s = [_prf(y3 == l, p3 == l, w)[2] for l in labels]
        out["macro_f1_3class"] = float(np.mean(f1s))
        out.update({f"{l}_f1_3class": f for l, f in zip(labels, f1s)})
    return out


def evaluate_options(dev: pd.DataFrame, options: dict[str, dict[str, Any]], cfg: dict[str, Any]) -> tuple[dict, dict]:
    """Point estimates and stratified bootstrap samples (paired across options) for raw and reweighted."""
    y = (dev["label"] == "negative").to_numpy()
    y3 = dev["label"].to_numpy()
    score, app = dev["score"].to_numpy(), dev["app"].to_numpy()
    apps = sorted(set(app) - set(cfg["per_app_excluded"]))
    weights = {"raw": np.ones(len(dev)), "reweighted": dev["weight"].to_numpy(dtype=float)}

    def all_metrics(rows: np.ndarray) -> dict[str, dict[str, dict[str, float]]]:
        return {
            mode: {
                oid: metric_set(y[rows], o["pred"][rows], w[rows], score[rows], app[rows], apps,
                                y3[rows] if o["pred3"] is not None else None,
                                o["pred3"][rows] if o["pred3"] is not None else None, cfg["labels"])
                for oid, o in options.items()
            }
            for mode, w in weights.items()
        }

    point = all_metrics(np.arange(len(dev)))
    rng = np.random.default_rng(cfg["seed"])
    strata = [g.index.to_numpy() for _, g in dev.groupby(["app", "weak_label"])]
    samples: list[dict] = []
    for _ in range(cfg["bootstrap"]["n"]):
        rows = np.concatenate([rng.choice(s, size=len(s), replace=True) for s in strata])
        samples.append(all_metrics(rows))
    return point, samples


# ---------------------------------------------------------------- selection

def select(f1: dict[str, float], options: dict[str, dict[str, Any]], scfg: dict[str, Any]) -> dict[str, Any]:
    """Apply the pre-registered rule to {option_id: reweighted F1} over candidate options."""
    cand = [o for o in f1 if options[o]["candidate"]]
    rank = scfg["rank"]

    def key(o: str) -> tuple[int, int, int]:
        x = options[o]
        return rank["scheme"][x["scheme"]], rank["rule"][x["rule"]], rank["weights"][x["weights"]]

    best = max(cand, key=lambda o: f1[o])
    eligible = [o for o in cand if f1[o] >= f1[best] - scfg["tolerance"] - 1e-12]
    selected = min(eligible, key=lambda o: (key(o), -f1[o]))
    simplest = min(cand, key=lambda o: (key(o), -f1[o]))
    literal = simplest if f1[best] - f1[simplest] <= scfg["tolerance"] + 1e-12 else best
    return {"best": best, "simplest": simplest, "eligible": sorted(eligible, key=lambda o: -f1[o]),
            "selected": selected, "literal_reading": literal}


# ---------------------------------------------------------------- report

def build_report(dev, info, models, options, point, samples, cfg) -> dict[str, Any]:
    """Assemble the JSON report: per-option metrics with CIs, paired differences, selection."""
    q = [(1 - cfg["bootstrap"]["ci"]) / 2 * 100, (1 + cfg["bootstrap"]["ci"]) / 2 * 100]

    def ci(vals: list[float]) -> list[float]:
        return [float(v) for v in np.percentile(vals, q)]

    metric = cfg["selection"]["metric"]
    f1 = {o: point["reweighted"][o][metric] for o in options}
    sel = select(f1, options, cfg["selection"])
    freq = pd.Series([select({o: s["reweighted"][o][metric] for o in options}, options, cfg["selection"])["selected"]
                      for s in samples]).value_counts(normalize=True)

    def diff(a: str, b: str) -> dict[str, Any]:
        d = [s["reweighted"][a][metric] - s["reweighted"][b][metric] for s in samples]
        return {"value": f1[a] - f1[b], "ci95": ci(d), "share_of_resamples_above_0": float(np.mean(np.array(d) > 0))}

    ref = "reference/star_1_2"
    out_opts = {}
    for oid, o in options.items():
        entry = {k: o[k] for k in ("model", "kind", "scheme", "weights", "rule", "candidate")}
        for mode in ("raw", "reweighted"):
            entry[mode] = {k: {"value": v, "ci95": ci([s[mode][oid][k] for s in samples])}
                           for k, v in point[mode][oid].items()}
        if oid != ref:
            entry["diff_vs_star_1_2_reweighted_f1"] = diff(oid, ref)
        out_opts[oid] = entry

    s = sel["selected"]
    best_tfidf = max((o for o in options if options[o]["kind"] == "tfidf"), key=lambda o: f1[o])
    gap = f1[sel["best"]] - f1[s]
    if s == sel["best"]:
        reason = (f"{s} has the highest reweighted dev binary F1 ({f1[s]:.3f}) and no simpler candidate is within "
                  f"{cfg['selection']['tolerance']}")
    else:
        reason = (f"best is {sel['best']} ({f1[sel['best']]:.3f}); {s} ({f1[s]:.3f}) is {gap:.3f} lower, within "
                  f"{cfg['selection']['tolerance']}, and is the simplest eligible option")
    selection = {
        **sel,
        "selected_f1_reweighted": f1[s],
        "reason": reason,
        "literal_reading_agrees": sel["literal_reading"] == s,
        "selected_minus_best": diff(s, sel["best"]) if s != sel["best"] else None,
        "selected_minus_simplest": diff(s, sel["simplest"]) if s != sel["simplest"] else None,
        "selected_minus_star_1_2": diff(s, ref),
        "bootstrap_selection_frequency": {k: round(float(v), 4) for k, v in freq.head(8).items()},
        "best_tfidf_option": best_tfidf,
        "best_tfidf_minus_selected": diff(best_tfidf, s),
    }
    ranking = sorted(options, key=lambda o: -f1[o])
    return {
        "seed": cfg["seed"],
        "split": "dev (gold not read)",
        "weighting": ("raw = dev rows as sampled; reweighted = each row weighted by in_model_pool stratum size / "
                      "dev stratum size (app x weak_label), estimating the pool of unique review texts. Quote "
                      "reweighted numbers. OVO has 12 dev rows; OVO per-app results are not computed."),
        "target": "dev label == negative (complaint) vs neutral/positive",
        "bootstrap": {**cfg["bootstrap"], "method": "stratified by app x weak_label, paired across options"},
        "rules": cfg["rules"],
        "selection_rule": cfg["selection"],
        "dev_validation": info,
        "models": {mid: {k: m[k] for k in ("kind", "scheme", "weights", "path", "val")} for mid, m in models.items()},
        "selection": selection,
        "ranking_reweighted_bin_neg_f1": [[o, round(f1[o], 4)] for o in ranking],
        "options": out_opts,
    }


def short_name(o: dict[str, Any]) -> str:
    """Row label for a model in tables and figures."""
    if o["kind"] == "reference":
        return "Star rating 1-2"
    kind = "IndoBERT" if o["kind"] == "indobert" else "TF-IDF"
    w = f" · {o['weights']}" if o["kind"] == "indobert" else ""
    return f"{kind} {o['scheme']}{w}"


def plot_selection(report: dict[str, Any], path: Path) -> None:
    """Dot plot: reweighted dev binary F1 with 95% CI for every model under both rules."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    opts = report["options"]
    sel = report["selection"]["selected"]
    rows = list(dict.fromkeys(o["model"] for o in opts.values()))  # IndoBERT, TF-IDF, then reference
    colors = {"model_only": SERIES[0], "star_1_2_or_model": SERIES[1], "star 1-2": TEXT_SECONDARY}
    offset = {"model_only": -0.15, "star_1_2_or_model": 0.15, "star 1-2": 0.0}
    fig, ax = plt.subplots(figsize=(10, 0.48 * len(rows) + 2.0), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    lo_all = 1.0
    for oid, o in opts.items():
        yi = rows.index(o["model"]) + offset[o["rule"]]
        m = o["reweighted"]["bin_neg_f1"]
        v, (lo, hi) = m["value"], m["ci95"]
        lo_all = min(lo_all, lo)
        c = colors[o["rule"]]
        ax.plot([lo, hi], [yi, yi], color=c, linewidth=2, solid_capstyle="round", zorder=2, alpha=0.55)
        marker = "D" if o["kind"] == "reference" else "o"
        ax.plot(v, yi, marker=marker, markersize=8, color=c, markeredgecolor=SURFACE, markeredgewidth=2, zorder=3)
        if oid == sel:
            ax.plot(v, yi, marker="o", markersize=15, markerfacecolor="none", markeredgecolor=TEXT_PRIMARY,
                    markeredgewidth=1.5, zorder=4)
            ax.text(hi + 0.006, yi, f"selected  {v:.3f}", va="center", color=TEXT_PRIMARY, fontsize=9, fontweight="bold")
        elif o["kind"] == "reference":
            ax.text(hi + 0.006, yi, f"{v:.3f}", va="center", color=TEXT_SECONDARY, fontsize=9)
            ax.axvline(v, color=TEXT_SECONDARY, linewidth=1, linestyle=(0, (3, 3)), zorder=1)
    labels = [short_name(next(o for o in opts.values() if o["model"] == r)) for r in rows]
    kinds = [next(o for o in opts.values() if o["model"] == r)["kind"] for r in rows]
    for i in range(1, len(rows)):
        if kinds[i] != kinds[i - 1]:  # separate IndoBERT / TF-IDF / reference groups
            ax.axhline(i - 0.5, color=GRID, linewidth=1, zorder=0)
    ax.set_yticks(range(len(rows)), labels)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    left = max(0.0, np.floor(lo_all * 20) / 20)
    ax.set_xlim(left, 1.07)
    ax.set_xticks(np.arange(left, 1.0001, 0.05))
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.spines["bottom"].set_bounds(left, 1.0)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=10, length=0)
    ax.set_xlabel("Complaint F1 on dev (negative vs rest, reweighted to the model pool)", color=TEXT_SECONDARY)
    fig.text(0.01, 0.985, "Phase 5c dev selection: complaint detection by model and decision rule",
             color=TEXT_PRIMARY, fontsize=12, va="top")
    fig.text(0.01, 0.985 - 0.32 / fig.get_figheight(),
             f"Dev set, n={report['dev_validation']['final_rows']}; dots = point estimate, lines = 95% stratified "
             "bootstrap CI; dashed line = star rating alone. Gold not used.",
             color=TEXT_SECONDARY, fontsize=9, va="top")
    handles = [Line2D([], [], color=SERIES[0], marker="o", linewidth=2, label="model only"),
               Line2D([], [], color=SERIES[1], marker="o", linewidth=2, label="star 1-2 OR model"),
               Line2D([], [], color=TEXT_SECONDARY, marker="D", linewidth=0, label="star rating alone"),
               Line2D([], [], color=TEXT_PRIMARY, marker="o", markersize=11, markerfacecolor="none", linewidth=0,
                      label="selected")]
    ax.legend(handles=handles, frameon=False, labelcolor=TEXT_PRIMARY, loc="lower left",
              bbox_to_anchor=(0, 1.0), ncol=4)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.62 / fig.get_figheight()))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def print_report(report: dict[str, Any]) -> None:
    """Console tables: dev validation, per-option metrics, selection."""
    v = report["dev_validation"]
    print("\n== dev validation per stratum ==")
    print(pd.DataFrame(v["strata"]).T.drop(columns=["label_counts"]).to_string())
    print(f"invalid dropped: {v['invalid_dropped']}  final rows: {v['final_rows']}  note tags: {v['note_tags']}")

    print("\n== dev binary complaint metrics: reweighted value [95% CI] | raw ==")
    cols = ["bin_neg_f1", "bin_neg_precision", "bin_neg_recall", "recall_star_1-2", "recall_star_3", "recall_star_4-5"]
    print(f"{'option':<52}" + "".join(f"{c:>26}" for c in cols))
    for oid, _ in report["ranking_reweighted_bin_neg_f1"]:
        o = report["options"][oid]
        cells = [f"{o['reweighted'][c]['value']:.3f} [{o['reweighted'][c]['ci95'][0]:.2f},"
                 f"{o['reweighted'][c]['ci95'][1]:.2f}] | {o['raw'][c]['value']:.2f}" for c in cols]
        tag = "" if o["candidate"] or o["kind"] == "reference" else " (cmp)"
        print(f"{short_name(o) + ' | ' + o['rule'] + tag:<52}" + "".join(f"{c:>26}" for c in cells))

    s = report["selection"]
    print("\n== selection ==")
    for k in ("best", "simplest", "selected", "literal_reading"):
        print(f"{k:<16} {s[k]}")
    print("eligible:", s["eligible"])
    print("reason:", s["reason"])
    d = s["selected_minus_star_1_2"]
    print(f"selected minus star 1-2: {d['value']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}] "
          f"P(>0)={d['share_of_resamples_above_0']:.3f}")
    d = s["best_tfidf_minus_selected"]
    print(f"best TF-IDF ({s['best_tfidf_option']}) minus selected: {d['value']:+.3f} "
          f"[{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}]")
    print("bootstrap selection frequency:", s["bootstrap_selection_frequency"])


def main() -> None:
    """CLI entry point."""
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path("configs/dev_select.yaml"))
    parser.add_argument("--smoke", action="store_true", help="smoke IndoBERT runs, 50 bootstrap draws, output under models/")
    parser.add_argument("--device", default=None, help="cuda or cpu (default: cuda if available)")
    parser.add_argument("--validate-only", action="store_true", help="check the dev workbook and stop")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    for noisy in ("httpx", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    cfg = load_config(args.config)
    logger.info("seed=%s config=%s smoke=%s", cfg["seed"], args.config, args.smoke)
    np.random.seed(cfg["seed"])
    dev, info = load_dev(cfg)
    if args.validate_only:
        print(pd.DataFrame(info["strata"]).T.to_string())
        print(json.dumps({k: v for k, v in info.items() if k != "strata"}, indent=2))
        return

    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    report_path, fig_path = Path(cfg["report_path"]), Path(cfg["figures_dir"]) / cfg["figure"]
    if args.smoke:
        cfg["bootstrap"]["n"] = cfg["smoke"]["bootstrap_n"]
        out = Path(cfg["smoke"]["out_dir"])
        report_path, fig_path = out / "report.json", out / cfg["figure"]

    models = collect_predictions(dev, cfg, args.smoke, device)
    options = build_options(dev, models, cfg)
    logger.info("%d models, %d options; bootstrap n=%d", len(models), len(options), cfg["bootstrap"]["n"])
    point, samples = evaluate_options(dev, options, cfg)
    report = build_report(dev, info, models, options, point, samples, cfg)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    plot_selection(report, fig_path)
    logger.info("wrote %s and %s", report_path, fig_path)
    print_report(report)


if __name__ == "__main__":
    main()
