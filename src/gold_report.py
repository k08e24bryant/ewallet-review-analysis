"""Phase 3b analysis: final gold set, weak-vs-gold agreement, and figures.

The gold set is stratified equally (50 per app x weak_label stratum), not in
proportion to the data. Every metric is reported twice:

    raw         computed on the 600 gold rows as sampled (describes the gold set)
    reweighted  each row weighted by pool_stratum_size / 50 (estimates the
                in_model_pool population of unique review texts)

Population statements ("X% of 4-5 star reviews are complaints") must quote the
reweighted number. Within a single app x weak_label stratum both are identical.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix, precision_recall_fscore_support

# Reference palette tokens (dataviz skill, light mode)
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SERIES = ["#2a78d6", "#eb6834"]  # validated pair: blue, orange


def assemble_final(
    sheets: dict[str, pd.DataFrame], key: pd.DataFrame, filled: list[dict[str, Any]], invalid: str
) -> pd.DataFrame:
    """Build the 600-row gold set: valid gold rows plus the reserve rows that fill invalid slots."""
    from src.gold import parse_tags

    labeled = pd.concat(
        [df.assign(sheet=name) for name, df in sheets.items()], ignore_index=True
    ).set_index("gold_id")
    k = key.set_index("gold_id")

    def row(gid: str, slot: str, source: str) -> dict[str, Any]:
        return {
            "gold_id": gid,
            "slot_id": slot,
            "reviewId": k.at[gid, "reviewId"],
            "app": k.at[gid, "app"],
            "score": int(k.at[gid, "score"]),
            "weak_label": k.at[gid, "weak_label"],
            "gold_label": labeled.at[gid, "label"],
            "notes": parse_tags(labeled.at[gid, "notes"]),
            "source": source,
            "reserve_set": None if source == "gold" else k.at[gid, "set"],
        }

    gold = sheets["gold"]
    rows = [row(g, g, "gold") for g, lab in zip(gold["gold_id"], gold["label"]) if lab != invalid]
    rows += [row(f["reserve_id"], f["gold_id"], "reserve") for f in filled]
    out = pd.DataFrame(rows).sort_values("slot_id").reset_index(drop=True)
    return out


def check_final(final: pd.DataFrame, cfg: dict[str, Any], train_ids: set[str], val_ids: set[str]) -> dict[str, Any]:
    """Assert the final gold set is complete, valid, balanced, and disjoint from train/val."""
    n_strata = final.groupby(["app", "weak_label"]).size()
    assert (n_strata == cfg["per_stratum"]).all(), f"strata not all {cfg['per_stratum']}:\n{n_strata}"
    assert len(n_strata) == len(final["app"].unique()) * len(cfg["labels"])
    assert final["gold_label"].isin(cfg["labels"]).all(), "invalid/blank labels in final gold set"
    assert final["reviewId"].is_unique and final["slot_id"].is_unique
    ids = set(final["reviewId"])
    overlap = {"train": len(ids & train_ids), "val": len(ids & val_ids)}
    assert overlap == {"train": 0, "val": 0}, f"gold overlaps splits: {overlap}"
    return {"rows": len(final), "per_stratum": int(n_strata.iloc[0]), "overlap": overlap}


def add_weights(final: pd.DataFrame, pool: pd.DataFrame) -> pd.DataFrame:
    """Attach the population weight: pool stratum size / gold stratum size."""
    sizes = pool.groupby(["app", "weak_label"]).size().rename("pool_n")
    out = final.join(sizes, on=["app", "weak_label"])
    out["weight"] = out["pool_n"] / out.groupby(["app", "weak_label"])["reviewId"].transform("size")
    return out


def _share(mask: pd.Series, w: pd.Series | None) -> float:
    if w is None:
        return float(mask.mean())
    return float((mask * w).sum() / w.sum())


def _kappa(df: pd.DataFrame, weighted: bool) -> float:
    return float(cohen_kappa_score(df["weak_label"], df["gold_label"], sample_weight=df["weight"] if weighted else None))


def bootstrap_ci(df: pd.DataFrame, stat, n: int, ci: float, seed: int) -> list[float]:
    """Percentile CI from resampling rows within each app x weak_label stratum."""
    rng = np.random.default_rng(seed)
    groups = [g.index.to_numpy() for _, g in df.groupby(["app", "weak_label"])]
    vals = []
    for _ in range(n):
        idx = np.concatenate([rng.choice(g, size=len(g), replace=True) for g in groups])
        vals.append(stat(df.loc[idx]))
    lo, hi = np.nanpercentile(vals, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
    return [float(lo), float(hi)]


def agreement_report(final: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, Any]:
    """Distribution, kappa, confusion matrices, precision/recall, mismatch, tags, replacements."""
    labels = cfg["labels"]
    bs = cfg["bootstrap"]
    seed = cfg["seed"]
    rep: dict[str, Any] = {}

    def dist(df: pd.DataFrame) -> dict[str, Any]:
        return {
            "counts": {l: int((df["gold_label"] == l).sum()) for l in labels},
            "raw_share": {l: _share(df["gold_label"] == l, None) for l in labels},
            "reweighted_share": {l: _share(df["gold_label"] == l, df["weight"]) for l in labels},
        }

    rep["gold_label_distribution"] = {"all": dist(final), **{a: dist(g) for a, g in final.groupby("app")}}

    def kappas(df: pd.DataFrame) -> dict[str, Any]:
        return {
            "raw": _kappa(df, False),
            "reweighted": _kappa(df, True),
            "reweighted_ci95": bootstrap_ci(df, lambda d: _kappa(d, True), bs["n"], bs["ci"], seed),
        }

    rep["kappa"] = {"all": kappas(final), **{a: kappas(g) for a, g in final.groupby("app")}}

    cm_raw = confusion_matrix(final["weak_label"], final["gold_label"], labels=labels)
    cm_w = confusion_matrix(final["weak_label"], final["gold_label"], labels=labels, sample_weight=final["weight"])
    rep["confusion_matrix"] = {
        "rows": "weak_label",
        "cols": "gold_label",
        "labels": labels,
        "raw_counts": cm_raw.tolist(),
        "reweighted_share_of_pool": (cm_w / cm_w.sum()).round(4).tolist(),
        "reweighted_row_share": (cm_w / cm_w.sum(axis=1, keepdims=True)).round(4).tolist(),
    }

    def prf(df: pd.DataFrame, weighted: bool) -> dict[str, Any]:
        p, r, f, s = precision_recall_fscore_support(
            df["gold_label"], df["weak_label"], labels=labels, zero_division=0,
            sample_weight=df["weight"] if weighted else None,
        )
        return {l: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i])} for i, l in enumerate(labels)}

    rep["weak_label_quality_vs_gold"] = {
        "note": "weak_label treated as a prediction of gold_label",
        "raw": prf(final, False),
        "reweighted": prf(final, True),
    }

    def mismatch(df: pd.DataFrame) -> dict[str, Any]:
        pos = df[df["weak_label"] == "positive"]
        neg = df[df["weak_label"] == "negative"]
        out = {
            "pos_stars_labeled_negative": {
                "n": int(len(pos)),
                "raw": _share(pos["gold_label"] == "negative", None),
                "reweighted": _share(pos["gold_label"] == "negative", pos["weight"]),
            },
            "neg_stars_labeled_positive": {
                "n": int(len(neg)),
                "raw": _share(neg["gold_label"] == "positive", None),
                "reweighted": _share(neg["gold_label"] == "positive", neg["weight"]),
            },
        }
        for name, sub, target in (
            ("pos_stars_labeled_negative", pos, "negative"),
            ("neg_stars_labeled_positive", neg, "positive"),
        ):
            out[name]["reweighted_ci95"] = bootstrap_ci(
                sub, lambda d, t=target: _share(d["gold_label"] == t, d["weight"]), bs["n"], bs["ci"], seed
            )
        return out

    rep["rating_text_mismatch"] = {"all": mismatch(final), **{a: mismatch(g) for a, g in final.groupby("app")}}
    by_star = {}
    for s, g in final.groupby("score"):
        by_star[int(s)] = {
            "n": int(len(g)),
            **{f"reweighted_{l}": _share(g["gold_label"] == l, g["weight"]) for l in labels},
        }
    rep["gold_label_by_star_reweighted"] = by_star

    tags = pd.Series([t for ts in final["notes"] for t in ts], dtype="object").value_counts()
    rep["notes_tags"] = {t: int(tags.get(t, 0)) for t in cfg["note_tags"]}
    rep["notes_tags_by_gold_label"] = {
        t: final[final["notes"].map(lambda ts, t=t: t in ts)]["gold_label"].value_counts().to_dict()
        for t in cfg["note_tags"]
    }
    rep["replaced_invalid_per_stratum"] = {
        f"{a}|{w}": {
            "replaced": int((g["source"] == "reserve").sum()),
            "by_set": g.loc[g["source"] == "reserve", "reserve_set"].value_counts().to_dict(),
        }
        for (a, w), g in final.groupby(["app", "weak_label"])
    }
    rep["pool_stratum_sizes"] = {
        f"{a}|{w}": int(g["pool_n"].iloc[0]) for (a, w), g in final.groupby(["app", "weak_label"])
    }
    return rep


def _style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=10)


def plot_confusion(rep: dict[str, Any], path: Path) -> None:
    """Two panels: raw gold counts and reweighted row shares; cells shaded by row share."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    cm = rep["confusion_matrix"]
    labels = cm["labels"]
    raw = np.array(cm["raw_counts"])
    raw_row = raw / raw.sum(axis=1, keepdims=True)
    w_row = np.array(cm["reweighted_row_share"])
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), facecolor=SURFACE)
    panels = [
        (axes[0], raw_row, lambda i, j: f"{raw[i, j]}\n{raw_row[i, j]:.0%}", "Gold set as sampled (n=600)\ncount and row share"),
        (axes[1], w_row, lambda i, j: f"{w_row[i, j]:.0%}", "Reweighted to the model pool\nrow share (estimate)"),
    ]
    for ax, shade, text, title in panels:
        ax.imshow(shade, cmap=cmap, vmin=0, vmax=1)
        for i in range(len(labels)):
            for j in range(len(labels)):
                ink = "white" if shade[i, j] >= 0.55 else TEXT_PRIMARY
                ax.text(j, i, text(i, j), ha="center", va="center", color=ink, fontsize=11)
        ax.set_xticks(range(len(labels)), labels)
        ax.set_yticks(range(len(labels)), labels)
        ax.set_xlabel("Gold label (manual, text only)", color=TEXT_SECONDARY)
        ax.set_ylabel("Weak label (from stars)", color=TEXT_SECONDARY)
        ax.set_title(title, color=TEXT_PRIMARY, fontsize=11, loc="left")
        ax.tick_params(colors=TEXT_SECONDARY, length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_xticks(np.arange(-0.5, len(labels)), minor=True)
        ax.set_yticks(np.arange(-0.5, len(labels)), minor=True)
        ax.grid(which="minor", color=SURFACE, linewidth=2)
        ax.tick_params(which="minor", length=0)
    k = rep["kappa"]["all"]
    fig.suptitle(
        f"Star-based weak labels vs manual gold labels   "
        f"Cohen's kappa: raw {k['raw']:.2f}, reweighted {k['reweighted']:.2f}",
        x=0.02, ha="left", color=TEXT_PRIMARY, fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_mismatch(rep: dict[str, Any], path: Path) -> None:
    """Grouped bars per app: 4-5 star labeled negative, 1-2 star labeled positive (with 95% CIs)."""
    import matplotlib.pyplot as plt

    mm = rep["rating_text_mismatch"]
    apps = sorted(a for a in mm if a != "all") + ["all"]
    series = [
        ("pos_stars_labeled_negative", "4-5 star review, text is negative"),
        ("neg_stars_labeled_positive", "1-2 star review, text is positive"),
    ]
    x = np.arange(len(apps))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9, 4.6), facecolor=SURFACE)
    _style_axes(ax)
    for s, ((key, label), color) in enumerate(zip(series, SERIES)):
        vals = np.array([mm[a][key]["reweighted"] for a in apps])
        lo = np.array([mm[a][key]["reweighted_ci95"][0] for a in apps])
        hi = np.array([mm[a][key]["reweighted_ci95"][1] for a in apps])
        pos = x + (s - 0.5) * (width + 0.02)
        ax.bar(pos, vals, width, color=color, label=label, edgecolor=SURFACE, linewidth=2, zorder=2)
        ax.errorbar(pos, vals, yerr=[vals - lo, hi - vals], fmt="none", ecolor=TEXT_SECONDARY,
                    elinewidth=1, capsize=3, zorder=3)
        for xi, v, h in zip(pos, vals, hi):
            ax.text(xi, h + 0.015, f"{v:.0%}", ha="center", va="bottom", color=TEXT_PRIMARY, fontsize=9)
    ax.set_xticks(x, [a if a != "all" else "all apps\n(reweighted)" for a in apps])
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_ylabel("Share of reviews in that star group", color=TEXT_SECONDARY)
    ax.set_title("Rating-text mismatch in the gold set (95% bootstrap CI)", color=TEXT_PRIMARY,
                 fontsize=12, loc="left")
    ax.legend(frameon=False, labelcolor=TEXT_PRIMARY, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def export_relabel(final: pd.DataFrame, pool: pd.DataFrame, cfg: dict[str, Any], label_cfg: dict[str, Any]) -> pd.DataFrame:
    """Write the blind relabel workbook (refuses to overwrite) and its key."""
    from openpyxl import Workbook

    from src.label import write_label_sheet

    rl = cfg["relabel"]
    path = Path(rl["xlsx_path"])
    if path.exists():
        raise FileExistsError(f"{path} exists; refusing to overwrite (it may contain labels)")
    pick = final.sample(n=rl["n"], random_state=cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    pick = pick.iloc[rng.permutation(len(pick))].reset_index(drop=True)
    pick.insert(0, "blind_id", [f"{rl['id_prefix']}{i:0{rl['id_width']}d}" for i in range(1, len(pick) + 1)])
    text = pool.set_index("reviewId")["text_clean"]
    sheet_rows = pd.DataFrame({"gold_id": pick["blind_id"], "text_clean": pick["reviewId"].map(text)})

    wb = Workbook()
    ws = wb.active
    ws.title = "relabel"
    write_label_sheet(ws, sheet_rows, label_cfg["gold"])
    wb.save(path)
    key = pick[["blind_id", "gold_id", "slot_id", "reviewId", "app", "weak_label"]]
    key.to_parquet(rl["key_path"], index=False)
    return key
