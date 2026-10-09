"""Phase 7: weekly complaint trends per app, spikes, hidden complaints, topic trends, versions.

Complaint = PRIMARY (IndoBERT spec_3class/none + "star 1-2 OR model"). Complaint shares use
ALL rows (duplicates included); topic shares use fitted documents only (unique, non-short
texts) because topic fit on short reviews is poor (Phase 6c-2). Complete Monday-Sunday
weeks only; the partial Sep 28 week is shown separately. No gold or dev labels.

Usage:
    uv run python -m src.trends --config configs/trends.yaml
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

from src.gold_report import GRID, SERIES, SURFACE, TEXT_PRIMARY, TEXT_SECONDARY
from src.topics_final import wilson

logger = logging.getLogger("trends")

# Reference categorical palette (dataviz skill, light mode), fixed slot order
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
MUTED = "#c9c8c2"


def load_config(path: Path) -> dict[str, Any]:
    """Load a YAML config."""
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def newcombe(k1: int, n1: int, k2: int, n2: int, ci: float) -> list[float]:
    """Newcombe hybrid score interval for p1 - p2 (from two Wilson intervals)."""
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = wilson(k1, n1, ci)
    l2, u2 = wilson(k2, n2, ci)
    d = p1 - p2
    return [float(d - np.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)), float(d + np.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2))]


def share(k: int, n: int, ci: float) -> dict[str, Any]:
    """k, n, share and Wilson CI."""
    return {"k": int(k), "n": int(n), "share": k / n if n else float("nan"), "ci95": wilson(int(k), int(n), ci) if n else [None, None]}


# ---------------------------------------------------------------- data

def load(cfg: dict[str, Any]) -> pd.DataFrame:
    """All reviews with PRIMARY predictions and v3 topics (complaint rows only have topics)."""
    rv = pd.read_parquet(cfg["input_path"], columns=["reviewId", "app", "score", "at", "week_start", "is_partial_week",
                                                     "version_major_minor", "in_model_pool", "is_short", "text_clean"])
    pr = pd.read_parquet(cfg["predictions_path"], columns=["reviewId", "is_complaint", "pred_label", "flagged_by"])
    asg = pd.read_parquet(cfg["assignments_path"], columns=["reviewId", "topic", "topic_name", "in_fit", "assigned_by_transform"])
    df = rv.merge(pr, on="reviewId", validate="one_to_one").merge(asg, on="reviewId", how="left", validate="one_to_one")
    if df["is_complaint"].sum() != len(asg) or df.loc[df["is_complaint"], "topic"].isna().any():
        raise ValueError("topic assignments do not match PRIMARY complaint rows")
    df["week"] = df["week_start"].dt.strftime("%Y-%m-%d")
    df["date"] = df["at"].dt.strftime("%Y-%m-%d")
    df["in_fit"] = df["in_fit"].fillna(False).astype(bool)
    return df


def week_days(df: pd.DataFrame, week: str) -> float:
    """Days of data in a week: 7 for complete weeks, elapsed days for the partial week."""
    w = df[df["week"] == week]
    if not w["is_partial_week"].iloc[0]:
        return 7.0
    first = max(w["at"].min().floor("D"), pd.Timestamp(week, tz="UTC"))
    last = min(w["at"].max(), pd.Timestamp(week, tz="UTC") + pd.Timedelta(days=7))
    return float((last - first).total_seconds() / 86400)


# ---------------------------------------------------------------- 1. weekly complaint share

def weekly(df: pd.DataFrame, apps: list[str], ci: float, partial: str) -> pd.DataFrame:
    """Per app x week (complete weeks + the shown partial week): volume, complaints, share, Wilson CI."""
    keep = df[~df["is_partial_week"] | (df["week"] == partial)]
    g = keep.groupby(["app", "week"]).agg(n=("reviewId", "size"), k=("is_complaint", "sum"),
                                          partial=("is_partial_week", "first")).reset_index()
    g["share"] = g["k"] / g["n"]
    cis = [wilson(int(k), int(n), ci) for k, n in zip(g["k"], g["n"])]
    g["lo"], g["hi"] = [c[0] for c in cis], [c[1] for c in cis]
    return g[g["app"].isin(apps)].sort_values(["app", "week"]).reset_index(drop=True)


# ---------------------------------------------------------------- 2. spikes

def spike_analysis(df: pd.DataFrame, wk: pd.DataFrame, cfg: dict[str, Any]) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    """Volume, complaint share, stars and specific-topic shifts for each spike vs the app's non-spike weeks."""
    ci = cfg["ci"]
    excluded = set(cfg["excluded_topics"]) | {cfg["outlier_label"]}
    out, examples = [], []
    rng_seed = cfg["seed"]
    for sp in cfg["spikes"]:
        app, week = sp["app"], sp["week"]
        app_spikes = {s["week"] for s in cfg["spikes"] if s["app"] == app}
        a = df[df["app"] == app]
        complete = sorted(a.loc[~a["is_partial_week"], "week"].unique())
        base_weeks = [w for w in complete if w not in app_spikes]
        s_rows, b_rows = a[a["week"] == week], a[a["week"].isin(base_weeks)]
        days = week_days(a, week)
        wb = wk[(wk["app"] == app) & wk["week"].isin(base_weeks)]
        base_med_n, base_med_share = float(wb["n"].median()), float(wb["share"].median())
        k_s, n_s = int(s_rows["is_complaint"].sum()), len(s_rows)
        k_b, n_b = int(b_rows["is_complaint"].sum()), len(b_rows)
        stars = {int(s): {"spike": float((s_rows["score"] == s).mean()), "baseline": float((b_rows["score"] == s).mean()),
                          "diff_pp": float(100 * ((s_rows["score"] == s).mean() - (b_rows["score"] == s).mean()))}
                 for s in range(1, 6)}

        # Topics: fitted complaint documents; denominator = all fitted complaint docs of the app (outliers included)
        fs = s_rows[s_rows["is_complaint"] & s_rows["in_fit"]]
        fb = b_rows[b_rows["is_complaint"] & b_rows["in_fit"]]
        topics = []
        for t in sorted(set(fs["topic_name"]) | set(fb["topic_name"])):
            ks, kb = int((fs["topic_name"] == t).sum()), int((fb["topic_name"] == t).sum())
            topics.append({"topic": t, "specific": t not in excluded, "k_spike": ks, "n_spike": len(fs), "k_base": kb, "n_base": len(fb),
                           "share_spike": ks / len(fs), "share_base": kb / len(fb),
                           "diff_pp": 100 * (ks / len(fs) - kb / len(fb)),
                           "diff_ci95_pp": [100 * x for x in newcombe(ks, len(fs), kb, len(fb), ci)]})
        topics.sort(key=lambda r: -r["diff_pp"])
        rising = [r for r in topics if r["specific"] and r["diff_pp"] > 0][: cfg["spike_top_topics"]]
        for r in rising:
            pool = fs[fs["topic_name"] == r["topic"]]
            pick = pool.sample(n=min(cfg["spike_examples_per_topic"], len(pool)), random_state=rng_seed)
            examples.append(pick.assign(spike=f"{app} {week}" + (" (partial)" if sp.get("partial") else ""))[
                ["spike", "app", "week", "topic_name", "at", "score", "reviewId", "text_clean"]])

        # Daily volume window: N days before the spike week through its last day with data
        start = pd.Timestamp(week, tz="UTC") - pd.Timedelta(days=cfg["spike_window_days_before"])
        end = pd.Timestamp(week, tz="UTC") + pd.Timedelta(days=7)
        win = a[(a["at"] >= start) & (a["at"] < end)]
        daily = win.groupby("date").agg(n=("reviewId", "size"), k=("is_complaint", "sum")).reset_index()
        in_week = s_rows.groupby("date").size()
        peak_day = str(in_week.idxmax())
        peak = {"date": peak_day, "n": int(in_week.max()), "share_of_week": float(in_week.max() / n_s),
                "complaint_share": share(int(s_rows.loc[s_rows["date"] == peak_day, "is_complaint"].sum()), int(in_week.max()), ci),
                "rest_of_week_per_day": float((n_s - in_week.max()) / max(days - 1, 1e-9))}

        out.append({
            "app": app, "week": week, "partial": bool(sp.get("partial", False)), "days_of_data": round(days, 2),
            "baseline_weeks": base_weeks, "peak_day": peak,
            "volume": {"spike": n_s, "spike_per_day": n_s / days, "baseline_median_week": base_med_n,
                       "baseline_median_per_day": base_med_n / 7, "ratio_per_day": (n_s / days) / (base_med_n / 7)},
            "complaint_share": {"spike": share(k_s, n_s, ci), "baseline_median_week": base_med_share,
                                "baseline_pooled": share(k_b, n_b, ci),
                                "diff_vs_pooled_pp": 100 * (k_s / n_s - k_b / n_b),
                                "diff_ci95_pp": [100 * x for x in newcombe(k_s, n_s, k_b, n_b, ci)]},
            "stars": stars,
            "topics_fitted_docs": topics,
            "top_rising_specific": [r["topic"] for r in rising],
            "daily": daily.to_dict("records"),
        })
    ex = pd.concat(examples, ignore_index=True) if examples else pd.DataFrame()
    return out, ex


# ---------------------------------------------------------------- 3-5

def hidden_weekly(df: pd.DataFrame, apps: list[str], ci: float, partial: str) -> pd.DataFrame:
    """Weekly share of 4-5 star reviews that PRIMARY's model part flags as complaints (all rows)."""
    pos = df[(df["score"] >= 4) & (~df["is_partial_week"] | (df["week"] == partial))]
    g = pos.groupby(["app", "week"]).agg(n=("reviewId", "size"), k=("pred_label", lambda s: int((s == "negative").sum())),
                                         partial=("is_partial_week", "first")).reset_index()
    g["share"] = g["k"] / g["n"]
    cis = [wilson(int(k), int(n), ci) for k, n in zip(g["k"], g["n"])]
    g["lo"], g["hi"] = [c[0] for c in cis], [c[1] for c in cis]
    return g[g["app"].isin(apps)].sort_values(["app", "week"]).reset_index(drop=True)


def topic_weekly(df: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Weekly share of each app's fitted complaint documents for its top specific topics (complete weeks)."""
    excluded = set(cfg["excluded_topics"]) | {cfg["outlier_label"]}
    f = df[df["is_complaint"] & df["in_fit"] & ~df["is_partial_week"]]
    tops, rows = {}, []
    for app in cfg["apps"]:
        a = f[f["app"] == app]
        spec = a[~a["topic_name"].isin(excluded)]
        tops[app] = spec["topic_name"].value_counts().head(cfg["topics_per_app"]).index.tolist()
        den = a.groupby("week").size()
        for t in tops[app]:
            num = a[a["topic_name"] == t].groupby("week").size().reindex(den.index, fill_value=0)
            for w in den.index:
                lo, hi = wilson(int(num[w]), int(den[w]), cfg["ci"])
                rows.append({"app": app, "topic": t, "week": w, "k": int(num[w]), "n": int(den[w]),
                             "share": num[w] / den[w], "lo": lo, "hi": hi})
    return pd.DataFrame(rows), tops


def versions(df: pd.DataFrame, cfg: dict[str, Any], peak_days: dict[str, set[str]]) -> dict[str, Any]:
    """Top major.minor versions per app by review count, complaint share with Wilson CI (all rows, all weeks),
    plus the same share excluding the app's spike peak days (version adoption can coincide with spikes)."""
    out = {}
    for app in cfg["apps"]:
        a = df[df["app"] == app]
        top = a["version_major_minor"].value_counts().head(cfg["versions_per_app"]).index.tolist()
        peak = a["date"].isin(peak_days.get(app, set()))
        vers = {}
        for v in top:
            m = a["version_major_minor"] == v
            vers[v] = {**share(int(a.loc[m, "is_complaint"].sum()), int(m.sum()), cfg["ci"]),
                       "rows_on_spike_peak_days": float(peak[m].mean()),
                       "excluding_spike_peak_days": share(int(a.loc[m & ~peak, "is_complaint"].sum()), int((m & ~peak).sum()), cfg["ci"])}
        out[app] = {"missing_version_share": float(a["version_major_minor"].isna().mean()),
                    "app_share": share(int(a["is_complaint"].sum()), len(a), cfg["ci"]),
                    "spike_peak_days": sorted(peak_days.get(app, set())), "versions": vers}
    return out


# ---------------------------------------------------------------- figures

def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=8, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)


def _week_labels(weeks: list[str]) -> list[str]:
    return [pd.Timestamp(w).strftime("%b %d").replace(" 0", " ") for w in weeks]


def _header(fig, title: str, subtitle: str) -> None:
    h = fig.get_figheight()
    fig.text(0.01, 0.995, title, color=TEXT_PRIMARY, fontsize=12, va="top")
    fig.text(0.01, 0.995 - 0.3 / h, subtitle, color=TEXT_SECONDARY, fontsize=9, va="top", wrap=True)


def plot_weekly_lines(tbl: pd.DataFrame, cfg: dict[str, Any], path: Path, title: str, subtitle: str, ylabel: str,
                      with_volume: bool, spikes: dict[str, set[str]]) -> None:
    """Per app: share line with Wilson band over complete weeks; the partial week as a hollow marker;
    optional weekly volume bars below (same x, own y axis panel, no dual axis)."""
    import matplotlib.pyplot as plt

    apps = cfg["apps"]
    rows = 2 if with_volume else 1
    fig, axes = plt.subplots(rows, len(apps), figsize=(16, 6.4 if with_volume else 4.4), facecolor=SURFACE,
                             sharex="col", squeeze=False, gridspec_kw={"height_ratios": [2.2, 1] if with_volume else [1]})
    weeks = sorted(tbl["week"].unique())
    x = {w: i for i, w in enumerate(weeks)}
    for j, app in enumerate(apps):
        t = tbl[tbl["app"] == app]
        comp, part = t[~t["partial"]], t[t["partial"]]
        ax = axes[0, j]
        _style(ax)
        for w in spikes.get(app, set()):
            ax.axvspan(x[w] - 0.45, x[w] + 0.45, color="#eeede9", zorder=0)
        xs = [x[w] for w in comp["week"]]
        ax.fill_between(xs, comp["lo"], comp["hi"], color=SERIES[0], alpha=0.15, linewidth=0, zorder=1)
        ax.plot(xs, comp["share"], color=SERIES[0], linewidth=2, marker="o", markersize=4, zorder=3)
        for r in part.itertuples():
            ax.errorbar(x[r.week], r.share, yerr=[[r.share - r.lo], [r.hi - r.share]], fmt="o", mfc=SURFACE,
                        mec=SERIES[0], ecolor=SERIES[0], markersize=6, mew=1.5, zorder=4)
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.set_title(app, color=TEXT_PRIMARY, fontsize=11, loc="left")
        if j == 0:
            ax.set_ylabel(ylabel, color=TEXT_SECONDARY, fontsize=9)
        last = comp.iloc[-1]
        ax.text(x[last["week"]], last["hi"] + 0.03, f"{last['share']:.0%}", ha="center", color=TEXT_PRIMARY, fontsize=8)
        if with_volume:
            bx = axes[1, j]
            _style(bx)
            bx.bar(xs, comp["n"], color="#9ec5f4", width=0.7, zorder=2)
            for r in part.itertuples():
                bx.bar(x[r.week], r.n, color=SURFACE, edgecolor="#9ec5f4", hatch="///", width=0.7, zorder=2)
            bx.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v / 1000:.0f}k" if v >= 1000 else f"{v:.0f}"))
            if j == 0:
                bx.set_ylabel("Reviews per week", color=TEXT_SECONDARY, fontsize=9)
        bottom = axes[rows - 1, j]
        bottom.set_xticks(range(len(weeks)), _week_labels(weeks), rotation=60, ha="right")
    _header(fig, title, subtitle)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.75 / fig.get_figheight()))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_spike_daily(spk: list[dict[str, Any]], path: Path, subtitle: str) -> None:
    """Per spike: daily reviews, complaints stacked on non-complaints; spike week shaded."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    fig, axes = plt.subplots(1, len(spk), figsize=(16, 4.4), facecolor=SURFACE, squeeze=False)
    for ax, s in zip(axes[0], spk):
        _style(ax)
        d = pd.DataFrame(s["daily"])
        xs = np.arange(len(d))
        in_week = d["date"] >= s["week"]
        if in_week.any():
            first = int(np.flatnonzero(in_week.to_numpy())[0])
            ax.axvspan(first - 0.5, len(d) - 0.5, color="#eeede9", zorder=0)
        ax.bar(xs, d["k"], color=SERIES[0], width=0.75, zorder=2)
        ax.bar(xs, d["n"] - d["k"], bottom=d["k"], color=MUTED, width=0.75, zorder=2)
        ax.set_xticks(xs, [pd.Timestamp(v).strftime("%b %d").replace(" 0", " ") for v in d["date"]], rotation=70, ha="right")
        lab = f"{s['app']} · week of {pd.Timestamp(s['week']).strftime('%b %d')}" + (" (partial)" if s["partial"] else "")
        ax.set_title(f"{lab}\n×{s['volume']['ratio_per_day']:.1f} reviews/day, complaints {s['complaint_share']['spike']['share']:.0%}",
                     color=TEXT_PRIMARY, fontsize=10, loc="left")
    axes[0, 0].set_ylabel("Reviews per day (UTC)", color=TEXT_SECONDARY, fontsize=9)
    fig.legend(handles=[Patch(color=SERIES[0], label="complaint (PRIMARY)"), Patch(color=MUTED, label="not a complaint")],
               frameon=False, labelcolor=TEXT_PRIMARY, loc="upper right", ncol=2, fontsize=9)
    _header(fig, "Phase 7: daily reviews around each spike week (shaded)", subtitle)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.75 / fig.get_figheight()))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_spike_topics(spk: list[dict[str, Any]], path: Path, subtitle: str, top: int = 6) -> None:
    """Per spike: specific topics with the largest change in share of fitted complaint docs (pp, Newcombe CI)."""
    import matplotlib.pyplot as plt

    ncol = 2
    nrow = int(np.ceil(len(spk) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(15, 3.6 * nrow + 1.0), facecolor=SURFACE, squeeze=False)
    for ax in axes.flat[len(spk):]:
        ax.set_visible(False)
    for ax, s in zip(axes.flat, spk):
        _style(ax)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
        rows = [r for r in s["topics_fitted_docs"] if r["specific"]][:top]
        ys = np.arange(len(rows))
        vals = np.array([r["diff_pp"] for r in rows])
        lo = np.array([r["diff_ci95_pp"][0] for r in rows])
        hi = np.array([r["diff_ci95_pp"][1] for r in rows])
        ax.barh(ys, vals, color=[SERIES[0] if v > 0 else MUTED for v in vals], height=0.65, zorder=2)
        ax.errorbar(vals, ys, xerr=[vals - lo, hi - vals], fmt="none", ecolor=TEXT_SECONDARY, elinewidth=1, capsize=2, zorder=3)
        ax.axvline(0, color=TEXT_SECONDARY, linewidth=0.8)
        ax.set_yticks(ys, [r["topic"] for r in rows], fontsize=8)
        ax.invert_yaxis()
        for y, r in zip(ys, rows):
            ax.text(max(r["diff_ci95_pp"][1], 0) + 0.3, y, f"{r['share_base']:.1%}→{r['share_spike']:.1%} ({r['k_spike']})",
                    va="center", color=TEXT_SECONDARY, fontsize=7.5)
        ax.set_xlabel("Change in share of complaints (pp)", color=TEXT_SECONDARY, fontsize=8)
        lab = f"{s['app']} · week of {pd.Timestamp(s['week']).strftime('%b %d')}" + (" (partial)" if s["partial"] else "")
        ax.set_title(lab, color=TEXT_PRIMARY, fontsize=10, loc="left")
        xmax = max(hi.max(), 1) * 1.45 + 2
        ax.set_xlim(min(lo.min(), 0) - 1, xmax)
    _header(fig, "Phase 7: specific topics that gained share of complaints in each spike week", subtitle)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.75 / fig.get_figheight()))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_topic_trends(tt: pd.DataFrame, tops: dict[str, list[str]], cfg: dict[str, Any], path: Path, subtitle: str) -> dict[str, str]:
    """Per app: weekly share of fitted complaint docs for its top specific topics; one fixed color per topic."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    order = tt.groupby("topic")["k"].sum().sort_values(ascending=False).index.tolist()
    # Fixed palette slots for the 8 most frequent topics; any further topic gets a neutral dashed
    # line (composite encoding) instead of a generated 9th hue
    color = {t: CATEGORICAL[i] if i < len(CATEGORICAL) else TEXT_SECONDARY for i, t in enumerate(order)}
    style = {t: "-" if i < len(CATEGORICAL) else (0, (3, 2)) for i, t in enumerate(order)}
    weeks = sorted(tt["week"].unique())
    fig, axes = plt.subplots(1, len(cfg["apps"]), figsize=(16, 4.8), facecolor=SURFACE, squeeze=False, sharey=True)
    for ax, app in zip(axes[0], cfg["apps"]):
        _style(ax)
        for t in tops[app]:
            d = tt[(tt["app"] == app) & (tt["topic"] == t)].sort_values("week")
            ax.plot(range(len(d)), d["share"], color=color[t], linestyle=style[t], linewidth=2, marker="o", markersize=3.5, zorder=3)
        n = tt[tt["app"] == app].groupby("week")["n"].first()
        ax.set_title(f"{app}  (fitted complaint docs/week: {int(n.min()):,}–{int(n.max()):,})", color=TEXT_PRIMARY, fontsize=10, loc="left")
        ax.set_xticks(range(len(weeks)), _week_labels(weeks), rotation=60, ha="right")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    axes[0, 0].set_ylabel("Share of the app's complaints", color=TEXT_SECONDARY, fontsize=9)
    fig.legend(handles=[Line2D([], [], color=color[t], linestyle=style[t], linewidth=2.5, label=t) for t in order], frameon=False,
               labelcolor=TEXT_PRIMARY, loc="lower center", ncol=4, fontsize=8.5, bbox_to_anchor=(0.5, 0.0))
    _header(fig, "Phase 7: top 5 specific complaint topics per app, weekly", subtitle)
    fig.tight_layout(rect=(0, 0.11, 1, 1 - 0.8 / fig.get_figheight()))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return color


def plot_versions(ver: dict[str, Any], cfg: dict[str, Any], path: Path, subtitle: str) -> None:
    """Per app: complaint share by installed major.minor version (Wilson CI), app average as a dashed line."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(cfg["apps"]), figsize=(18, 3.9), facecolor=SURFACE, squeeze=False)
    for ax, app in zip(axes[0], cfg["apps"]):
        _style(ax)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
        v = ver[app]["versions"]
        names = list(v)
        ys = np.arange(len(names))
        for y, nm in zip(ys, names):
            s = v[nm]
            ax.plot(s["ci95"], [y, y], color=SERIES[0], linewidth=2, alpha=0.5, solid_capstyle="round")
            ax.plot(s["share"], y, "o", color=SERIES[0], markersize=8, markeredgecolor=SURFACE, markeredgewidth=2)
            label = f"{s['share']:.0%}, n={s['n']:,}\nexcl. spike days {s['excluding_spike_peak_days']['share']:.0%}"
            ax.text(s["ci95"][1] + 0.035, y, label, va="center", color=TEXT_SECONDARY, fontsize=7.5, linespacing=1.1)
        ax.axvline(ver[app]["app_share"]["share"], color=TEXT_SECONDARY, linestyle=(0, (3, 3)), linewidth=1)
        ax.set_yticks(ys, [f"v{n}" for n in names])
        ax.set_xlim(0, 1.4)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
        ax.set_title(f"{app}  (version missing {ver[app]['missing_version_share']:.0%})", color=TEXT_PRIMARY, fontsize=10, loc="left")
        ax.set_ylim(len(names) - 0.4, -0.6)
    _header(fig, "Phase 7 (secondary): complaint share by installed app version", subtitle)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.75 / fig.get_figheight()))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------- main

def main() -> None:  # noqa: PLR0915 - linear analysis script
    """CLI entry point."""
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path("configs/trends.yaml"))
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    cfg = load_config(args.config)
    np.random.seed(cfg["seed"])
    ci, apps, partial = cfg["ci"], cfg["apps"], cfg["partial_week_shown"]
    fig_dir = Path(cfg["figures_dir"])
    fig_dir.mkdir(parents=True, exist_ok=True)
    figs = {k: fig_dir / v for k, v in cfg["figures"].items()}

    df = load(cfg)
    complete = sorted(df.loc[~df["is_partial_week"], "week"].unique())
    logger.info("seed=%d rows=%d complete weeks %s..%s (%d); partial shown %s", cfg["seed"], len(df), complete[0],
                complete[-1], len(complete), partial)
    spikes_by_app: dict[str, set[str]] = {}
    for s in cfg["spikes"]:
        spikes_by_app.setdefault(s["app"], set()).add(s["week"])

    # 1. weekly complaint share (all rows)
    wk = weekly(df, apps, ci, partial)
    comp_rows = df[~df["is_partial_week"]]
    overall = {a: share(int(comp_rows.loc[comp_rows["app"] == a, "is_complaint"].sum()), int((comp_rows["app"] == a).sum()), ci)
               for a in apps}
    days_partial = week_days(df, partial)
    plot_weekly_lines(wk, cfg, figs["weekly"], "Phase 7: weekly complaint share and review volume per e-wallet",
                      f"All reviews (duplicates included), PRIMARY complaints; line = complete Mon–Sun weeks with 95% Wilson band; "
                      f"hollow = partial week of Sep 28 ({days_partial:.1f} days, hatched volume); shaded = spike weeks. UTC.",
                      "Complaint share", True, spikes_by_app)

    # 2. spikes
    spk, ex = spike_analysis(df, wk, cfg)
    ex.to_csv(cfg["examples_path"], index=False, encoding="utf-8-sig")
    plot_spike_daily(spk, figs["spike_daily"], "Daily reviews, complaints stacked on non-complaints; window = 7 days before "
                     "the spike week through its end (UTC). Ratio = reviews per day vs the app's median non-spike week.")
    plot_spike_topics(spk, figs["spike_topics"], "Fitted documents (unique, non-short complaint texts); change vs the app's "
                      "non-spike complete weeks pooled; whiskers = 95% Newcombe CI; label = baseline→spike share (spike count). "
                      "Specific topics only.")

    # 3. hidden complaints
    hd = hidden_weekly(df, apps, ci, partial)
    plot_weekly_lines(hd, cfg, figs["hidden"], "Phase 7: hidden complaints — 4–5★ reviews the text model flags as complaints",
                      "Share of 4–5★ reviews (all rows) with PRIMARY's model part = negative; 95% Wilson band; hollow = partial "
                      "week of Sep 28; shaded = spike weeks. On gold, the model caught 49% [31, 66] of 4–5★ complaints.",
                      "Share of 4–5★ reviews", False, spikes_by_app)

    # 4. topic trends
    tt, tops = topic_weekly(df, cfg)
    colors = plot_topic_trends(tt, tops, cfg, figs["topics"], "Specific topics only (excludes outages, app bugs, unspecified, "
                               "outliers); fitted documents (unique, non-short), complete weeks; share of all the app's fitted "
                               "complaint docs. " + cfg["fit_caveat"])

    # 5. versions
    peak_days: dict[str, set[str]] = {}
    for sp in spk:
        peak_days.setdefault(sp["app"], set()).add(sp["peak_day"]["date"])
    ver = versions(df, cfg, peak_days)
    plot_versions(ver, cfg, figs["versions"], "All reviews, all weeks; 95% Wilson CI; dashed = app average. Versions are installed "
                  "versions, not release timing (many users run old versions); a new version can coincide with a spike day.")

    report = {
        "seed": cfg["seed"], "timezone": cfg["timezone_note"], "complete_weeks": complete,
        "partial_week_shown": {"week": partial, "days_of_data": round(days_partial, 2)},
        "denominators": {"complaint_share": "all rows (duplicates included)",
                         "topic_share": "fitted documents (unique, non-short complaint texts)",
                         "hidden": "all 4–5★ rows"},
        "overall_complaint_share_complete_weeks": overall,
        "weekly": wk.to_dict("records"),
        "spikes": spk,
        "spike_examples": {"path": cfg["examples_path"], "rows": int(len(ex))},
        "hidden_weekly": hd.to_dict("records"),
        "hidden_overall_complete_weeks": {a: share(int(hd[(hd["app"] == a) & ~hd["partial"]]["k"].sum()),
                                                   int(hd[(hd["app"] == a) & ~hd["partial"]]["n"].sum()), ci) for a in apps},
        "topic_trends": {"top_topics": tops, "colors": colors, "weekly": tt.to_dict("records")},
        "versions": ver,
        "figures": {k: str(v) for k, v in figs.items()},
    }
    rp = Path(cfg["report_path"])
    rp.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    if not args.no_wandb:
        report["wandb_url"] = log_wandb(report, cfg, figs, wk)
        rp.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    logger.info("wrote %s, %s, figures", rp, cfg["examples_path"])
    print_summary(report)


def log_wandb(report: dict[str, Any], cfg: dict[str, Any], figs: dict[str, Path], wk: pd.DataFrame) -> str | None:
    """One W&B run: figures, weekly table, spike summary."""
    import wandb

    from src.train_indobert import wandb_logged_in

    if not wandb_logged_in():
        logger.warning("W&B not logged in; skipping")
        return None
    icfg = load_config(Path("configs/indobert.yaml"))
    try:
        run = wandb.init(project=icfg["wandb"]["project"], entity=icfg["wandb"]["entity"], name=cfg["wandb"]["name"],
                         job_type=cfg["wandb"]["job_type"], tags=["phase7", "trends"], config={"seed": cfg["seed"], "spikes": cfg["spikes"]})
        spikes = pd.DataFrame([{"app": s["app"], "week": s["week"], "partial": s["partial"],
                                "ratio_per_day": s["volume"]["ratio_per_day"],
                                "complaint_share": s["complaint_share"]["spike"]["share"],
                                "baseline_median_share": s["complaint_share"]["baseline_median_week"],
                                "top_rising": ", ".join(s["top_rising_specific"])} for s in report["spikes"]])
        run.log({"weekly_complaint_share": wandb.Table(dataframe=wk), "spikes": wandb.Table(dataframe=spikes),
                 **{k: wandb.Image(str(v)) for k, v in figs.items()}})
        url = run.url
        run.finish()
        return url
    except Exception as exc:  # noqa: BLE001 - outputs are saved; W&B is a mirror
        logger.warning("W&B logging failed: %s", exc)
        return f"failed: {exc}"


def print_summary(r: dict[str, Any]) -> None:
    """Console summary for writing the findings."""
    print("\n== overall complaint share, complete weeks (all rows) ==")
    for a, s in r["overall_complaint_share_complete_weeks"].items():
        print(f"   {a:<10} {s['share']:.1%} [{s['ci95'][0]:.1%}, {s['ci95'][1]:.1%}]  n={s['n']:,}")
    print("\n== spikes ==")
    for s in r["spikes"]:
        v, c = s["volume"], s["complaint_share"]
        print(f"   {s['app']} {s['week']}{' (partial)' if s['partial'] else ''}: {v['spike']:,} reviews "
              f"({v['spike_per_day']:.0f}/day vs median {v['baseline_median_per_day']:.0f}/day, x{v['ratio_per_day']:.2f}); "
              f"complaints {c['spike']['share']:.1%} [{c['spike']['ci95'][0]:.1%}, {c['spike']['ci95'][1]:.1%}] vs median week "
              f"{c['baseline_median_week']:.1%} (pooled diff {c['diff_vs_pooled_pp']:+.1f} pp [{c['diff_ci95_pp'][0]:+.1f}, {c['diff_ci95_pp'][1]:+.1f}])")
        pk = s["peak_day"]
        print(f"      peak day {pk['date']}: {pk['n']:,} reviews ({pk['share_of_week']:.0%} of the week), complaints "
              f"{pk['complaint_share']['share']:.1%}; rest of week {pk['rest_of_week_per_day']:.0f}/day")
        print("      stars 1★/5★ diff pp:", f"{s['stars'][1]['diff_pp']:+.1f} / {s['stars'][5]['diff_pp']:+.1f}",
              "| 1★ share", f"{s['stars'][1]['baseline']:.1%}→{s['stars'][1]['spike']:.1%}")
        for t in [t for t in s["topics_fitted_docs"]][:4] + [t for t in s["topics_fitted_docs"] if t["specific"]][:4]:
            print(f"      {'*' if t['specific'] else ' '} {t['topic'][:44]:<44} {t['share_base']:6.1%} -> {t['share_spike']:6.1%} "
                  f"({t['diff_pp']:+5.1f} pp [{t['diff_ci95_pp'][0]:+.1f}, {t['diff_ci95_pp'][1]:+.1f}]) k={t['k_spike']}")
        print("      daily:", ", ".join(f"{d['date'][5:]}:{d['n']}" for d in s["daily"]))
    print("\n== hidden complaints (4–5★ flagged by model), complete weeks ==")
    for a, s in r["hidden_overall_complete_weeks"].items():
        wk = [h for h in r["hidden_weekly"] if h["app"] == a]
        print(f"   {a:<10} {s['share']:.1%} [{s['ci95'][0]:.1%}, {s['ci95'][1]:.1%}]  weekly range "
              f"{min(h['share'] for h in wk if not h['partial']):.1%}–{max(h['share'] for h in wk if not h['partial']):.1%}; "
              f"partial {[round(h['share'], 3) for h in wk if h['partial']]}")
    print("\n== top specific topics per app ==", json.dumps(r["topic_trends"]["top_topics"], ensure_ascii=False))
    print("\n== versions ==")
    for a, v in r["versions"].items():
        print(f"   {a:<10} app {v['app_share']['share']:.1%}; " + "; ".join(
            f"v{n}: {s['share']:.1%} [{s['ci95'][0]:.1%}, {s['ci95'][1]:.1%}] n={s['n']:,} (spike days {s['rows_on_spike_peak_days']:.0%}; "
            f"excl. {s['excluding_spike_peak_days']['share']:.1%} [{s['excluding_spike_peak_days']['ci95'][0]:.1%}, "
            f"{s['excluding_spike_peak_days']['ci95'][1]:.1%}])" for n, s in v["versions"].items()))


if __name__ == "__main__":
    main()
