"""Phase 6b: apply manual topic names + merges, refresh keywords, final topic table.

Steps:
    1. validate data/topics/topics_to_name.xlsx (names / merge_into; no chains or cycles)
    2. merge topics in the BERTopic model and in topic_assignments.parquet; check that
       assignments change only by the merges
    3. refresh keywords with extra stopwords (update_topics; assignments unchanged)
    4. final topic table: size, share of complaints per app, keywords, over-represented topics
    5. export data/topics/topic_fit_check.xlsx (60 rows, 15 per app) for a manual fit check
    6. figures, reports/phase6_final_topics.json, W&B
    --fit-rate: score the filled fit check (yes / partly / no with Wilson CIs) into the report

No gold or dev labels are read.

Usage:
    uv run python -m src.topics_final --config configs/topics_final.yaml --validate-only
    uv run python -m src.topics_final --config configs/topics_final.yaml
    uv run python -m src.topics_final --config configs/topics_final.yaml --fit-rate
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

from src import topics as tp
from src.gold_report import GRID, SERIES, SURFACE, TEXT_PRIMARY, TEXT_SECONDARY

logger = logging.getLogger("topics_final")


def load_config(path: Path) -> dict[str, Any]:
    """Load a YAML config."""
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------- 1. validation

def read_names(cfg: dict[str, Any]) -> pd.DataFrame:
    """topic_id, topic_name, merge_into from the naming workbook (blank -> None)."""
    return pd.read_excel(cfg["workbook"], sheet_name=cfg["sheet"])[["topic_id", "topic_name", "merge_into", "notes"]]


def _clean_name(v: Any) -> str | None:
    """Stripped name, or None for blank cells (which arrive as NaN)."""
    return v.strip() if isinstance(v, str) and v.strip() else None


def validate(names: pd.DataFrame, model_topics: set[int]) -> tuple[list[str], dict[int, int], dict[int, str]]:
    """Return (errors, merge map source -> target, names of kept topics)."""
    errors: list[str] = []
    ids = names["topic_id"].tolist()
    if sorted(ids) != sorted(model_topics) or len(ids) != len(set(ids)):
        errors.append(f"workbook topic_ids {sorted(ids)} differ from the model's {sorted(model_topics)}")
    merge: dict[int, int] = {}
    named: dict[int, str] = {}
    for r in names.itertuples():
        t = int(r.topic_id)
        m = r.merge_into
        name = _clean_name(r.topic_name)
        has_merge = m is not None and not (isinstance(m, float) and np.isnan(m)) and str(m).strip() != ""
        if t == -1:
            if has_merge:
                errors.append("topic -1 (outliers) has merge_into; outliers are not merged")
            continue
        if has_merge:
            try:
                mf = float(m)
            except (TypeError, ValueError):
                errors.append(f"topic {t}: merge_into {m!r} is not a topic id")
                continue
            if not mf.is_integer():
                errors.append(f"topic {t}: merge_into {m!r} is not an integer")
                continue
            merge[t] = int(mf)
        if name:
            named[t] = name
        if has_merge and name:
            errors.append(f"topic {t}: has both topic_name ({name!r}) and merge_into ({m}); keep one")
        if not has_merge and not name:
            errors.append(f"topic {t}: needs topic_name or merge_into")
    for s, d in merge.items():
        if d == s:
            errors.append(f"topic {s}: merges into itself")
        elif d == -1:
            errors.append(f"topic {s}: merges into -1 (outliers)")
        elif d not in model_topics:
            errors.append(f"topic {s}: merge target {d} does not exist")
        elif d in merge:
            errors.append(f"topic {s}: merge target {d} is itself merged into {merge[d]} (chain/cycle)")
        elif d not in named:
            errors.append(f"topic {s}: merge target {d} has no topic_name")
    dup = pd.Series(named).loc[lambda s: s.duplicated(keep=False)]
    if len(dup):
        errors.append(f"duplicate topic names (use merge_into instead): {dup.to_dict()}")
    kept = {t: n for t, n in named.items() if t not in merge}
    return errors, merge, kept


# ---------------------------------------------------------------- 2-3. model + assignments

def apply_merges(model, docs: list[str], merge: dict[int, int]) -> dict[int, int]:
    """Merge topics in the model; return the 6a -> final id mapping (final ids renumbered by size)."""
    old = np.asarray(model.topics_)
    groups: dict[int, list[int]] = {}
    for s, d in sorted(merge.items()):
        groups.setdefault(d, [d]).append(s)
    if groups:
        model.merge_topics(docs, [g for _, g in sorted(groups.items())])
    new = np.asarray(model.topics_)
    pairs = pd.DataFrame({"old": old, "new": new}).drop_duplicates()
    if pairs["old"].duplicated().any():
        raise RuntimeError("a 6a topic was split by merge_topics")
    mapping = dict(zip(pairs["old"].astype(int), pairs["new"].astype(int)))
    # The partition must equal the requested merges: same final id <=> same target
    target = {t: merge.get(t, t) for t in mapping}
    by_target = pd.Series({t: mapping[t] for t in mapping}).groupby(pd.Series(target)).nunique()
    if (by_target != 1).any() or pd.Series(mapping).groupby(pd.Series(mapping)).size().sum() != len(mapping):
        raise RuntimeError("merge result does not match the requested merges")
    if pd.Series({target[t]: mapping[t] for t in mapping}).duplicated().any():
        raise RuntimeError("two merge groups ended up in the same final topic")
    return mapping


def refresh_keywords(model, docs: list[str], tcfg: dict[str, Any], cfg: dict[str, Any]) -> list[str]:
    """update_topics with the extended stopword list; topic assignments must not change."""
    from sklearn.feature_extraction.text import CountVectorizer

    stop = sorted(set(tp.stopwords(tcfg)) | set(cfg["extra_stopwords"]))
    clash = set(stop) & set(cfg["keep_words"])
    if clash:
        raise ValueError(f"keep_words are in the stopword list: {clash}")
    before = list(model.topics_)
    v = tcfg["vectorizer"]
    model.update_topics(docs, vectorizer_model=CountVectorizer(ngram_range=tuple(v["ngram_range"]), min_df=v["min_df"],
                                                               stop_words=stop),
                        top_n_words=tcfg["top_n_words"])
    if list(model.topics_) != before:
        raise RuntimeError("update_topics changed topic assignments")
    return stop


# ---------------------------------------------------------------- 4. tables

def share_frame(assign: pd.DataFrame, apps: list[str]) -> dict[str, pd.DataFrame]:
    """Per basis: rows per (topic, app) and share of that app's complaint rows (outliers in the denominator)."""
    out = {}
    for basis, part in (("unique_texts", assign[assign["in_model_pool"]]), ("all_rows", assign)):
        counts = pd.crosstab(part["topic"], part["app"]).reindex(columns=apps, fill_value=0)
        counts["all"] = counts.sum(axis=1)
        out[basis] = counts
    return out


def overrepresented(counts: pd.DataFrame, apps: list[str], top_n: int, min_rows: int, mode: str) -> dict[str, list[dict[str, Any]]]:
    """Per app: topics with the highest ratio of the app's share to a reference share.

    mode "leave_one_out": reference = the topic's share of the other three apps' complaints combined.
    mode "pooled": reference = the topic's share of all apps' complaints (DANA dominates it).
    Shares use all complaint rows of the app as denominator (outliers included); outliers are never ranked.
    """
    c = counts.drop(index=-1, errors="ignore")
    out = {}
    for a in apps:
        share_app = counts[a] / counts[a].sum()
        if mode == "leave_one_out":
            ref_counts = counts[[x for x in apps if x != a]].sum(axis=1)
        elif mode == "pooled":
            ref_counts = counts[apps].sum(axis=1)
        else:
            raise ValueError(f"unknown mode {mode!r}")
        share_ref = ref_counts / ref_counts.sum()
        ratio = (share_app.loc[c.index] / share_ref.loc[c.index]).where(c[a] >= min_rows).dropna()
        out[a] = [{"topic": int(t), "ratio": round(float(r), 2), "share_in_app": round(float(share_app.loc[t]), 4),
                   "share_reference": round(float(share_ref.loc[t]), 4), "rows_in_app": int(c.loc[t, a])}
                  for t, r in ratio.sort_values(ascending=False).head(top_n).items()]
    return out


def plot_overrepresented(over: dict[str, list[dict[str, Any]]], apps: list[str], path: Path, subtitle: str) -> None:
    """Dumbbells per app: the app's share vs the other three apps' share for its most over-represented topics."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    n = max(len(v) for v in over.values())
    fig, axes = plt.subplots(len(apps), 1, figsize=(10, 0.55 * n * len(apps) + 1.2 * len(apps) + 0.6), facecolor=SURFACE)
    xmax = max(max(e["share_in_app"], e["share_reference"]) for v in over.values() for e in v) * 1.35
    for ax, a in zip(axes, apps):
        ax.set_facecolor(SURFACE)
        rows = over[a]
        for yi, e in enumerate(rows):
            lo, hi = sorted((e["share_reference"], e["share_in_app"]))
            ax.plot([lo, hi], [yi, yi], color=GRID, linewidth=3, solid_capstyle="round", zorder=1)
            ax.plot(e["share_reference"], yi, "o", color=TEXT_SECONDARY, markersize=8, markeredgecolor=SURFACE,
                    markeredgewidth=2, zorder=2)
            ax.plot(e["share_in_app"], yi, "o", color=SERIES[0], markersize=9, markeredgecolor=SURFACE,
                    markeredgewidth=2, zorder=3)
            ax.text(hi + xmax * 0.012, yi, f"{e['share_in_app']:.1%} vs {e['share_reference']:.1%}  (x{e['ratio']:.1f})",
                    va="center", color=TEXT_PRIMARY, fontsize=9)
        ax.set_yticks(range(len(rows)), [e["topic_name"] for e in rows], fontsize=9)
        ax.set_ylim(len(rows) - 0.5, -0.6)
        ax.set_xlim(0, xmax)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(colors=TEXT_SECONDARY, length=0, labelsize=9)
        ax.set_title(a, color=TEXT_PRIMARY, fontsize=11, loc="left")
    handles = [Line2D([], [], marker="o", color=SERIES[0], linewidth=0, markersize=8, label="share of this app's complaints"),
               Line2D([], [], marker="o", color=TEXT_SECONDARY, linewidth=0, markersize=8, label="share in the other three apps")]
    h = fig.get_figheight()
    fig.text(0.01, 0.995, "Phase 6b: complaint topics most specific to each e-wallet (leave-one-out)",
             color=TEXT_PRIMARY, fontsize=12, va="top")
    fig.text(0.01, 0.995 - 0.28 / h, subtitle, color=TEXT_SECONDARY, fontsize=9, va="top")
    fig.legend(handles=handles, frameon=False, labelcolor=TEXT_PRIMARY, loc="upper left",
               bbox_to_anchor=(0.005, 1 - 0.5 / h), ncol=2, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.85 / h))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------- 5. fit check

def export_fit_check(assign: pd.DataFrame, texts: pd.Series, labels: dict[int, str], cfg: dict[str, Any], apps: list[str]) -> pd.DataFrame:
    """60 random complaint rows (15 per app, distinct texts, named topics only) with a yes/partly/no dropdown."""
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.worksheet.datavalidation import DataValidation

    fc = cfg["fit_check"]
    if not all(isinstance(o, str) for o in fc["options"]):
        raise ValueError(f"fit_check.options must be strings (quote yes/no in YAML): {fc['options']}")
    xlsx, key_path = Path(fc["xlsx_path"]), Path(fc["key_path"])
    for p in (xlsx, key_path):
        if p.exists():
            raise FileExistsError(f"{p} exists; refusing to overwrite (it may contain your judgments)")
    pool = assign[assign["topic"] != -1].assign(text_clean=lambda d: d["reviewId"].map(texts))
    pool = pool.drop_duplicates("text_clean")
    pick = pd.concat([pool[pool["app"] == a].sample(n=fc["per_app"], random_state=cfg["seed"]) for a in apps])
    pick = pick.sample(frac=1, random_state=cfg["seed"]).reset_index(drop=True)
    pick.insert(0, "check_id", [f"{fc['id_prefix']}{i:0{fc['id_width']}d}" for i in range(1, len(pick) + 1)])
    pick["topic_name"] = pick["topic"].map(labels)

    wb = Workbook()
    ws = wb.active
    ws.title = "fit_check"
    cols = ["check_id", "app", "text_clean", "topic_name", "fits", "notes"]
    ws.append(cols)
    for r in pick.itertuples():
        ws.append([r.check_id, r.app, r.text_clean, r.topic_name, None, None])
    for col, width in zip("ABCDEF", (10, 11, 80, 34, 9, 30)):
        ws.column_dimensions[col].width = width
    for c in ws[1]:
        c.font = Font(bold=True)
    for col in ("E", "F"):
        ws[f"{col}1"].fill = PatternFill("solid", fgColor="FFF6D5")
    for row in ws.iter_rows(min_row=2):
        for c in row:
            c.alignment = Alignment(wrap_text=True, vertical="top")
    dv = DataValidation(type="list", formula1='"' + ",".join(fc["options"]) + '"', allow_blank=True,
                        showErrorMessage=True, error="Choose " + " / ".join(fc["options"]))
    ws.add_data_validation(dv)
    dv.add(f"E2:E{len(pick) + 1}")
    ws.freeze_panes = "A2"
    info = wb.create_sheet("README")
    for line in ["Does the topic_name describe what the review complains about?",
                 "fits: yes = the topic is the main complaint; partly = related or one of several complaints; "
                 "no = wrong topic.", "15 random complaint rows per app (seed 42), distinct texts, outliers excluded."]:
        info.append([line])
    info.column_dimensions["A"].width = 110
    xlsx.parent.mkdir(parents=True, exist_ok=True)
    wb.save(xlsx)
    key = pick[["check_id", "reviewId", "app", "topic", "topic_name", "assigned_by_transform", "in_fit", "is_short", "in_model_pool"]]
    key.to_parquet(key_path, index=False)
    return key


# ---------------------------------------------------------------- 6. figures

def plot_share_per_app(counts: pd.DataFrame, labels: dict[int, str], apps: list[str], top_n: int, path: Path, subtitle: str) -> None:
    """Small multiples: share of each app's complaints by topic (top-n overall + Other incl. outliers)."""
    import matplotlib.pyplot as plt

    share = counts / counts.sum(axis=0)
    top = counts.drop(index=-1, errors="ignore")["all"].sort_values(ascending=False).head(top_n).index.tolist()
    rows = [labels[t] for t in top] + ["Other topics + outliers"]
    data = pd.concat([share.loc[top], (1 - share.loc[top].sum(axis=0)).to_frame("other").T])
    panels = ["all"] + apps
    fig, axes = plt.subplots(1, len(panels), figsize=(3.0 * len(panels) + 3.0, 0.42 * len(rows) + 1.9),
                             facecolor=SURFACE, sharey=True)
    y = np.arange(len(rows))
    xmax = float(data[panels].to_numpy().max()) * 1.3
    for ax, p in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        vals = data[p].to_numpy()
        colors = [SERIES[0]] * len(top) + [TEXT_SECONDARY]
        ax.barh(y, vals, color=colors, height=0.7, edgecolor=SURFACE, linewidth=2, zorder=2)
        for yi, v in zip(y, vals):
            ax.text(v, yi, f" {v:.0%}", va="center", color=TEXT_SECONDARY, fontsize=8)
        ax.set_xlim(0, xmax)
        ax.set_ylim(len(rows) - 0.5, -0.5)
        ax.set_xticks([])
        for s in ("top", "right", "bottom"):
            ax.spines[s].set_visible(False)
        ax.spines["left"].set_color(GRID)
        ax.tick_params(colors=TEXT_SECONDARY, length=0)
        n = int(counts[p].sum())
        ax.set_title(f"{'All apps' if p == 'all' else p}\n{n:,} complaints", color=TEXT_PRIMARY, fontsize=10, loc="left")
    axes[0].set_yticks(y, rows, fontsize=9)
    h = fig.get_figheight()
    fig.text(0.01, 0.995, "Phase 6b: what users complain about, per e-wallet (share of each app's complaints)",
             color=TEXT_PRIMARY, fontsize=12, va="top")
    fig.text(0.01, 0.995 - 0.3 / h, subtitle, color=TEXT_SECONDARY, fontsize=9, va="top")
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.55 / h))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------- fit rate (after manual judgment)

def wilson(k: int, n: int, ci: float) -> list[float]:
    """Wilson score interval for k successes out of n."""
    from scipy.stats import norm

    if n == 0:
        return [float("nan"), float("nan")]
    z = float(norm.ppf(0.5 + ci / 2))
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [float(centre - half), float(centre + half)]


def fit_shares(df: pd.DataFrame, options: list[str], ci: float) -> dict[str, Any]:
    """n, and count / share / Wilson CI for each fit option."""
    n = int(len(df))
    out: dict[str, Any] = {"n": n}
    for o in options:
        k = int((df["fits"] == o).sum())
        out[o] = {"count": k, "share": k / n if n else float("nan"), "ci95": wilson(k, n, ci)}
    return out


def fit_rate(cfg: dict[str, Any], tcfg: dict[str, Any]) -> dict[str, Any]:
    """Validate the filled fit-check workbook against its key and compute fit shares."""
    fc = cfg["fit_check"]
    options, apps = fc["options"], tcfg["apps"]
    sheet = pd.read_excel(fc["xlsx_path"], sheet_name="fit_check")
    key = pd.read_parquet(fc["key_path"])
    errors = []
    if sheet["check_id"].tolist() != key["check_id"].tolist():
        errors.append("check_id order/content differs from the key")
    if (sheet["topic_name"].to_numpy() != key["topic_name"].to_numpy()).any():
        errors.append("topic_name column was changed")
    sheet["fits"] = sheet["fits"].map(lambda v: v.strip().lower() if isinstance(v, str) and v.strip() else None)
    if sheet["fits"].isna().any():
        errors.append(f"rows without a fits value: {sheet.loc[sheet['fits'].isna(), 'check_id'].tolist()}")
    bad = sheet[sheet["fits"].notna() & ~sheet["fits"].isin(options)]
    if len(bad):
        errors.append(f"unknown fits values: {bad[['check_id', 'fits']].values.tolist()}")
    if errors:
        raise ValueError("fit check failed validation: " + "; ".join(errors))
    df = key.merge(sheet[["check_id", "fits", "notes", "text_clean"]], on="check_id", validate="one_to_one")
    df["notes"] = df["notes"].map(lambda v: v.strip() if isinstance(v, str) and v.strip() else "")
    df["group"] = np.where(df["topic_name"].isin(fc["broad_topics"]), "broad", "specific")

    # Reweighted overall: each app weighted by its share of the sampling frame
    # (complaint rows with a named topic, one row per distinct text)
    assign = pd.read_parquet(cfg["assignments_path"], columns=["reviewId", "app", "topic"])
    texts = pd.read_parquet(tcfg["input_path"], columns=["reviewId", tcfg["text_col"]])
    frame = assign[assign["topic"] != -1].merge(texts, on="reviewId").drop_duplicates(tcfg["text_col"])
    w = frame["app"].value_counts(normalize=True).reindex(apps)
    per_app = {a: df[df["app"] == a] for a in apps}

    def rew(dfs: dict[str, pd.DataFrame], o: str) -> float:
        return float(sum(w[a] * (d["fits"] == o).mean() for a, d in dfs.items()))

    rng = np.random.default_rng(cfg["seed"])
    boot = {o: [] for o in options}
    for _ in range(fc["bootstrap_n"]):
        res = {a: d.iloc[rng.integers(0, len(d), len(d))] for a, d in per_app.items()}
        for o in options:
            boot[o].append(rew(res, o))
    q = [(1 - fc["ci"]) / 2 * 100, (1 + fc["ci"]) / 2 * 100]
    reweighted = {o: {"share": rew(per_app, o), "ci95": [float(x) for x in np.percentile(boot[o], q)]} for o in options}

    tag = fc["not_complaint_tag"].lower()
    tagged = df[df["notes"].str.lower().str.contains(tag, regex=False)]
    possible = df[~df.index.isin(tagged.index)
                  & df["notes"].str.lower().apply(lambda n: any(p in n for p in fc["possible_not_complaint"]))]
    no_rows = df[df["fits"] == "no"].sort_values("check_id")
    cols = ["check_id", "app", "topic_name", "notes", "text_clean"]
    return {
        "n": int(len(df)), "options": options,
        "sample": ("15 random complaint rows per app (seed 42), one row per distinct text, named topics only "
                   "(outliers excluded); judged by hand from text and topic name"),
        "raw": {"overall": fit_shares(df, options, fc["ci"]),
                "per_app": {a: fit_shares(d, options, fc["ci"]) for a, d in per_app.items()},
                "by_topic_group": {g: fit_shares(df[df["group"] == g], options, fc["ci"]) for g in ("broad", "specific")}},
        "reweighted_overall": {"weights_app_share_of_frame": {a: round(float(w[a]), 4) for a in apps},
                               "method": f"app-weighted mean; {fc['bootstrap_n']} stratified bootstrap draws", **reweighted},
        "broad_topics": fc["broad_topics"],
        "per_topic": df.groupby("topic_name")["fits"].value_counts().unstack(fill_value=0).reindex(columns=options, fill_value=0)
        .assign(n=lambda t: t.sum(axis=1)).sort_values("n", ascending=False).to_dict("index"),
        "not_complaint": {"tag": fc["not_complaint_tag"], "count": int(len(tagged)), "rows": tagged[cols].to_dict("records"),
                          "possible_untagged": possible[cols + ["fits"]].to_dict("records"),
                          "note": "possible_untagged: notes that describe a non-complaint without the exact tag; "
                                  "listed for review, not counted"},
        "no_rows": no_rows[cols].to_dict("records"),
        "assigned_by_transform_in_sample": int(df["assigned_by_transform"].sum()),
        "caveat": ("The sample is almost entirely fitted documents (unique, non-short texts): only "
                   f"{int(df['assigned_by_transform'].sum())} of {len(df)} rows were assigned by transform. "
                   "The fit rate does not describe short or duplicate reviews, which are 10% of complaint rows "
                   "(6,708 of 66,722) and had more outliers after transform (15% vs 9%)."),
    }


def print_fit_rate(r: dict[str, Any]) -> None:
    """Console summary of the fit check."""
    def line(label: str, s: dict[str, Any]) -> str:
        return f"   {label:<12} n={s['n']:>2}  " + "  ".join(
            f"{o}: {s[o]['share']:5.1%} [{s[o]['ci95'][0]:.0%}, {s[o]['ci95'][1]:.0%}]" for o in r["options"])

    print("\n== topic fit check (raw shares, 95% Wilson CI) ==")
    print(line("overall", r["raw"]["overall"]))
    for a, s in r["raw"]["per_app"].items():
        print(line(a, s))
    for g, s in r["raw"]["by_topic_group"].items():
        print(line(f"{g} topics", s))
    rw = r["reweighted_overall"]
    print("   reweighted to app shares " + str(rw["weights_app_share_of_frame"]) + ": " + "  ".join(
        f"{o}: {rw[o]['share']:.1%} [{rw[o]['ci95'][0]:.0%}, {rw[o]['ci95'][1]:.0%}]" for o in r["options"]))
    nc = r["not_complaint"]
    print(f"\nnotes tagged {nc['tag']!r}: {nc['count']}; possible untagged: "
          + "; ".join(f"{x['check_id']} ({x['notes']})" for x in nc["possible_untagged"]))
    print(f"\n== 'no' rows ({len(r['no_rows'])}) ==")
    for x in r["no_rows"]:
        print(f"   {x['check_id']} {x['app']:<9} [{x['topic_name']}] {x['notes']}\n        {x['text_clean'][:150]!r}")
    print("\ncaveat:", r["caveat"])


# ---------------------------------------------------------------- main

def main() -> None:  # noqa: PLR0915 - linear pipeline
    """CLI entry point."""
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path("configs/topics_final.yaml"))
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="run everything but write nothing (no W&B)")
    parser.add_argument("--fit-rate", action="store_true", help="score the filled topic_fit_check.xlsx and stop")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    for noisy in ("httpx", "huggingface_hub", "BERTopic", "numba"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    from bertopic import BERTopic

    cfg = load_config(args.config)
    tcfg = load_config(Path(cfg["topics_config"]))
    apps, text = tcfg["apps"], tcfg["text_col"]
    np.random.seed(cfg["seed"])
    if args.fit_rate:
        r = fit_rate(cfg, tcfg)
        rp = Path(cfg["report_path"])
        report = json.loads(rp.read_text(encoding="utf-8"))
        report["fit_check_results"] = r
        rp.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        print_fit_rate(r)
        logger.info("added fit_check_results to %s", rp)
        return

    model = BERTopic.load(cfg["model_in"], embedding_model=tcfg["embedding"]["model"])
    names = read_names(cfg)
    errors, merge, kept = validate(names, set(int(t) for t in set(model.topics_)))
    groups: dict[int, list[int]] = {}
    for s, d in merge.items():
        groups.setdefault(d, []).append(s)
    print(f"\n== validation: {len(names) - 1} topics, {len(merge)} merged into {len(groups)} targets, "
          f"{len(kept)} named topics after merging ==")
    for d, ss in sorted(groups.items()):
        print(f"   {d:>3} {kept.get(d, '?')!r:<40} <- {sorted(ss)}")
    if errors:
        print("\nVALIDATION FAILED")
        for e in errors:
            print(" -", e)
        sys.exit(1)
    print("validation passed")
    if args.validate_only:
        return

    # ---- fit docs in 6a order; the model's topics must match the saved assignments
    assign = pd.read_parquet(cfg["assignments_path"])
    if "topic_6a" in assign.columns:  # re-run: start again from the 6a ids
        assign = assign.assign(topic=assign["topic_6a"]).drop(columns=["topic_6a", "topic_name"])
    comp = tp.load_complaints(tcfg)
    fit_df = comp[comp["in_fit"]].reset_index(drop=True)
    docs = fit_df[text].tolist()
    a_fit = assign.set_index("reviewId").loc[fit_df["reviewId"], "topic"].to_numpy()
    if not np.array_equal(a_fit, np.asarray(model.topics_)):
        raise RuntimeError("6a model topics do not match topic_assignments.parquet (fit docs)")

    keywords_6a = {t: tp.keywords(model, t, tcfg["top_n_words"]) for t in sorted(set(model.topics_)) if t != -1}
    mapping = apply_merges(model, docs, merge)
    label_of = {mapping[t]: n for t, n in kept.items()} | {-1: cfg["outlier_label"]}
    stop = refresh_keywords(model, docs, tcfg, cfg)
    model.set_topic_labels(label_of)

    # ---- assignments: 6a ids -> final ids (merges only)
    final = assign.copy()
    final["topic_6a"] = final["topic"]
    final["topic"] = final["topic_6a"].map(mapping)
    if final["topic"].isna().any():
        raise RuntimeError("some 6a topics have no final id")
    final["topic"] = final["topic"].astype(int)
    final["topic_name"] = final["topic"].map(label_of)
    fit_final = final.set_index("reviewId").loc[fit_df["reviewId"], "topic"].to_numpy()
    expected = final["topic_6a"].map(lambda t: merge.get(t, t))
    pairs = pd.DataFrame({"target": expected, "final": final["topic"]}).drop_duplicates()
    partition_ok = not pairs["target"].duplicated().any() and not pairs["final"].duplicated().any()  # bijection
    check = {
        "rows_before": int(len(assign)), "rows_after": int(len(final)),
        "model_topics_match_fit_rows": bool(np.array_equal(fit_final, np.asarray(model.topics_))),
        "partition_equals_requested_merges": partition_ok,
        "outlier_rows_unchanged": bool(((assign["topic"] == -1) == (final["topic"] == -1)).all()),
        "rows_moved_by_merges": int(final["topic_6a"].isin(list(merge)).sum()),
        "rows_in_kept_topics": int((~final["topic_6a"].isin(list(merge)) & (final["topic_6a"] != -1)).sum()),
    }
    if not all(v for k, v in check.items() if isinstance(v, bool)) or check["rows_before"] != check["rows_after"]:
        raise RuntimeError(f"assignment check failed: {check}")
    cols = ["reviewId", "app", "topic", "topic_name", "topic_6a", "topic_before_outlier_reduction",
            "assigned_by_transform", "in_fit", "in_model_pool", "is_short"]
    final = final[cols]

    # ---- tables
    counts = share_frame(final, apps)
    kw = {t: tp.keywords(model, t, tcfg["top_n_words"]) for t in sorted(set(model.topics_)) if t != -1}
    merged_from = {mapping[t]: sorted([t] + groups.get(t, [])) for t in kept}
    table = []
    for t in sorted(counts["unique_texts"].index):
        u, a = counts["unique_texts"], counts["all_rows"]
        table.append({
            "topic": int(t), "topic_name": label_of[t], "merged_from_6a": merged_from.get(t, [-1]),
            "keywords": kw.get(t, []), "fit_docs": int((np.asarray(model.topics_) == t).sum()),
            "unique_texts": int(u.loc[t, "all"]), "all_rows": int(a.loc[t, "all"]),
            "share_unique_texts": {c: round(float(u.loc[t, c] / u[c].sum()), 4) for c in apps + ["all"]},
            "share_all_rows": {c: round(float(a.loc[t, c] / a[c].sum()), 4) for c in apps + ["all"]},
        })
    ocfg = cfg["overrepresented"]
    modes = [ocfg["primary"], *ocfg["also_report"]]
    over = {m: {b: overrepresented(counts[b], apps, ocfg["top_n"], ocfg["min_rows_in_app"], m)
                for b in ("unique_texts", "all_rows")} for m in modes}
    for by_mode in over.values():
        for by_basis in by_mode.values():
            for lst in by_basis.values():
                for e in lst:
                    e["topic_name"] = label_of[e["topic"]]
    over_main = over[ocfg["primary"]][ocfg["basis"]]
    danacicil = {t: ("danacicil" in v) for t, v in kw.items() if "danacicil" in v}

    print(f"\n== final topics: {len(kept)} named + outliers; check: {check}")
    print(f"{'id':>3} {'topic':<40}{'uniq':>7}  " + "  ".join(f"{a[:5]:>6}" for a in apps + ['all']) + "  keywords")
    for r in table:
        s = r["share_unique_texts"]
        print(f"{r['topic']:>3} {r['topic_name'][:39]:<40}{r['unique_texts']:>7}  "
              + "  ".join(f"{s[c]:>6.1%}" for c in apps + ["all"]) + "  " + ", ".join(r["keywords"][:6]))
    for m in modes:
        print(f"\n== most over-represented per app ({m}, {ocfg['basis']}; ratio = app share / reference share) ==")
        for a, lst in over[m][ocfg["basis"]].items():
            print(f"   {a:<10} " + " | ".join(f"{e['topic_name']} x{e['ratio']} ({e['share_in_app']:.1%} vs "
                                             f"{e['share_reference']:.1%})" for e in lst))
    if args.dry_run:
        logger.info("dry run: nothing written")
        return

    # ---- outputs
    tcfg_paths = Path(cfg["figures_dir"])
    tcfg_paths.mkdir(parents=True, exist_ok=True)
    texts = pd.read_parquet(tcfg["input_path"], columns=["reviewId", text]).set_index("reviewId")[text]
    key = export_fit_check(final, texts, label_of, cfg, apps)
    final.to_parquet(cfg["assignments_path"], index=False)
    model.save(cfg["model_out"], serialization="pickle", save_embedding_model=False)

    u = counts["unique_texts"]
    ov = pd.DataFrame({"topic_id": u.index, "size": u["all"].to_numpy(),
                       **{f"share_{a}": (u[a] / u[a].sum()).to_numpy() for a in apps},
                       "keywords": [", ".join(kw.get(t, [])) for t in u.index],
                       "label": [f"{label_of[t]}" for t in u.index]})
    n_unique = int(u["all"].sum())
    tp.plot_overview(ov, tcfg_paths / cfg["figures"]["overview"], tcfg,
                     f"{n_unique:,} unique complaint texts (incl. short, assigned by transform); {len(kept)} named topics "
                     f"after merging 45; outliers {u.loc[-1, 'all'] / n_unique:.0%} not shown.",
                     title="Phase 6b: complaint topics across apps (final names)", label_col="label",
                     size_label="Unique complaint texts")
    plot_overrepresented(over_main, apps, tcfg_paths / cfg["figures"]["overrepresented"],
                         f"Top {ocfg['top_n']} topics per app by share in the app / share in the other three apps combined; "
                         f"unique complaint texts; topics with < {ocfg['min_rows_in_app']} rows in the app ignored.")
    plot_share_per_app(u, label_of, apps, cfg["share_figure"]["top_n"], tcfg_paths / cfg["figures"]["share_per_app"],
                       f"Unique complaint texts (duplicates excluded), PRIMARY predictions; top {cfg['share_figure']['top_n']} "
                       "topics overall, the rest and outliers grouped as Other.")

    report = {
        "seed": cfg["seed"], "labels_used": "none (manual topic names; PRIMARY predictions; no gold or dev)",
        "workbook": cfg["workbook"], "validation": {"errors": [], "merged": {int(d): sorted(ss) for d, ss in groups.items()},
                                                    "n_topics_6a": len(names) - 1, "n_topics_final": len(kept)},
        "id_mapping_6a_to_final": {int(k): int(v) for k, v in mapping.items()},
        "assignment_check": check,
        "keyword_refresh": {"extra_stopwords": cfg["extra_stopwords"], "n_stopwords": len(stop),
                            "keep_words": cfg["keep_words"], "topics_with_danacicil_in_keywords": list(danacicil),
                            "assignments_unchanged": True},
        "keywords_6a": {int(k): v for k, v in keywords_6a.items()},
        "bases": {"unique_texts": "in_model_pool complaint rows (quote these; say 'unique review texts')",
                  "all_rows": "all complaint rows incl. duplicates (Phase 7 trends)",
                  "denominator": "all complaint rows of the app on that basis, outliers included"},
        "complaints_per_app": {b: {c: int(counts[b][c].sum()) for c in apps + ["all"]} for b in counts},
        "topics": table,
        "overrepresented": {
            "primary": ocfg["primary"], "basis_for_summary": ocfg["basis"],
            "methods": {"leave_one_out": "app share / the topic's share of the other three apps' complaints combined",
                        "pooled": "app share / the topic's share of all apps' complaints (DANA is 64% of complaints, "
                                  "so pooled hides what is specific to DANA)"},
            "min_rows_in_app": ocfg["min_rows_in_app"], "denominator": "all complaint rows of the app, outliers included",
            **over},
        "fit_check": {"xlsx": cfg["fit_check"]["xlsx_path"], "key": cfg["fit_check"]["key_path"], "n": int(len(key)),
                      "per_app": key["app"].value_counts().to_dict(),
                      "assigned_by_transform": int(key["assigned_by_transform"].sum()),
                      "topics_covered": int(key["topic"].nunique())},
        "outputs": {"model": cfg["model_out"], "assignments": cfg["assignments_path"],
                    "figures": {k: str(tcfg_paths / v) for k, v in cfg["figures"].items()}},
    }
    rp = Path(cfg["report_path"])
    rp.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    report["wandb_url"] = log_wandb(report, cfg, tcfg_paths)
    rp.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("wrote %s, %s, %s, %s, figures", cfg["assignments_path"], cfg["model_out"], cfg["fit_check"]["xlsx_path"], rp)


def log_wandb(report: dict[str, Any], cfg: dict[str, Any], fig_dir: Path) -> str | None:
    """One W&B run with the final topic table and both figures."""
    import wandb

    from src.train_indobert import wandb_logged_in

    if not wandb_logged_in():
        logger.warning("W&B not logged in; skipping")
        return None
    icfg = load_config(Path("configs/indobert.yaml"))
    try:
        run = wandb.init(project=icfg["wandb"]["project"], entity=icfg["wandb"]["entity"], name=cfg["wandb"]["name"],
                         job_type=cfg["wandb"]["job_type"], tags=["phase6", "bertopic", "final"],
                         config={"seed": cfg["seed"], "extra_stopwords": cfg["extra_stopwords"],
                                 "merged": report["validation"]["merged"]})
        rows = [{"topic": r["topic"], "topic_name": r["topic_name"], "unique_texts": r["unique_texts"],
                 "all_rows": r["all_rows"], **{f"share_{k}": v for k, v in r["share_unique_texts"].items()},
                 "keywords": ", ".join(r["keywords"])} for r in report["topics"]]
        run.log({"final_topics": wandb.Table(dataframe=pd.DataFrame(rows)),
                 **{k: wandb.Image(str(fig_dir / v)) for k, v in cfg["figures"].items()}})
        run.summary.update({"n_topics_final": report["validation"]["n_topics_final"],
                            "rows_moved_by_merges": report["assignment_check"]["rows_moved_by_merges"]})
        url = run.url
        run.finish()
        return url
    except Exception as exc:  # noqa: BLE001 - outputs are saved; W&B is a mirror
        logger.warning("W&B logging failed: %s", exc)
        return f"failed: {exc}"


if __name__ == "__main__":
    main()
