"""Clean scraped reviews, add flag columns, and report EDA stats (Phase 2).

No rows are dropped: every raw review is kept and annotated with flags
(is_short, is_dup_text, is_partial_week) so later phases choose their own filters.

Usage:
    uv run python -m src.preprocess --config configs/preprocess.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger("preprocess")

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
WHITESPACE_RE = re.compile(r"\s+")
# Common emoji / pictograph blocks; used for stats only, emoji are never removed.
EMOJI_RE = re.compile(
    "[\U0001f000-\U0001faff☀-➿⬀-⯿⌀-⏿️‍]"
)


def load_config(path: Path) -> dict[str, Any]:
    """Load the preprocess YAML config."""
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_raw(raw_dir: Path, apps: list[str]) -> pd.DataFrame:
    """Concatenate the per-app raw parquet files."""
    return pd.concat([pd.read_parquet(raw_dir / f"{app}.parquet") for app in apps], ignore_index=True)


def make_cleaner(max_char_repeat: int):
    """Return a text cleaning function: strip URLs, cap repeated non-digit chars, collapse whitespace."""
    repeat_re = re.compile(r"(\D)\1{%d,}" % max_char_repeat, re.IGNORECASE)
    replacement = r"\1" * max_char_repeat

    def clean(text: str | None) -> str:
        if not text:
            return ""
        text = URL_RE.sub(" ", text)
        text = repeat_re.sub(replacement, text)
        return WHITESPACE_RE.sub(" ", text).strip()

    return clean


def add_features(df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Add text_clean and the flag columns; returns a new frame sorted by app, time."""
    clean = make_cleaner(cfg["cleaning"]["max_char_repeat"])
    out = df.sort_values(["app", "at", "reviewId"], kind="stable").reset_index(drop=True)

    out["text_clean"] = pd.Series([clean(t) for t in out["content"]], index=out.index, dtype="string")
    out["n_words"] = out["text_clean"].str.split().str.len().fillna(0).astype("int32")
    out["is_short"] = out["n_words"] < cfg["flags"]["min_words"]

    # Sorted oldest-first per app, so keep="first" leaves the earliest occurrence unflagged.
    dup_key = out["text_clean"].str.lower()
    out["is_dup_text"] = pd.DataFrame({"app": out["app"], "k": dup_key}).duplicated(keep="first")

    at_utc = out["at"].dt.tz_convert("UTC")
    out["week_start"] = at_utc.dt.normalize() - pd.to_timedelta(at_utc.dt.weekday, unit="D")
    partial = pd.to_datetime(cfg["flags"]["partial_weeks"]).tz_localize("UTC")
    out["is_partial_week"] = out["week_start"].isin(partial)

    out["version_major_minor"] = out["reviewCreatedVersion"].str.extract(r"^(\d+\.\d+)", expand=False)
    return out


def check_partial_weeks(df: pd.DataFrame, cfg: dict[str, Any]) -> list[str]:
    """Return warnings if configured partial weeks don't match the first/last week in the data."""
    weeks = df["week_start"]
    expected = {weeks.min().strftime("%Y-%m-%d"), weeks.max().strftime("%Y-%m-%d")}
    configured = set(cfg["flags"]["partial_weeks"])
    if expected != configured:
        return [f"partial_weeks {sorted(configured)} != first/last week in data {sorted(expected)}"]
    return []


def short_report(df: pd.DataFrame) -> pd.DataFrame:
    """Rows that is_short would remove, per app and star (count and share of that cell)."""
    g = df.groupby(["app", "score"])["is_short"].agg(total="size", short="sum")
    g["share"] = g["short"] / g["total"]
    return g


def dup_report(df: pd.DataFrame, top_n: int) -> dict[str, Any]:
    """Per app: duplicate-text share (all rows and non-short rows) and the most repeated texts."""
    report: dict[str, Any] = {}
    for app, g in df.groupby("app"):
        counts = g["text_clean"].str.lower().value_counts()
        long_ = g[~g["is_short"]]
        report[app] = {
            "rows": int(len(g)),
            "dup_rows": int(g["is_dup_text"].sum()),
            "dup_share": float(g["is_dup_text"].mean()),
            "dup_share_non_short": float(long_["is_dup_text"].mean()) if len(long_) else None,
            "top_texts": [{"text": t, "count": int(n)} for t, n in counts.head(top_n).items()],
        }
    return report


def token_report(df: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, Any]:
    """Token-count percentiles (incl. [CLS]/[SEP]), share above max_length, and [UNK] stats."""
    from transformers import AutoTokenizer

    eda = cfg["eda"]
    tok = AutoTokenizer.from_pretrained(eda["tokenizer"])
    enc = tok(df["text_clean"].tolist(), add_special_tokens=True, truncation=False)["input_ids"]
    lengths = np.fromiter((len(ids) for ids in enc), dtype=np.int32, count=len(enc))
    n_unk = np.fromiter((ids.count(tok.unk_token_id) for ids in enc), dtype=np.int32, count=len(enc))
    n_content = lengths - 2  # tokens other than [CLS]/[SEP]

    def stats(mask: np.ndarray) -> dict[str, Any]:
        lens = lengths[mask]
        return {
            "rows": int(mask.sum()),
            "percentiles": {f"p{p}": float(np.percentile(lens, p)) for p in eda["token_percentiles"]},
            "max": int(lens.max()),
            "above": {
                str(t): {"count": int((lens > t).sum()), "share": float((lens > t).mean())}
                for t in eda["length_thresholds"]
            },
        }

    apps = df["app"].to_numpy()
    non_short = ~df["is_short"].to_numpy()
    has_emoji = df["text_clean"].str.contains(EMOJI_RE.pattern, regex=True).to_numpy()
    return {
        "tokenizer": eda["tokenizer"],
        "all": stats(np.ones(len(df), dtype=bool)),
        "non_short": stats(non_short),
        "per_app": {app: stats(apps == app) for app in sorted(set(apps))},
        "unk": {
            "share_with_any_unk": float((n_unk > 0).mean()),
            "share_all_unk": float(((n_unk == n_content) & (n_content > 0)).mean()),
            "share_with_emoji": float(has_emoji.mean()),
            "share_emoji_rows_with_unk": float((n_unk[has_emoji] > 0).mean()) if has_emoji.any() else None,
        },
    }


NON_LATIN_RE = re.compile(
    r"[؀-ۿऀ-ॿ฀-๿぀-ヿ㐀-鿿가-힯Ѐ-ӿ]"
)
WORD_RE = re.compile(r"[a-z]+")


def classify_language(texts: list[str], n_words: list[int], cfg: dict[str, Any]) -> list[str]:
    """Label each text as indonesian / english / javanese? / sundanese? / non_latin / too_short.

    lingua decides between the configured candidates; Javanese/Sundanese markers
    and non-Latin script override it, since lingua cannot detect those.
    """
    from lingua import Language, LanguageDetectorBuilder

    lc = cfg["language"]
    detector = LanguageDetectorBuilder.from_languages(*[getattr(Language, c) for c in lc["candidates"]]).build()
    merged = set(lc["merge_into_indonesian"])
    markers = {name: set(words) for name, words in lc["regional_markers"].items()}

    labels: list[str] = []
    to_detect: list[int] = []
    for i, (text, n) in enumerate(zip(texts, n_words)):
        words = set(WORD_RE.findall(text.lower()))
        regional = [name for name, m in markers.items() if len(words & m) >= lc["min_markers"]]
        if NON_LATIN_RE.search(text):
            labels.append("non_latin")
        elif n < lc["min_words"]:
            labels.append("too_short")
        elif regional:
            labels.append(f"{regional[0]}?")
        else:
            labels.append("")
            to_detect.append(i)
    detected = detector.detect_languages_in_parallel_of([texts[i] for i in to_detect])
    for i, lang in zip(to_detect, detected):
        name = lang.name if lang else "UNDETERMINED"
        labels[i] = "indonesian" if name == "INDONESIAN" or name in merged else name.lower()
    return labels


def language_report(df: pd.DataFrame, cfg: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """Estimate the non-Indonesian share on a per-app stratified sample; returns stats and flagged rows."""
    lc = cfg["language"]
    sample = (
        df.groupby("app", group_keys=False)
        .sample(n=lc["sample_per_app"], random_state=cfg["seed"])
        .reset_index(drop=True)
    )
    sample["lang"] = classify_language(sample["text_clean"].tolist(), sample["n_words"].tolist(), cfg)

    def shares(s: pd.DataFrame) -> dict[str, Any]:
        classified = s[s["lang"] != "too_short"]
        return {
            "rows": int(len(s)),
            "counts": {k: int(v) for k, v in s["lang"].value_counts().items()},
            "too_short_share": float((s["lang"] == "too_short").mean()),
            "non_indonesian_share_of_classified": (
                float((classified["lang"] != "indonesian").mean()) if len(classified) else None
            ),
        }

    report = {
        "method": (
            f"lingua restricted to {lc['candidates']} (merged into indonesian: {lc['merge_into_indonesian']}); "
            f"texts under {lc['min_words']} words not classified; non-Latin script and "
            "Javanese/Sundanese marker-word heuristic override lingua"
        ),
        "sample": f"{lc['sample_per_app']} per app, random_state={cfg['seed']}",
        "overall": shares(sample),
        "per_app": {app: shares(g) for app, g in sample.groupby("app")},
    }
    flagged = sample[~sample["lang"].isin(["indonesian", "too_short"])]
    examples = [{"app": r.app, "lang": r.lang, "text": r.text_clean[:140]} for r in flagged.itertuples()]
    return report, examples


def version_summary(df: pd.DataFrame, top_n: int = 5) -> dict[str, Any]:
    """Per app: null share and the most-reviewed major.minor versions."""
    out: dict[str, Any] = {}
    for app, g in df.groupby("app"):
        vc = g["version_major_minor"].value_counts()
        out[app] = {
            "null_share": float(g["version_major_minor"].isna().mean()),
            "distinct_major_minor": int(vc.size),
            "top": [{"version": v, "rows": int(n), "share": float(n / len(g))} for v, n in vc.head(top_n).items()],
        }
    return out


def main() -> None:
    """CLI entry point."""
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path("configs/preprocess.yaml"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    cfg = load_config(args.config)
    logger.info("seed=%s config=%s", cfg["seed"], args.config)

    raw = load_raw(Path(cfg["raw_dir"]), cfg["apps"])
    df = add_features(raw, cfg)
    assert len(df) == len(raw), "no rows may be dropped"
    for w in check_partial_weeks(df, cfg):
        logger.warning(w)

    out_path = Path(cfg["output_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    logger.info("wrote %d rows, %d columns to %s", len(df), df.shape[1], out_path)

    changed = (df["text_clean"] != df["content"].str.strip()).mean()
    shorts = short_report(df)
    dups = dup_report(df, cfg["eda"]["top_n_dup_texts"])
    logger.info("tokenizing %d reviews", len(df))
    tokens = token_report(df, cfg)
    logger.info("detecting language on sample")
    lang, lang_examples = language_report(df, cfg)
    versions = version_summary(df)

    report = {
        "seed": cfg["seed"],
        "config": str(args.config),
        "rows": int(len(df)),
        "columns": list(df.columns),
        "text_clean_changed_share": float(changed),
        "flag_totals": {c: int(df[c].sum()) for c in ["is_short", "is_dup_text", "is_partial_week"]},
        "short_by_app_score": [
            {"app": a, "score": int(s), "total": int(r.total), "short": int(r.short), "share": float(r.share)}
            for (a, s), r in shorts.iterrows()
        ],
        "duplicates": dups,
        "tokens": tokens,
        "language": lang,
        "language_examples": lang_examples,
        "versions": versions,
    }
    report_path = Path(cfg["report_path"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("report written to %s", report_path)

    print_report(report, shorts)


def print_report(report: dict[str, Any], shorts: pd.DataFrame) -> None:
    """Print the EDA report in readable tables."""
    print(f"\nrows: {report['rows']}  text_clean differs from content: {report['text_clean_changed_share']:.2%}")
    print("flag totals:", report["flag_totals"])

    print("\n== is_short would remove (count / share of cell) ==")
    pivot = shorts.apply(lambda r: f"{int(r.short)}/{int(r.total)} ({r.share:.0%})", axis=1).unstack("score")
    print(pivot.to_string())
    per_app = shorts.groupby("app")[["short", "total"]].sum()
    per_app["share"] = per_app["short"] / per_app["total"]
    print(per_app.to_string())

    print("\n== duplicate text ==")
    for app, d in report["duplicates"].items():
        print(f"{app}: dup share {d['dup_share']:.2%} ({d['dup_rows']} rows), non-short dup share {d['dup_share_non_short']:.2%}")
        print("   " + " | ".join(f"{t['text']!r} x{t['count']}" for t in d["top_texts"]))

    t = report["tokens"]
    print(f"\n== token length ({t['tokenizer']}, incl. [CLS]/[SEP]) ==")
    for name, s in [("all", t["all"]), ("non_short", t["non_short"]), *t["per_app"].items()]:
        p = s["percentiles"]
        above = "  ".join(f">{k}: {a['share']:.2%} ({a['count']})" for k, a in s["above"].items())
        print(
            f"{name:<10} n={s['rows']:>6}  " + "  ".join(f"{k}={v:.0f}" for k, v in p.items())
            + f"  max={s['max']}  {above}"
        )
    u = t["unk"]
    print(
        f"[UNK]: any={u['share_with_any_unk']:.2%}  all-UNK={u['share_all_unk']:.2%}  "
        f"has emoji={u['share_with_emoji']:.2%}  emoji rows with UNK={u['share_emoji_rows_with_unk']:.2%}"
    )

    lang = report["language"]
    print(f"\n== language ({lang['sample']}) ==\n{lang['method']}")
    for name, s in [("overall", lang["overall"]), *lang["per_app"].items()]:
        counts = "  ".join(f"{k}={v}" for k, v in s["counts"].items())
        print(
            f"{name:<10} n={s['rows']:>5}  too_short={s['too_short_share']:.1%}  "
            f"non-id of classified={s['non_indonesian_share_of_classified']:.1%}  [{counts}]"
        )
    print(f"all {len(report['language_examples'])} sample rows not labeled indonesian/too_short:")
    for e in report["language_examples"]:
        print(f"  [{e['app']}/{e['lang']}] {e['text']}")

    print("\n== versions (major.minor) ==")
    for app, v in report["versions"].items():
        top = ", ".join(f"{x['version']} ({x['share']:.0%})" for x in v["top"])
        print(f"{app}: null {v['null_share']:.1%}, distinct {v['distinct_major_minor']}, top: {top}")


if __name__ == "__main__":
    main()
