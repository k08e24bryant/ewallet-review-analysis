"""Scrape Google Play reviews for the e-wallet apps defined in configs/scrape.yaml.

Reviews are fetched newest-first. Each app stops at whichever comes first:
a review older than ``since`` (UTC), the ``max_per_app`` cap, or the end of
the review stream. The stop reason is recorded in ``{raw_dir}/{app_key}.meta.json``.

Output is one parquet per app at ``{raw_dir}/{app_key}.parquet``. Progress is
checkpointed after every page (one parquet part per page + a state file with
the continuation token) under ``{raw_dir}/{checkpoint_subdir}/{app_key}/`` so an
interrupted run resumes where it stopped.

Usage:
    uv run python -m src.scrape --app all --since 2026-07-01 --max-per-app 120000
    uv run python -m src.scrape --app dana --max-per-app 100 --out-dir data/raw/_test
    uv run python -m src.scrape --app all --verify-only
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from google_play_scraper import Sort
from google_play_scraper import app as fetch_app_details
from google_play_scraper.constants.element import ElementSpecs
from google_play_scraper.constants.request import Formats

# The public reviews() helper swallows every exception and returns token=None,
# which is indistinguishable from "no more reviews". We call the page fetcher
# directly so errors reach our retry/backoff logic and the token is a plain string.
from google_play_scraper.features.reviews import _fetch_review_items

logger = logging.getLogger("scrape")

PII_COLUMNS = ("userName", "userImage")
STOP_SINCE = "since_reached"
STOP_CAP = "cap_reached"
STOP_EXHAUSTED = "reviews_exhausted"


def load_config(path: Path) -> dict[str, Any]:
    """Load the scrape YAML config."""
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_since(value: str) -> datetime:
    """Parse an ISO date/datetime as UTC (naive input is interpreted as UTC)."""
    dt = datetime.fromisoformat(value)
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)


def verify_apps(apps: dict[str, str], lang: str, country: str) -> None:
    """Print title, developer, and installs for each app ID so they can be confirmed."""
    print(f"{'key':<10} {'app_id':<20} {'title':<35} {'developer':<30} {'installs':>14} {'realInstalls':>14}")
    for key, app_id in apps.items():
        try:
            d = fetch_app_details(app_id, lang=lang, country=country)
            print(
                f"{key:<10} {app_id:<20} {str(d.get('title'))[:35]:<35} "
                f"{str(d.get('developer'))[:30]:<30} {str(d.get('installs')):>14} "
                f"{d.get('realInstalls') or 0:>14,}"
            )
        except Exception as exc:  # noqa: BLE001 - report and continue with other apps
            print(f"{key:<10} {app_id:<20} ERROR: {type(exc).__name__}: {exc}")


def parse_review(raw: list, keep_columns: list[str], null_versions: set[str]) -> dict[str, Any]:
    """Extract only the kept fields from a raw review; PII fields are never read."""
    row: dict[str, Any] = {}
    for col in keep_columns:
        if col in PII_COLUMNS:
            raise ValueError(f"{col} is PII and must not be kept")
        if col == "at":
            # Library converts to naive local time; store explicit UTC instead.
            ts = raw[5][0] if raw[5] else None
            row["at"] = datetime.fromtimestamp(ts, tz=timezone.utc) if ts else None
        elif col == "reviewCreatedVersion":
            # API sends JSON null when unknown; also guard against placeholder strings.
            v = ElementSpecs.Review[col].extract_content(raw)
            row[col] = None if v is None or str(v).strip() in null_versions else str(v).strip()
        else:
            row[col] = ElementSpecs.Review[col].extract_content(raw)
    return row


def fetch_page(
    app_id: str,
    cfg: dict[str, Any],
    token: str | None,
    page_size: int,
) -> tuple[list[list], str | None]:
    """Fetch one page of raw reviews, retrying with exponential backoff on errors."""
    url = Formats.Reviews.build(lang=cfg["lang"], country=cfg["country"])
    sort = Sort[cfg["sort"].upper()].value
    for attempt in range(cfg["max_retries"] + 1):
        try:
            items, next_token = _fetch_review_items(
                url, app_id, sort, page_size, None, None, token
            )
            if isinstance(next_token, list):  # library's end-of-stream quirk
                next_token = None
            return items, next_token
        except Exception as exc:  # noqa: BLE001 - network/parse errors are all retryable
            if attempt == cfg["max_retries"]:
                raise RuntimeError(
                    f"{app_id}: page fetch failed after {attempt + 1} attempts; "
                    "checkpoint kept, rerun to resume"
                ) from exc
            wait = min(cfg["backoff_base_seconds"] * 2**attempt, cfg["backoff_max_seconds"])
            logger.warning("%s: %s: %s — retry %d in %ss", app_id, type(exc).__name__, exc, attempt + 1, wait)
            time.sleep(wait)
    raise AssertionError("unreachable")


def _replace_with_retry(tmp: Path, path: Path, attempts: int = 10) -> None:
    # On Windows, os.replace fails with WinError 5 while another process (antivirus,
    # search indexer, a log reader) briefly holds the target open. Retry a few times.
    for attempt in range(attempts):
        try:
            tmp.replace(path)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(0.2 * (attempt + 1))


def _atomic_write_parquet(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False)
    _replace_with_retry(tmp, path)


def _atomic_write_json(obj: dict[str, Any], path: Path) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    _replace_with_retry(tmp, path)


def _rows_to_frame(rows: list[dict[str, Any]], columns: list[str]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=columns)
    df["reviewCreatedVersion"] = df["reviewCreatedVersion"].astype("string")
    return df


def scrape_app(
    key: str,
    app_id: str,
    cfg: dict[str, Any],
    since: datetime,
    max_rows: int,
    out_dir: Path,
    fresh: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Scrape one app newest-first until ``since``, ``max_rows``, or the end of the stream.

    Returns the reviews and a metadata dict including ``stop_reason``.
    """
    final_path = out_dir / f"{key}.parquet"
    meta_path = out_dir / f"{key}.meta.json"
    ckpt_dir = out_dir / cfg["checkpoint_subdir"] / key
    state_path = ckpt_dir / "state.json"
    columns = [*cfg["keep_columns"], "app"]
    null_versions = set(cfg["null_version_values"])

    run_params = {
        "app_id": app_id,
        "lang": cfg["lang"],
        "country": cfg["country"],
        "sort": cfg["sort"],
        "batch_size": cfg["batch_size"],
        "since": since.isoformat(),
        "max_per_app": max_rows,
    }

    if fresh:
        final_path.unlink(missing_ok=True)
        meta_path.unlink(missing_ok=True)
        shutil.rmtree(ckpt_dir, ignore_errors=True)

    if final_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta["params"] != run_params:
            raise ValueError(
                f"{key}: existing output was scraped with {meta['params']}, not {run_params}; "
                "use --fresh to redo"
            )
        logger.info("%s: %s already complete (%s) — skipping", key, final_path, meta["stop_reason"])
        return pd.read_parquet(final_path), meta

    # Resume: state lists the committed parts; any part beyond n_parts is an
    # orphan from a crash between writing the part and updating the state.
    token: str | None = None
    n_parts = 0
    n_rows = 0
    seen: set[str] = set()
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state["params"] != run_params:
            raise ValueError(
                f"{key}: checkpoint params {state['params']} differ from current {run_params}; "
                "use --fresh to discard the checkpoint"
            )
        token, n_parts, n_rows = state["token"], state["n_parts"], state["n_rows"]
        for i in range(1, n_parts + 1):
            seen.update(pd.read_parquet(ckpt_dir / f"part-{i:05d}.parquet", columns=["reviewId"])["reviewId"])
        logger.info("%s: resuming from checkpoint with %d rows in %d parts", key, n_rows, n_parts)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    stop_reason: str | None = None
    started = time.monotonic()
    while stop_reason is None:
        page_size = min(cfg["batch_size"], max_rows - n_rows)
        items, token = fetch_page(app_id, cfg, token, page_size)

        page_rows: list[dict[str, Any]] = []
        for raw in items:
            row = parse_review(raw, cfg["keep_columns"], null_versions)
            if row["at"] is not None and row["at"] < since:
                # Newest-first: everything after this is older too. Finish the
                # page (skipping old rows) in case ordering is not strict.
                stop_reason = STOP_SINCE
                continue
            if row["reviewId"] in seen:
                continue
            seen.add(row["reviewId"])
            row["app"] = key
            page_rows.append(row)

        page_rows = page_rows[: max_rows - n_rows]
        n_rows += len(page_rows)
        if stop_reason is None:
            if n_rows >= max_rows:
                stop_reason = STOP_CAP
            elif token is None or not items:
                stop_reason = STOP_EXHAUSTED

        n_parts += 1
        _atomic_write_parquet(_rows_to_frame(page_rows, columns), ckpt_dir / f"part-{n_parts:05d}.parquet")
        _atomic_write_json(
            {
                "params": run_params,
                "token": token,
                "n_parts": n_parts,
                "n_rows": n_rows,
                "oldest_at": page_rows[-1]["at"].isoformat() if page_rows else None,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
            state_path,
        )
        if n_parts % 10 == 0 or stop_reason:
            oldest = page_rows[-1]["at"].strftime("%Y-%m-%d %H:%M") if page_rows else "-"
            logger.info(
                "%s: page %d, %d rows, oldest %s, %.0fs elapsed",
                key, n_parts, n_rows, oldest, time.monotonic() - started,
            )
        if stop_reason is None:
            time.sleep(cfg["sleep_seconds"])

    parts = [pd.read_parquet(ckpt_dir / f"part-{i:05d}.parquet") for i in range(1, n_parts + 1)]
    df = pd.concat(parts, ignore_index=True) if parts else _rows_to_frame([], columns)
    meta = {
        "app_id": app_id,
        "params": run_params,
        "stop_reason": stop_reason,
        "rows": int(len(df)),
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_write_parquet(df, final_path)
    _atomic_write_json(meta, meta_path)
    shutil.rmtree(ckpt_dir)
    logger.info("%s: wrote %d rows to %s (stop: %s)", key, len(df), final_path, stop_reason)
    return df, meta


def weekly_counts(at: pd.Series) -> pd.Series:
    """Reviews per ISO week (Mon–Sun), including empty weeks, excluding the partial first and last week."""
    if at.empty:
        return pd.Series(dtype="int64")
    weeks = at.dt.tz_convert("UTC").dt.tz_localize(None).dt.to_period("W-SUN")
    counts = weeks.value_counts().sort_index()
    full_range = pd.period_range(counts.index.min(), counts.index.max(), freq="W-SUN")
    counts = counts.reindex(full_range, fill_value=0)
    return counts.iloc[1:-1]


def summarize(frames: dict[str, pd.DataFrame], metas: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Per-app summary: rows, stop reason, date range, versions, null share, scores, weekly volume."""
    summary: dict[str, Any] = {}
    for key, df in frames.items():
        versions = df["reviewCreatedVersion"]
        n_placeholder = int(versions.isin(["", "0"]).sum())
        if n_placeholder:
            raise AssertionError(f"{key}: {n_placeholder} placeholder versions not normalized to null")
        weekly = weekly_counts(df["at"])
        summary[key] = {
            "rows": int(len(df)),
            "stop_reason": metas[key]["stop_reason"],
            "oldest": df["at"].min().isoformat() if len(df) else None,
            "newest": df["at"].max().isoformat() if len(df) else None,
            "distinct_versions": int(versions.nunique(dropna=True)),
            "null_version_share": round(float(versions.isna().mean()), 4) if len(df) else None,
            "score_distribution": {
                int(s): int(n) for s, n in df["score"].value_counts().sort_index().items()
            },
            "weekly_reviews": {
                "full_weeks": int(len(weekly)),
                "min": int(weekly.min()) if len(weekly) else None,
                "median": float(weekly.median()) if len(weekly) else None,
                "max": int(weekly.max()) if len(weekly) else None,
            },
        }
    return summary


def print_summary(summary: dict[str, Any]) -> None:
    """Print the per-app summary as a table."""
    print(
        f"\n{'app':<10} {'rows':>7} {'stop':<18} {'oldest':<17} {'newest':<17} "
        f"{'vers':>5} {'null_ver':>8} {'wk_min':>7} {'wk_med':>7} {'wk_max':>7} {'wks':>4}"
    )
    for key, s in summary.items():
        w = s["weekly_reviews"]
        print(
            f"{key:<10} {s['rows']:>7} {s['stop_reason']:<18} {(s['oldest'] or '')[:16]:<17} "
            f"{(s['newest'] or '')[:16]:<17} {s['distinct_versions']:>5} {s['null_version_share']:>8.2%} "
            f"{w['min'] if w['min'] is not None else '-':>7} {w['median'] if w['median'] is not None else '-':>7} "
            f"{w['max'] if w['max'] is not None else '-':>7} {w['full_weeks']:>4}"
        )
    print(f"\n{'app':<10} score distribution (count / share)")
    for key, s in summary.items():
        dist = s["score_distribution"]
        parts = [f"{i}* {dist.get(i, 0):>6} ({dist.get(i, 0) / max(s['rows'], 1):5.1%})" for i in range(1, 6)]
        print(f"{key:<10} " + "  ".join(parts))


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path("configs/scrape.yaml"))
    parser.add_argument("--app", default="all", help="app key from config, or 'all'")
    parser.add_argument("--since", default=None, help="UTC date/datetime cutoff (overrides config)")
    parser.add_argument("--max-per-app", type=int, default=None, help="row cap per app (overrides config)")
    parser.add_argument("--out-dir", type=Path, default=None, help="override raw_dir (e.g. for test runs)")
    parser.add_argument("--fresh", action="store_true", help="discard existing output and checkpoint")
    parser.add_argument("--verify-only", action="store_true", help="only print app details and exit")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    cfg = load_config(args.config)
    if args.app == "all":
        apps = cfg["apps"]
    elif args.app in cfg["apps"]:
        apps = {args.app: cfg["apps"][args.app]}
    else:
        parser.error(f"unknown app {args.app!r}; choose from {list(cfg['apps'])} or 'all'")

    since = parse_since(args.since or str(cfg["since"]))
    max_rows = args.max_per_app or cfg["max_per_app"]
    out_dir = args.out_dir or Path(cfg["raw_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        "seed=%s since=%s max_per_app=%d out_dir=%s apps=%s",
        cfg["seed"], since.isoformat(), max_rows, out_dir, list(apps),
    )

    verify_apps(apps, cfg["lang"], cfg["country"])
    if args.verify_only:
        return

    frames: dict[str, pd.DataFrame] = {}
    metas: dict[str, dict[str, Any]] = {}
    for key, app_id in apps.items():
        frames[key], metas[key] = scrape_app(key, app_id, cfg, since, max_rows, out_dir, args.fresh)

    summary = summarize(frames, metas)
    print_summary(summary)
    summary_path = out_dir / "scrape_summary.json"
    _atomic_write_json(
        {"seed": cfg["seed"], "config": str(args.config), "since": since.isoformat(),
         "max_per_app": max_rows, "summary": summary},
        summary_path,
    )
    logger.info("summary written to %s", summary_path)


if __name__ == "__main__":
    main()
