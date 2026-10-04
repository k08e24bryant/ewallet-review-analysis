"""Phase 5a: build the dev set used for model and label-scheme selection (never gold).

Per app x weak_label stratum (target ``per_stratum`` rows):
    1. reuse reserve / reserve_extra rows that are labeled (not invalid) and were
       not used in the final gold set; they were never used for evaluation
    2. top up from in_model_pool rows outside train, val, gold, reserve, reserve_extra

Only the new top-up rows are exported for labeling (data/dev/dev_to_label.xlsx).
The hidden key (data/dev/dev_key.parquet) lists every dev row with its source.
Strata that cannot reach the target are reported, not filled from train/val.

Usage:
    uv run python -m src.dev_set --config configs/dev.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.gold import read_sheet
from src.label import write_label_sheet

logger = logging.getLogger("dev_set")


def load_config(path: Path) -> dict[str, Any]:
    """Load a YAML config."""
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def reusable_reserve(gcfg: dict[str, Any], labels: list[str]) -> pd.DataFrame:
    """Labeled, valid reserve/reserve_extra rows that are not part of the final gold set."""
    key = pd.read_parquet(gcfg["key_path"])
    gold_final = set(pd.read_parquet(gcfg["output_path"])["reviewId"])
    sheets = [read_sheet(Path(gcfg["workbook"]), "reserve")]
    ex = gcfg["reserve_extra"]
    if Path(ex["xlsx_path"]).exists():
        sheets.append(read_sheet(Path(ex["xlsx_path"]), ex["sheet"]))
    labeled = pd.concat(sheets, ignore_index=True)[["gold_id", "label"]]
    r = key[key["set"].isin(["reserve", "reserve_extra"])].merge(labeled, on="gold_id")
    r = r[~r["reviewId"].isin(gold_final) & r["label"].isin(labels)]
    return r.rename(columns={"gold_id": "dev_id", "set": "source"})


def build_dev(cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Return (dev key with all rows, new rows to label, per-stratum summary)."""
    gcfg = load_config(Path(cfg["gold_config"]))
    seed, target = cfg["seed"], cfg["per_stratum"]
    pool = pd.read_parquet(cfg["pool_path"])
    splits = Path(cfg["splits_dir"])
    train_ids = set(pd.read_parquet(splits / "train.parquet")["reviewId"])
    val_ids = set(pd.read_parquet(splits / "val.parquet")["reviewId"])
    gold_key = pd.read_parquet(gcfg["key_path"])
    excluded = train_ids | val_ids | set(gold_key["reviewId"])
    spare = pool[~pool["reviewId"].isin(excluded)]

    reuse = reusable_reserve(gcfg, cfg["labels"])
    reused_parts, new_parts, summary = [], [], {}
    for (app, wl), stratum in pool.groupby(["app", "weak_label"]):
        r = reuse[(reuse["app"] == app) & (reuse["weak_label"] == wl)]
        if len(r) > target:
            r = r.sample(n=target, random_state=seed)
        need = target - len(r)
        cand = spare[(spare["app"] == app) & (spare["weak_label"] == wl)]
        top = cand.sample(n=min(need, len(cand)), random_state=seed)
        reused_parts.append(r)
        new_parts.append(top)
        summary[f"{app}|{wl}"] = {
            "reused": int(len(r)), "new": int(len(top)), "total": int(len(r) + len(top)),
            "short": int(target - len(r) - len(top)), "spare_pool": int(len(cand)),
        }

    new = pd.concat(new_parts, ignore_index=True)
    rng = np.random.default_rng(seed)
    new = new.iloc[rng.permutation(len(new))].reset_index(drop=True)
    new["dev_id"] = [f"{cfg['id_prefix']}{i:0{cfg['id_width']}d}" for i in range(1, len(new) + 1)]
    new["source"] = "new"
    new["label"] = None

    cols = ["dev_id", "source", "reviewId", "app", "score", "weak_label", "label"]
    key = pd.concat([pd.concat(reused_parts, ignore_index=True)[cols], new[cols]], ignore_index=True)

    ids = set(key["reviewId"])
    overlap = {"train": len(ids & train_ids), "val": len(ids & val_ids),
               "gold": len(ids & set(pd.read_parquet(gcfg["output_path"])["reviewId"]))}
    assert overlap == {"train": 0, "val": 0, "gold": 0}, f"dev overlaps: {overlap}"
    assert key["reviewId"].is_unique, "duplicate reviewIds in dev set"
    new_ids = set(new["reviewId"])
    assert not new_ids & set(gold_key["reviewId"]), "new dev rows overlap reserve/reserve_extra"
    return key, new, {"strata": summary, "overlap": overlap}  # new rows come from the pool, text_clean included


def export(key: pd.DataFrame, new: pd.DataFrame, cfg: dict[str, Any]) -> None:
    """Write the labeling workbook (new rows only) and the hidden key; refuse to overwrite either."""
    from openpyxl import Workbook

    xlsx, key_path = Path(cfg["xlsx_path"]), Path(cfg["key_path"])
    for p in (xlsx, key_path):
        if p.exists():
            raise FileExistsError(f"{p} exists; refusing to overwrite (it may contain labels)")
    xlsx.parent.mkdir(parents=True, exist_ok=True)
    label_cfg = load_config(Path(cfg["label_config"]))
    wb = Workbook()
    ws = wb.active
    ws.title = cfg["sheet"]
    rows = new.sort_values("dev_id")[["dev_id", "text_clean"]].rename(columns={"dev_id": "gold_id"})
    write_label_sheet(ws, rows, label_cfg["gold"])
    wb.save(xlsx)
    key.to_parquet(key_path, index=False)


def main() -> None:
    """CLI entry point."""
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path("configs/dev.yaml"))
    parser.add_argument("--dry-run", action="store_true", help="report counts without writing files")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    cfg = load_config(args.config)
    logger.info("seed=%s config=%s", cfg["seed"], args.config)
    key, new, info = build_dev(cfg)

    s = pd.DataFrame(info["strata"]).T
    print("\n== dev set per stratum (target %d) ==" % cfg["per_stratum"])
    print(s.to_string())
    print(f"total: {len(key)} rows ({(key['source'] != 'new').sum()} reused, {len(new)} new to label), "
          f"short {int(s['short'].sum())}")
    print("reused by source:", key.loc[key["source"] != "new", "source"].value_counts().to_dict())
    print("overlap checks (must be 0):", info["overlap"])
    if args.dry_run:
        return
    export(key, new, cfg)
    logger.info("wrote %s (%d new rows) and %s (%d rows)", cfg["xlsx_path"], len(new), cfg["key_path"], len(key))


if __name__ == "__main__":
    main()
