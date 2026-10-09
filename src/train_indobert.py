"""Phase 5b: fine-tune indobenchmark/indobert-base-p1 on weak labels.

Training and early stopping use train/val only. Gold and dev are never read
here; they are evaluated in separate steps.

Usage:
    uv run python -m src.train_indobert --config configs/indobert.yaml
    uv run python -m src.train_indobert --config configs/indobert.yaml \
        --label-scheme binary_drop_3star --class-weights sqrt-balanced
    uv run python -m src.train_indobert --config configs/indobert.yaml --smoke
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import platform
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import accuracy_score, f1_score

from src.schemes import SCHEMES, apply_scheme

logger = logging.getLogger("indobert")

WEIGHTS = ("none", "balanced", "sqrt-balanced")


def load_config(path: Path) -> dict[str, Any]:
    """Load the IndoBERT YAML config."""
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def class_weight_vector(y: np.ndarray, n_classes: int, mode: str) -> np.ndarray | None:
    """None, balanced n/(k*n_c), or the square root of balanced."""
    if mode == "none":
        return None
    counts = np.bincount(y, minlength=n_classes).astype(float)
    w = len(y) / (n_classes * counts)
    if mode == "sqrt-balanced":
        return np.sqrt(w)
    if mode == "balanced":
        return w
    raise ValueError(f"unknown class_weights {mode!r}; choose from {WEIGHTS}")


def make_loss(weights: np.ndarray | None):
    """Cross-entropy for Trainer(compute_loss_func=...).

    With compute_loss_func set, Trainer does not divide by gradient-accumulation
    steps, so the loss is summed and divided by num_items_in_batch (all items in
    the accumulated batch). Without it (evaluation), it falls back to the mean.
    """
    w = None if weights is None else torch.tensor(weights, dtype=torch.float32)

    def loss_fn(outputs, labels, num_items_in_batch=None):
        logits = outputs.logits.float()
        wt = None if w is None else w.to(logits.device)
        loss = torch.nn.functional.cross_entropy(logits, labels, weight=wt, reduction="sum")
        denom = num_items_in_batch if num_items_in_batch is not None else labels.numel()
        return loss / denom

    return loss_fn


def wandb_logged_in() -> bool:
    """True if W&B credentials are available (env var or netrc)."""
    try:
        import wandb

        return bool(wandb.Api().api_key)
    except Exception:  # noqa: BLE001 - any failure means "not available"
        return False


def main() -> None:  # noqa: PLR0915 - linear training script
    """CLI entry point."""
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path("configs/indobert.yaml"))
    parser.add_argument("--label-scheme", choices=SCHEMES, default=None)
    parser.add_argument("--class-weights", choices=WEIGHTS, default=None)
    parser.add_argument("--smoke", action="store_true", help="200 train / 200 val rows, 1 epoch")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    from datasets import Dataset
    from transformers import (
        AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, EarlyStoppingCallback,
        Trainer, TrainerCallback, TrainingArguments, set_seed,
    )

    cfg = load_config(args.config)
    scheme = args.label_scheme or cfg["label_scheme"]
    weights_mode = args.class_weights or cfg["class_weights"]
    tr = dict(cfg["training"])
    seed = cfg["seed"]
    out_root = Path(cfg["output_root"])
    if args.smoke:
        sm = cfg["smoke"]
        tr["num_train_epochs"] = sm["num_train_epochs"]
        tr["eval_steps"] = sm["eval_steps"]
        out_root = Path(sm["output_root"])
    out_dir = out_root / scheme / weights_mode
    set_seed(seed)

    splits = Path(cfg["splits_dir"])
    train_df, labels = apply_scheme(pd.read_parquet(splits / "train.parquet"), scheme)
    val_df, _ = apply_scheme(pd.read_parquet(splits / "val.parquet"), scheme)
    if args.smoke:
        train_df = train_df.sample(n=cfg["smoke"]["train_rows"], random_state=seed)
        val_df = val_df.sample(n=cfg["smoke"]["val_rows"], random_state=seed)
    label2id = {l: i for i, l in enumerate(labels)}
    y_train = train_df["target"].map(label2id).to_numpy()
    cw = class_weight_vector(y_train, len(labels), weights_mode)
    logger.info("seed=%d scheme=%s weights=%s train=%d val=%d label counts=%s class_weights=%s",
                seed, scheme, weights_mode, len(train_df), len(val_df),
                train_df["target"].value_counts().to_dict(), None if cw is None else np.round(cw, 3).tolist())

    tok = AutoTokenizer.from_pretrained(cfg["model_name"])
    text = cfg["text_col"]

    def to_ds(df: pd.DataFrame) -> Dataset:
        ds = Dataset.from_dict({"text": df[text].tolist(), "label": df["target"].map(label2id).tolist()})
        return ds.map(
            lambda b: tok(b["text"], truncation=True, max_length=cfg["max_length"],
                          padding=False if cfg["dynamic_padding"] else "max_length"),
            batched=True, remove_columns=["text"],
        )

    train_ds, val_ds = to_ds(train_df), to_ds(val_df)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["model_name"], id2label=dict(enumerate(labels)), label2id=label2id  # implies num_labels
    )

    neg = label2id["negative"]

    def compute_metrics(p) -> dict[str, float]:
        pred = np.argmax(p.predictions, axis=-1)
        f1s = f1_score(p.label_ids, pred, average=None, labels=list(range(len(labels))), zero_division=0)
        out = {"macro_f1": float(f1s.mean()), "accuracy": float(accuracy_score(p.label_ids, pred)),
               "negative_f1": float(f1s[neg])}
        out.update({f"f1_{l}": float(f1s[i]) for i, l in enumerate(labels)})
        return out

    class StepTimer(TrainerCallback):
        """Wall time per optimizer step (CUDA-synchronized)."""

        def __init__(self) -> None:
            self.times: list[float] = []
            self._t = None

        def on_step_begin(self, args, state, control, **kw):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self._t = time.perf_counter()

        def on_step_end(self, args, state, control, **kw):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self.times.append(time.perf_counter() - self._t)

    os.environ["WANDB_LOG_MODEL"] = "false"
    use_wandb = wandb_logged_in()
    run_name = f"indobert-{scheme}-{weights_mode}" + ("-smoke" if args.smoke else "")
    if not use_wandb:
        logger.warning("W&B not logged in; skipping W&B logging")

    def start_wandb() -> None:
        """Start the run only once everything is built, so setup errors leave no crashed runs."""
        import wandb

        wandb.init(
            project=cfg["wandb"]["project"], entity=cfg["wandb"]["entity"], name=run_name,
            job_type="smoke" if args.smoke else "train", tags=[scheme, weights_mode, "smoke" if args.smoke else "full"],
            config={"seed": seed, "label_scheme": scheme, "class_weights": weights_mode,
                    "class_weight_values": None if cw is None else cw.tolist(), "train_rows": len(train_df),
                    "val_rows": len(val_df), "model_name": cfg["model_name"], "max_length": cfg["max_length"],
                    **{f"cfg_{k}": v for k, v in tr.items()}},
        )

    targs = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        num_train_epochs=tr["num_train_epochs"], learning_rate=tr["learning_rate"], weight_decay=tr["weight_decay"],
        warmup_steps=tr["warmup"], lr_scheduler_type=tr["lr_scheduler_type"],
        per_device_train_batch_size=tr["per_device_train_batch_size"],
        per_device_eval_batch_size=tr["per_device_eval_batch_size"],
        gradient_accumulation_steps=tr["gradient_accumulation_steps"], fp16=tr["fp16"],
        eval_strategy=tr["eval_strategy"], eval_steps=tr["eval_steps"],
        save_strategy=tr["save_strategy"], save_steps=tr["eval_steps"], save_total_limit=tr["save_total_limit"],
        save_only_model=tr["save_only_model"], load_best_model_at_end=True,
        metric_for_best_model=tr["metric_for_best_model"], greater_is_better=tr["greater_is_better"],
        logging_steps=tr["logging_steps"], dataloader_num_workers=tr["dataloader_num_workers"],
        seed=seed, data_seed=seed, report_to=["wandb"] if use_wandb else "none", run_name=run_name,
    )
    timer = StepTimer()
    trainer = Trainer(
        model=model, args=targs, train_dataset=train_ds, eval_dataset=val_ds, processing_class=tok,
        data_collator=DataCollatorWithPadding(tok), compute_loss_func=make_loss(cw), compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=tr["early_stopping_patience"]), timer],
    )

    if use_wandb:
        start_wandb()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    train_out = trainer.train()
    train_secs = time.perf_counter() - t0
    peak_train = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else None
    peak_train_reserved = torch.cuda.max_memory_reserved() / 2**30 if torch.cuda.is_available() else None

    t1 = time.perf_counter()
    val_metrics = trainer.evaluate()
    eval_secs = time.perf_counter() - t1

    final_dir = out_dir / "final"
    trainer.save_model(str(final_dir))
    tok.save_pretrained(str(final_dir))
    best_checkpoint = trainer.state.best_model_checkpoint
    if not cfg.get("keep_checkpoints", True):
        # final/ holds the best model (load_best_model_at_end); checkpoints are only disk use now
        shutil.rmtree(out_dir / "checkpoints", ignore_errors=True)
        logger.info("deleted %s (kept %s)", out_dir / "checkpoints", final_dir)

    steps = timer.times[2:] if len(timer.times) > 4 else timer.times  # drop warm-up steps
    sec_per_step = float(np.median(steps)) if steps else float("nan")
    run_info: dict[str, Any] = {
        "seed": seed, "label_scheme": scheme, "class_weights": weights_mode, "smoke": args.smoke,
        "labels": labels, "class_weight_values": None if cw is None else cw.tolist(),
        "train_rows": len(train_df), "val_rows": len(val_df),
        "optimizer_steps": int(train_out.global_step), "train_seconds": round(train_secs, 1),
        "sec_per_optimizer_step_median": round(sec_per_step, 4),
        "eval_seconds_val": round(eval_secs, 2),
        "best_checkpoint": best_checkpoint, "best_metric": trainer.state.best_metric,
        "val_metrics": val_metrics,
        "peak_vram_gib": {"allocated": peak_train, "reserved": peak_train_reserved,
                          "device_total": torch.cuda.get_device_properties(0).total_memory / 2**30 if torch.cuda.is_available() else None},
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "transformers": __import__("transformers").__version__},
    }

    if args.smoke and torch.cuda.is_available():
        run_info["stress_max_length_batch"] = stress_test(trainer, tok, cfg)
        full_train = len(apply_scheme(pd.read_parquet(splits / "train.parquet"), scheme)[0])
        full_val = len(apply_scheme(pd.read_parquet(splits / "val.parquet"), scheme)[0])
        eff_batch = tr["per_device_train_batch_size"] * tr["gradient_accumulation_steps"]
        steps_per_epoch = math.ceil(full_train / eff_batch)
        evals_per_epoch = steps_per_epoch / cfg["training"]["eval_steps"]
        eval_full = eval_secs * full_val / len(val_df)
        per_epoch = steps_per_epoch * sec_per_step + evals_per_epoch * eval_full
        run_info["estimate_full_run"] = {
            "train_rows": full_train, "val_rows": full_val, "steps_per_epoch": steps_per_epoch,
            "eval_seconds_full_val_est": round(eval_full, 1), "evals_per_epoch": round(evals_per_epoch, 2),
            "minutes_per_epoch": round(per_epoch / 60, 1),
            "minutes_max_epochs": round(per_epoch * cfg["training"]["num_train_epochs"] / 60, 1),
            "note": "excludes model load and checkpoint saves; smoke batches sampled from the same data",
        }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run_info.json").write_text(json.dumps(run_info, indent=2, default=str), encoding="utf-8")
    if use_wandb:
        import wandb

        wandb.summary.update({"peak_vram_gib_allocated": peak_train, "sec_per_optimizer_step": sec_per_step,
                              **{f"final_val/{k}": v for k, v in val_metrics.items()},
                              **({"stress_peak_vram_gib": run_info["stress_max_length_batch"]["peak_allocated_gib"],
                                  "est_minutes_per_epoch": run_info["estimate_full_run"]["minutes_per_epoch"]}
                                 if "estimate_full_run" in run_info else {})})
        run_info["wandb_url"] = wandb.run.url
        wandb.finish()
        (out_dir / "run_info.json").write_text(json.dumps(run_info, indent=2, default=str), encoding="utf-8")
    logger.info("saved model to %s", final_dir)
    print(json.dumps(run_info, indent=2, default=str))


def stress_test(trainer, tok, cfg: dict[str, Any]) -> dict[str, Any]:
    """Peak VRAM for one full-size optimizer step at max_length (worst case for dynamic padding)."""
    tr = cfg["training"]
    model = trainer.model
    model.train()
    opt = trainer.optimizer
    bs, steps = tr["per_device_train_batch_size"], tr["gradient_accumulation_steps"]
    ids = torch.randint(1000, tok.vocab_size - 1, (bs, cfg["max_length"]), device=model.device)
    batch = {"input_ids": ids, "attention_mask": torch.ones_like(ids),
             "labels": torch.zeros(bs, dtype=torch.long, device=model.device)}
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t = time.perf_counter()
    for _ in range(steps):
        with torch.autocast("cuda", dtype=torch.float16, enabled=tr["fp16"]):
            loss = model(**batch).loss / steps
        loss.backward()
    opt.step()
    opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    return {
        "batch": bs, "grad_accum": steps, "seq_len": cfg["max_length"],
        "seconds_per_optimizer_step": round(time.perf_counter() - t, 3),
        "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3),
        "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 3),
    }


if __name__ == "__main__":
    main()
