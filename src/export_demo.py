"""Phase 8a: export the v3 topic model for the demo (safetensors, embedding similarity only).

The pickled v3 model needs UMAP + HDBSCAN for transform. The demo instead assigns a topic by
cosine similarity between the review's brand-neutral sentence embedding and each final topic's
centroid. Steps:
    1. topic embeddings = centroids of the brand-neutral embeddings of the fitted documents
       per final v3 topic (outliers included as their own centroid)
    2. save with BERTopic safetensors serialization (+ c-TF-IDF, labels)
    3. measure agreement with the pipeline's assignments on 2,000 random fitted documents,
       embedded on CPU exactly as the demo does, for two assignment rules:
         argmax_all  - argmax over all centroids, the outlier centroid included
         threshold   - argmax over named topics; cosine below the v3 outlier threshold -> outlier
       the rule with the higher overall agreement is written to app/config.yaml
No gold or dev labels are read.

Usage:
    uv run python -m src.export_demo
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src import topics as tp
from src import topics_v2 as v2

logger = logging.getLogger("export_demo")

APP_CONFIG = Path("app/config.yaml")
REPORT = Path("reports/phase8_demo.json")


def main() -> None:
    """CLI entry point."""
    sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    for noisy in ("httpx", "huggingface_hub", "BERTopic", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    from bertopic import BERTopic

    sys.path.insert(0, str(Path("app").resolve()))
    from demo_core import Analyzer, load_config as load_app_config

    seed, n_check = 42, 2000
    tcfg = v2.load_config(Path("configs/topics.yaml"))
    vcfg = v2.load_config(Path("configs/topics_v2.yaml"))
    t3cfg = v2.load_config(Path("configs/topics_v3.yaml"))
    acfg = load_app_config(APP_CONFIG)
    out_dir = (APP_CONFIG.parent / acfg["models"]["topics"]["local"]).resolve()
    spec = next(s for s in vcfg["embeddings"] if s["name"] == t3cfg["embedding"])
    if spec["model"] != acfg["models"]["embedding_model"]:
        raise ValueError("demo embedding model differs from the v3 embedding model")

    model = BERTopic.load(t3cfg["model_out"], embedding_model=spec["model"])
    comp = tp.load_complaints(tcfg)
    fit_df = comp[comp["in_fit"]].reset_index(drop=True)
    assign = pd.read_parquet(t3cfg["assignments_path"], columns=["reviewId", "topic", "topic_name"]).set_index("reviewId")
    topics = assign.loc[fit_df["reviewId"], "topic"].to_numpy()
    if not np.array_equal(topics, np.asarray(model.topics_)):
        raise RuntimeError("final v3 model topics differ from topic_assignments.parquet")
    docs = fit_df[tcfg["text_col"]].tolist()

    # 1. centroids of brand-neutral embeddings (cached GPU embeddings from 6c), outliers included
    emb = v2.embed(v2.neutralize_all(docs, vcfg["neutralize"])[0], spec, vcfg)
    ids = sorted(int(t) for t in set(topics))
    if ids != list(range(-1, len(ids) - 1)):
        raise RuntimeError(f"unexpected topic ids {ids}")
    centroids = np.vstack([emb[topics == t].mean(axis=0) for t in ids])
    model.topic_embeddings_ = centroids

    # 2. safetensors export
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(out_dir), serialization="safetensors", save_ctfidf=True, save_embedding_model=spec["model"])
    reloaded = BERTopic.load(str(out_dir))
    labels = reloaded.custom_labels_
    logger.info("saved %s: %d topics, labels %s...", out_dir, len(ids), labels[:3])

    # 3. agreement on 2,000 fitted documents, embedded on CPU like the demo
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(docs), size=n_check, replace=False)
    acfg_cpu = {**acfg, "topics": {**acfg["topics"], "assignment": "argmax_all"}}
    analyzer = Analyzer(acfg_cpu)
    t0 = time.time()
    demo_emb = analyzer.embed_for_topics([docs[i] for i in pick])
    secs = time.time() - t0
    truth = topics[pick]
    gpu_sim = float(np.mean(np.sum(demo_emb * (emb[pick] / np.linalg.norm(emb[pick], axis=1, keepdims=True)), axis=1)))
    threshold = json.loads(Path(t3cfg["v3_report"]).read_text(encoding="utf-8"))["fit"]["threshold"]

    results = {}
    for rule in ("argmax_all", "threshold"):
        analyzer.cfg = {**acfg, "topics": {**acfg["topics"], "assignment": rule, "similarity_threshold": threshold}}
        pred, _ = analyzer.assign_topics(demo_emb)
        named = truth != -1
        results[rule] = {"overall": float(np.mean(pred == truth)),
                         "on_pipeline_named_topics": float(np.mean(pred[named] == truth[named])),
                         "pipeline_outliers_kept_as_outlier": float(np.mean(pred[~named] == -1)) if (~named).any() else None,
                         "predicted_outlier_share": float(np.mean(pred == -1))}
    best = max(results, key=lambda r: results[r]["overall"])
    per_topic = {}
    analyzer.cfg = {**acfg, "topics": {**acfg["topics"], "assignment": best, "similarity_threshold": threshold}}
    pred, _ = analyzer.assign_topics(demo_emb)
    for t in ids:
        m = truth == t
        if m.any():
            per_topic[analyzer.topic_names[t]] = {"n": int(m.sum()), "agreement": float(np.mean(pred[m] == t))}

    # write the chosen rule into app/config.yaml (text edit keeps comments)
    text = APP_CONFIG.read_text(encoding="utf-8")
    text = re.sub(r"(?m)^  assignment: .*$", f"  assignment: {best}                 # chosen by src/export_demo.py "
                  f"(agreement {results[best]['overall']:.1%} on {n_check} fitted docs)", text)
    thr_line = f"{threshold:.4f}" if best == "threshold" else "null"
    text = re.sub(r"(?m)^  similarity_threshold: .*$", f"  similarity_threshold: {thr_line}        # v3 outlier-reduction "
                  f"threshold (used in threshold mode)", text)
    APP_CONFIG.write_text(text, encoding="utf-8")

    rep = json.loads(REPORT.read_text(encoding="utf-8")) if REPORT.exists() else {"seed": seed}
    rep["topic_export"] = {
        "path": str(out_dir), "serialization": "safetensors (+ c-TF-IDF, labels)", "embedding_model": spec["model"],
        "topic_embeddings": "centroids of brand-neutral embeddings of fitted documents per final v3 topic (outliers included)",
        "agreement_check": {"n": n_check, "seed": seed, "embedded_on": "cpu (demo path)", "cpu_seconds": round(secs, 1),
                            "mean_cosine_cpu_vs_pipeline_embeddings": gpu_sim, "rules": results, "chosen": best,
                            "per_topic_chosen_rule": per_topic},
    }
    REPORT.write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(rep["topic_export"]["agreement_check"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
