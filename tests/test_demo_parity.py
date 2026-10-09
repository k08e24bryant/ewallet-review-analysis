"""Phase 8a parity tests: the demo (app/demo_core.py) must reproduce the training pipeline.

- preprocessing: text_clean, model_text, n_words vs src/preprocess.py (functions and the
  stored reviews_clean.parquet columns) on 200 random reviews
- brand neutralization vs src/topics_v2.py on the same 200 texts
- predictions vs data/processed/predictions.parquet on 200 random rows (CPU):
  same argmax label and complaint verdict, probabilities within 1e-3

Run:
    uv run pytest tests/test_demo_parity.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
sys.path.insert(0, str(ROOT))

from demo_core import TextPrep, load_config  # noqa: E402

N, SEED = 200, 42
CLEAN = ROOT / "data/processed/reviews_clean.parquet"
PRED = ROOT / "data/processed/predictions.parquet"
pytestmark = pytest.mark.skipif(not CLEAN.exists() or not PRED.exists(), reason="processed data not available")


@pytest.fixture(scope="module")
def cfg():
    return load_config(ROOT / "app/config.yaml")


@pytest.fixture(scope="module")
def prep(cfg):
    return TextPrep(cfg)


@pytest.fixture(scope="module")
def sample():
    df = pd.read_parquet(CLEAN, columns=["reviewId", "content", "text_clean", "model_text", "n_words", "score"])
    return df.sample(n=N, random_state=SEED).reset_index(drop=True)


def test_preprocessing_matches_stored_columns(prep, sample):
    for r in sample.itertuples():
        tc = prep.clean(r.content)
        assert tc == r.text_clean, r.reviewId
        assert prep.model_text(tc) == r.model_text, r.reviewId
        assert prep.n_words(tc) == r.n_words, r.reviewId


def test_preprocessing_matches_src_functions(prep, sample):
    from src import preprocess as pp

    pcfg = pp.load_config(ROOT / "configs/preprocess.yaml")
    clean = pp.make_cleaner(pcfg["cleaning"]["max_char_repeat"], pcfg["cleaning"]["unicode_normalization"])
    demojize, _ = pp.make_demojizer(pcfg["model_text"], pcfg["cleaning"]["max_char_repeat"])
    for r in sample.itertuples():
        assert prep.clean(r.content) == clean(r.content), r.reviewId
        assert prep.model_text(clean(r.content)) == demojize(clean(r.content)), r.reviewId


def test_neutralization_matches_topics_v2(prep, sample):
    from src import topics_v2 as v2

    vcfg = v2.load_config(ROOT / "configs/topics_v2.yaml")
    fn = v2.neutralizer(vcfg["neutralize"])
    texts = sample["text_clean"].tolist() + ["GoPay error", "gopaynya", "ShopeePay PayLater", "spay", "dana saya hilang", "OVOnya"]
    for t in texts:
        assert prep.neutralize(t) == fn(t)[0], t


@pytest.fixture(scope="module")
def analyzer(cfg):
    from demo_core import Analyzer

    return Analyzer(cfg)


def test_predictions_match_pipeline(analyzer, prep):
    pred = pd.read_parquet(PRED, columns=["reviewId", "score", "pred_label", "p_negative", "p_neutral", "p_positive",
                                          "is_complaint"])
    rows = pred.sample(n=N, random_state=SEED).merge(pd.read_parquet(CLEAN, columns=["reviewId", "content"]), on="reviewId")
    model_texts = [prep.model_text(prep.clean(c)) for c in rows["content"]]
    probs = analyzer.classify(model_texts)
    labels = np.array(analyzer.labels)[probs.argmax(axis=1)]
    assert (labels == rows["pred_label"].to_numpy()).all()
    expected = rows[[f"p_{l}" for l in analyzer.labels]].to_numpy()
    assert np.abs(probs - expected).max() < 1e-3
    star_rule = analyzer.cfg["complaint_rule"]["star_negative_max"]
    verdict = (rows["score"].to_numpy() <= star_rule) | (labels == "negative")
    assert (verdict == rows["is_complaint"].to_numpy()).all()
    # end-to-end analyze() on a few rows (raw text + star)
    for r in rows.head(10).itertuples():
        assert analyzer.analyze(r.content, int(r.score)).is_complaint == bool(r.is_complaint)
