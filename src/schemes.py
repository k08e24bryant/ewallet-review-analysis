"""Label schemes shared by IndoBERT, the TF-IDF baseline, and dev selection (Phase 5).

    spec_3class            weak_label: 1-2 negative, 3 neutral, 4-5 positive (pre-registered)
    binary_drop_3star      1-2 negative, 4-5 non_negative; 3-star rows dropped
    binary_3star_negative  1-3 negative, 4-5 non_negative
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SCHEMES = ("spec_3class", "binary_drop_3star", "binary_3star_negative")


def apply_scheme(df: pd.DataFrame, scheme: str) -> tuple[pd.DataFrame, list[str]]:
    """Return rows and string labels ('target') for a label scheme, plus the ordered label names."""
    if scheme == "spec_3class":
        return df.assign(target=df["weak_label"]), ["negative", "neutral", "positive"]
    if scheme == "binary_drop_3star":
        out = df[df["score"] != 3]
        return out.assign(target=np.where(out["score"] <= 2, "negative", "non_negative")), ["negative", "non_negative"]
    if scheme == "binary_3star_negative":
        return df.assign(target=np.where(df["score"] <= 3, "negative", "non_negative")), ["negative", "non_negative"]
    raise ValueError(f"unknown label_scheme {scheme!r}; choose from {SCHEMES}")
