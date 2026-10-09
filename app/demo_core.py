"""Self-contained inference for the demo (no imports from src/), so app/ can run as a HF Space.

PRIMARY = IndoBERT spec_3class/none + "star 1-2 OR model predicts negative". Preprocessing
matches src/preprocess.py (text_clean, model_text) and src/topics_v2.py (brand neutralization
for the topic embedding); tests/test_demo_parity.py checks both.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
WHITESPACE_RE = re.compile(r"\s+")
SKIN_TONE_RE = re.compile("[\U0001F3FB-\U0001F3FF]")


def load_config(path: Path) -> dict[str, Any]:
    """Load the demo YAML config and remember its directory for relative paths."""
    with Path(path).open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_dir"] = str(Path(path).resolve().parent)
    return cfg


# ---------------------------------------------------------------- preprocessing

class TextPrep:
    """text_clean, model_text, word count and brand neutralization, as in the training pipeline."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        import emoji

        p = cfg["preprocessing"]
        self.form = p["unicode_normalization"]
        self.max_repeat = p["max_char_repeat"]
        self.repeat_re = re.compile(r"(\D)\1{%d,}" % self.max_repeat, re.IGNORECASE)
        self.lang, self.fallback = p["emoji_language"], p["emoji_fallback_language"]
        self.left, self.right = p["emoji_delimiters"]
        self.underscores = p["underscores_to_spaces"]
        self.strip_skin = p["strip_skin_tones"]
        if self.lang != "en":
            emoji.config.load_language(self.lang)
        n = cfg["neutralize"]
        clit = "|".join(n["clitics"])
        self.rules = [(re.compile(rf"\b{x}(?=(?:{clit})?\b)", re.IGNORECASE), "paylater") for x in n["paylater"]]
        self.rules += [(re.compile(rf"\b{x}(?=(?:{clit})?\b)", re.IGNORECASE), n["token"]) for x in n["brands"]]

    def _cap_emoji_runs(self, text: str) -> str:
        import emoji

        found = emoji.emoji_list(text)
        if not found:
            return text
        parts, pos, run, prev_end, prev = [], 0, 0, -1, None
        for e in found:
            start, end = e["match_start"], e["match_end"]
            run = run + 1 if (start == prev_end and e["emoji"] == prev) else 1
            parts.append(text[pos:start])
            if run <= self.max_repeat:
                parts.append(text[start:end])
            pos, prev_end, prev = end, end, e["emoji"]
        parts.append(text[pos:])
        return "".join(parts)

    def clean(self, text: str | None) -> str:
        """text_clean: NFKC, URLs removed, repeated non-digit chars and emoji capped, whitespace collapsed."""
        if not text:
            return ""
        if self.form:
            text = unicodedata.normalize(self.form, text)
        text = URL_RE.sub(" ", text)
        text = self.repeat_re.sub(r"\1" * self.max_repeat, text)
        text = self._cap_emoji_runs(text)
        return WHITESPACE_RE.sub(" ", text).strip()

    def model_text(self, text_clean: str) -> str:
        """model_text: skin tones removed, emoji replaced by their (Indonesian) names."""
        import emoji

        def name_of(emj: str, data: dict[str, Any]) -> str:
            name = data.get(self.lang)
            if name is None:
                name = data[self.fallback]
            name = name.strip(":")
            if self.underscores:
                name = name.replace("_", " ")
            return f"{self.left}{name}{self.right}"

        text = text_clean
        if self.strip_skin:
            text = self._cap_emoji_runs(SKIN_TONE_RE.sub("", text))
        return WHITESPACE_RE.sub(" ", emoji.replace_emoji(text, replace=name_of)).strip()

    @staticmethod
    def n_words(text_clean: str) -> int:
        """Whitespace word count of text_clean (is_short: fewer than 3)."""
        return len(text_clean.split())

    def neutralize(self, text: str) -> str:
        """Topic-embedding input: paylater product names -> 'paylater', brands -> '[APP]'."""
        for rx, repl in self.rules:
            text = rx.sub(repl, text)
        return text


# ---------------------------------------------------------------- models

def _resolve(cfg: dict[str, Any], key: str) -> str:
    """Local directory (relative to the config) or a snapshot of the Hub repo subfolder."""
    m = cfg["models"]
    if m["source"] == "local":
        return str((Path(cfg["_dir"]) / m[key]["local"]).resolve())
    from huggingface_hub import snapshot_download

    sub = m[key]["hub_subfolder"]
    patterns = [f"{sub}/*"] if sub else ["*.json", "*.safetensors", "*.txt", "*.model"]
    root = snapshot_download(m["hub_repo"], allow_patterns=patterns)
    return str(Path(root) / sub) if sub else root


@dataclass
class Result:
    """One analyzed review."""

    text_clean: str
    model_text: str
    star: int | None
    probs: dict[str, float]
    model_label: str
    is_complaint: bool
    flagged_by: str
    topic_id: int | None
    topic_name: str | None
    topic_similarity: float | None
    topic_note: str | None
    keywords: list[str]
    language_warning: str | None = None


class Analyzer:
    """Loads the classifier, the embedding model and the topic model once; analyzes reviews."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        import torch
        from sentence_transformers import SentenceTransformer
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.cfg = cfg
        self.prep = TextPrep(cfg)
        m = cfg["models"]
        self.device = m["device"]
        clf_dir = _resolve(cfg, "classifier")
        self.tok = AutoTokenizer.from_pretrained(clf_dir)
        self.clf = AutoModelForSequenceClassification.from_pretrained(clf_dir).to(self.device).eval()
        self.labels = [self.clf.config.id2label[i] for i in range(self.clf.config.num_labels)]
        self.max_length = m["classifier"]["max_length"]
        self.embedder = SentenceTransformer(m["embedding_model"], device=self.device)
        self.embedder.max_seq_length = m["embedding_max_seq_length"]
        self._torch = torch
        self._load_topics(_resolve(cfg, "topics"))
        from lingua import Language, LanguageDetectorBuilder

        lc = cfg["language_check"]
        self.lang_detector = LanguageDetectorBuilder.from_languages(*[getattr(Language, c) for c in lc["candidates"]]).build()

    def language_warning(self, text_clean: str) -> str | None:
        """Warning if a text of min_words+ words is detected as English (restricted lingua, as in Phase 2)."""
        lc = self.cfg["language_check"]
        if self.prep.n_words(text_clean) < lc["min_words"]:
            return None
        lang = self.lang_detector.detect_language_of(text_clean)
        return lc["message"] if lang is not None and lang.name == lc["warn_on"] else None

    def _load_topics(self, path: str) -> None:
        from bertopic import BERTopic

        tm = BERTopic.load(path)
        ids = sorted(tm.get_topics())
        self.topic_ids = np.array(ids)
        emb = np.asarray(tm.topic_embeddings_, dtype=np.float32)
        self.topic_emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        labels = tm.custom_labels_ or [tm.topic_labels_[t] for t in ids]
        self.topic_names = dict(zip(ids, labels))
        self.topic_keywords = {t: [w for w, _ in (tm.get_topic(t) or [])][:6] for t in ids}

    # -------------------------------------------------------- classifier

    def classify(self, model_texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Softmax probabilities (fp32) for model_text inputs, in label order."""
        torch = self._torch
        out = []
        with torch.no_grad():
            for i in range(0, len(model_texts), batch_size):
                enc = self.tok(model_texts[i : i + batch_size], truncation=True, max_length=self.max_length,
                               padding=True, return_tensors="pt")
                logits = self.clf(**{k: v.to(self.device) for k, v in enc.items()}).logits.float()
                out.append(torch.softmax(logits, dim=-1).cpu().numpy())
        return np.vstack(out)

    # -------------------------------------------------------- topics

    def embed_for_topics(self, texts_clean: list[str]) -> np.ndarray:
        """Brand-neutral sentence embeddings (unit length) for topic assignment."""
        emb = self.embedder.encode([self.prep.neutralize(t) for t in texts_clean], convert_to_numpy=True,
                                   show_progress_bar=False, batch_size=64)
        return emb / np.linalg.norm(emb, axis=1, keepdims=True)

    def assign_topics(self, emb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Topic ids and cosine similarities, using the configured assignment rule."""
        sim = emb @ self.topic_emb.T
        tcfg = self.cfg["topics"]
        if tcfg["assignment"] == "argmax_all":          # outlier centroid competes like any topic
            best = sim.argmax(axis=1)
            return self.topic_ids[best], sim[np.arange(len(sim)), best]
        if tcfg["assignment"] == "threshold":           # named topics only; low similarity -> outlier
            named = self.topic_ids != -1
            s = sim[:, named]
            best = s.argmax(axis=1)
            top = s[np.arange(len(s)), best]
            ids = np.where(top >= tcfg["similarity_threshold"], self.topic_ids[named][best], -1)
            return ids, top
        raise ValueError(f"unknown topics.assignment {tcfg['assignment']!r}")

    def topic_note(self, topic_id: int) -> str:
        """Reliability note for a topic."""
        notes = self.cfg["topics"]["notes"]
        name = self.topic_names[topic_id]
        if topic_id == -1:
            return notes["outlier"]
        if name == "Unspecified complaint":
            return notes["unspecified"]
        return notes["low"] if name in self.cfg["topics"]["low_reliability"] else notes["specific"]

    # -------------------------------------------------------- end to end

    def analyze(self, text: str, star: int | None) -> Result:
        """PRIMARY verdict, class probabilities and (for complaints with 3+ words) a topic."""
        rule = self.cfg["complaint_rule"]
        tc = self.prep.clean(text)
        mt = self.prep.model_text(tc)
        p = self.classify([mt])[0]
        label = self.labels[int(p.argmax())]
        star_neg = star is not None and star <= rule["star_negative_max"]
        model_neg = label == "negative"
        is_complaint = star_neg or model_neg
        flagged = "star+model" if (star_neg and model_neg) else "star rule" if star_neg else "model" if model_neg else "-"
        topic_id = topic_name = sim = note = None
        keywords: list[str] = []
        if is_complaint:
            if rule["topic_only_if_model_negative"] and not model_neg:
                note = self.cfg["topics"]["notes"]["star_only"].format(label=label)
            elif self.prep.n_words(tc) < rule["min_words_for_topic"]:
                note = self.cfg["topics"]["notes"]["short"]
            else:
                ids, sims = self.assign_topics(self.embed_for_topics([tc]))
                topic_id, sim = int(ids[0]), float(sims[0])
                topic_name = self.topic_names[topic_id] if topic_id != -1 else None
                note = self.topic_note(topic_id)
                keywords = self.topic_keywords.get(topic_id, []) if topic_id != -1 else []
        return Result(tc, mt, star, {l: float(v) for l, v in zip(self.labels, p)}, label, is_complaint, flagged,
                      topic_id, topic_name, sim, note, keywords, self.language_warning(tc))
