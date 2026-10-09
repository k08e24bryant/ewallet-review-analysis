---
language: id
license: mit
base_model: indobenchmark/indobert-base-p1
pipeline_tag: text-classification
library_name: transformers
tags:
  - indonesian
  - sentiment
  - complaint-detection
  - app-reviews
  - e-wallet
  - bertopic
---

# IndoBERT e-wallet complaint classifier

A fine-tuned [indobenchmark/indobert-base-p1](https://huggingface.co/indobenchmark/indobert-base-p1)
that reads an Indonesian Google Play review of an e-wallet app (GoPay, OVO, DANA, ShopeePay) and
predicts **negative / neutral / positive**. It is the text part of a complaint detector:

> **complaint = the model predicts `negative`, or the review has a 1–2 star rating**

The repository also contains a BERTopic topic model (`bertopic/`) that assigns an approximate
complaint topic. Both are used in the demo Space
[k08e24bryant/ewallet-complaint-analyzer](https://huggingface.co/spaces/k08e24bryant/ewallet-complaint-analyzer).

## Intended use

- Research and portfolio analysis of complaint trends in public app reviews.
- Flagging likely complaints in Indonesian e-wallet reviews, including complaints written with 4–5 stars.

Not intended for decisions about individual users, for moderating or removing reviews, or for
domains other than Indonesian e-wallet app reviews without re-evaluation.

## How to use

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

repo = "k08e24bryant/indobert-ewallet-complaints"
tok = AutoTokenizer.from_pretrained(repo)
model = AutoModelForSequenceClassification.from_pretrained(repo).eval()

text = "Saldo saya terpotong tapi transfer belum masuk sejak kemarin."
with torch.no_grad():
    probs = torch.softmax(model(**tok(text, truncation=True, max_length=128, return_tensors="pt")).logits, -1)[0]
label = model.config.id2label[int(probs.argmax())]
star = 1                      # optional star rating; None if unknown
is_complaint = label == "negative" or (star is not None and star <= 2)
```

Training used preprocessed text: Unicode NFKC, URLs removed, repeated characters capped at 2,
and emoji replaced by their Indonesian names (Python `emoji` library). For best results apply the
same preprocessing; the demo Space contains the exact code (`demo_core.py`).

## Training data

- **Source:** public Google Play reviews of GoPay, OVO, DANA and ShopeePay (country Indonesia),
  scraped newest-first from Jul 1 to Oct 3, 2026. **The raw review data is not published.**
  User names and profile images were never read or stored.
- **Labels:** weak labels from the star rating (1–2 negative, 3 neutral, 4–5 positive).
  Training set: 45,386 reviews (text deduplicated across apps, down-sampled to 15,000 per app;
  OVO uses all its rows); validation: 5,043 reviews.
- **Training:** 3 classes, no class weights, max length 128, learning rate 2e-5, effective batch
  32, early stopping on validation macro-F1 (stopped at 2,800 steps), seed 42, fp16 on an RTX 3060.
  transformers 5.18, torch 2.14.

## Evaluation

Gold set: 600 reviews labeled by hand from the text only (stars hidden), stratified 50 per app ×
star group. "Reweighted" estimates performance on the pool of unique review texts (each gold row
weighted by its stratum size). 95% CIs from a stratified bootstrap (2,000 draws). The model and
the decision rule were fixed on a separate dev set before the gold set was used once.

**Complaint detection (negative vs not), model + star rule, gold set:**

| | F1 | Precision | Recall |
|---|---|---|---|
| Reweighted | **0.922** [0.890, 0.948] | 0.973 [0.946, 0.993] | 0.876 [0.828, 0.919] |
| Raw (as sampled) | 0.901 [0.879, 0.921] | 0.959 [0.938, 0.978] | 0.849 [0.817, 0.880] |
| Star rating alone (1–2★), reweighted | 0.822 [0.789, 0.855] | | |

Model + star rule minus star rating alone: +0.099 [+0.071, +0.129] F1 (reweighted).

Complaint recall by star group (reweighted): 1–2★ 1.00, 3★ 0.82 [0.75, 0.89], 4–5★ 0.49 [0.31, 0.66].

**The model alone** (no star rule) was evaluated on the dev set (191 reviews), not on gold:
complaint F1 0.859 [0.791, 0.922], precision 0.944, recall 0.788; 3-class macro-F1 0.567
(neutral is rarely predicted correctly; use the binary complaint view).

Label reliability: the annotator relabeled 50 gold reviews blind; Cohen's kappa 0.80 [0.61, 0.95]
(3 classes) and 0.90 [0.73, 1.00] (complaint vs not). One annotator only.

## Topic model (`bertopic/`)

BERTopic fitted on 60,014 unique complaint texts (3+ words), embedded with
[firqaaa/indo-sentence-bert-base](https://huggingface.co/firqaaa/indo-sentence-bert-base)
(apache-2.0) after replacing app names with `[APP]`; 28 clusters merged by hand into 17 named topics.
It is saved with safetensors serialization: a new review gets the topic whose centroid is most
similar (no UMAP/HDBSCAN). This agrees with the full pipeline's assignment for 66.9% of 2,000
fitted documents. In a fresh manual check, 40% [26, 55] of assignments were judged correct
(52% for specific topics, much less for the broad ones). Treat topics as approximate.

## Known failure modes

Observed when testing the demo (scores are softmax outputs, not calibrated probabilities):

| Input | Star | Text model scores (neg / neu / pos) | Demo output |
|---|---|---|---|
| `aplikasinya bagus banget, makasih` | 1 | 0.003 / 0.005 / **0.992** | Complaint, from the star rule only (the text reads positive; possibly a mis-tap). No topic. |
| `bintang 5 deh, tapi kenapa transfer saya gagal terus` | 5 | 0.006 / 0.009 / **0.985** | Not a complaint: a real complaint is missed. |
| `worst app ever` | – | 0.004 / 0.003 / **0.993** | Not a complaint: English is misread (English warning shown). |

The second case is a complaint written with 4–5 stars, the group where the model is weakest:
on the gold set it caught only **49% [31, 66]** of complaints written with 4–5★ (reweighted).

**A friendly opener before "tapi" (but) hides the complaint.** A small check (report only; the
model was not changed) scored five short complaints written for the test, with and without an
opener (CPU, same preprocessing):

| Variant | Predicted negative | Mean negative score |
|---|---|---|
| complaint only (e.g. `kenapa transfer saya gagal terus`) | 5 of 5 | 0.637 |
| `bintang 5 deh, tapi` + complaint | **0 of 5** | 0.007 |
| `oke deh, tapi` + complaint (control: same words, no rating) | **0 of 5** | 0.062 |
| `bintang 1 deh,` + complaint | 5 of 5 | 0.952 |

So the flip comes mainly from the positive opener followed by "tapi", not from the rating alone;
rating talk in the text pushes the scores further in its direction ("bintang 5" toward positive,
"bintang 1" toward negative). A likely cause is the weak training labels: reviews that praise first
and complain after "tapi" often carry 4–5 stars, so they were labeled positive. Five constructed
examples are not a measurement of how often this happens in real reviews.

**English is out of scope.** About 0.5% of the training reviews are English; English text can be
misread completely (see `worst app ever`). The demo shows a warning when it detects English
(lingua restricted to Indonesian, Malay and English).

## Limitations

- The model finds only about half of complaints written with 4–5 stars, and misses mixed reviews that open with praise and complain after "tapi" (see Known failure modes).
- The star rule counts 1–2★ praise as a complaint.
- Informal Indonesian, slang and regional languages (Javanese, Sundanese) vary. English (about 0.5% of reviews) is not supported and can be misread completely.
- Model scores are not calibrated: a 0.99 score can be wrong.
- Reviews from Jul–Oct 2026 only; app features, outages and slang change over time.
- Reviews are self-selected (people who chose to write one), so rates describe reviews, not users.
- Weak training labels come from star ratings, which often disagree with the text.

## License

This model is released under the MIT license, the same as its base model
[indobenchmark/indobert-base-p1](https://huggingface.co/indobenchmark/indobert-base-p1)
(MIT, from its model card). The topic model's embedding model,
[firqaaa/indo-sentence-bert-base](https://huggingface.co/firqaaa/indo-sentence-bert-base), is apache-2.0.
