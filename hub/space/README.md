---
title: E-wallet Complaint Analyzer
emoji: 💬
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.30.0
python_version: "3.12"
app_file: app.py
suggested_hardware: zero-a10g
pinned: false
license: mit
short_description: Detect complaints in Indonesian e-wallet app reviews
models:
  - k08e24bryant/indobert-ewallet-complaints
  - firqaaa/indo-sentence-bert-base
---

# E-wallet Complaint Analyzer

Paste an Indonesian Google Play review of an e-wallet app (GoPay, OVO, DANA, ShopeePay), optionally
with its star rating, and the demo tells you whether it is a **complaint** and, for complaints,
roughly **what it is about**. A second tab shows trends from 201,860 reviews (Jul–Oct 2026).

- **Model:** [k08e24bryant/indobert-ewallet-complaints](https://huggingface.co/k08e24bryant/indobert-ewallet-complaints)
  (fine-tuned IndoBERT + a BERTopic topic model; see the model card for evaluation and known failure modes)
- **Code:** [github.com/k08e24bryant/ewallet-review-analysis](https://github.com/k08e24bryant/ewallet-review-analysis)

## How it works

- **Complaint** = the text model predicts *negative*, or a 1–2★ rating is given.
  On 600 hand-labeled reviews this rule reached complaint F1 0.922 [0.89, 0.95].
- **Topic** (complaints with 3+ words that the text model reads as negative): the closest of 17
  named complaint topics. Topics are approximate: in a manual check, specific topics were judged
  correct about half the time and broad topics less often.
- Scores are model outputs, not calibrated probabilities. The model was trained on Indonesian;
  English input shows a warning.

## Limitations

- The model finds only about half of complaints written with 4–5 stars, and misses reviews that
  open with praise and complain after "tapi" (but).
- The star rule counts 1–2★ praise as a complaint.
- Trends describe reviews written on Google Play, not all users.
- No review data is stored or published by this Space; inputs are processed in memory only.

Runs on ZeroGPU: a GPU is attached only while a review is being analyzed. The first request after
the Space wakes up can take a while (model loading, GPU queue).
