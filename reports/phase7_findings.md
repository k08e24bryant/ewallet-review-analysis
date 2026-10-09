# Phase 7 findings: complaints per e-wallet over time

Data: 201,860 Google Play reviews (Jul 1 – Oct 3, 2026). Complaints = PRIMARY (IndoBERT +
"star 1–2 OR model"; gold complaint F1 0.922 [0.89, 0.95]). Complete Monday–Sunday weeks
Jul 6 – Sep 21 (12 weeks); the Sep 28 week is partial (5.7 days). Dates in UTC (WIB = UTC+7).
95% CIs: Wilson for shares, Newcombe for differences. Numbers: `reports/phase7_trends.json`.

## Findings

1. **OVO is the most complained-about app by far:** over complete weeks, 78.4% [77.3, 79.4] of OVO reviews are complaints, vs 37.9% [37.6, 38.2] for DANA, 24.9% [24.5, 25.3] for GoPay and 12.5% [12.1, 12.9] for ShopeePay (all reviews, duplicates included; `phase7_weekly_complaint_share.png`).

2. **The spikes are single-day events, not lasting shifts:** DANA's late-September surge is two separate days, Sep 25 (12,023 reviews, 72% of that week, 83.5% [82.9, 84.2] complaints) and Oct 1 (8,707 reviews, 85.9% [85.1, 86.6]), with normal volume (~790–950 reviews/day) on the other days (`phase7_spike_daily.png`).

3. **DANA's Jul 20 spike was about reward points:** "Reward points cannot be redeemed" grew from 1.3% to 14.1% of DANA's complaints that week (+12.8 pp [11.8, 13.8], 724 unique complaint texts), followed by "Cannot pay or transfer" (+10.2 pp [9.2, 11.2]) (`phase7_spike_topics.png`).

4. **DANA's Sep 25 and Oct 1 spikes were about access and payments:** "Network error or lag despite good signal" gained the most in the week of Sep 21 (+7.7 pp [7.1, 8.4], 1,005 texts) and "Cannot pay or transfer" in the partial week of Sep 28 (+7.7 pp [7.0, 8.5], 854 texts) (`phase7_spike_topics.png`).

5. **GoPay's Jul 28 spike was an outage day:** 4,357 reviews in one day with 81.2% [80.0, 82.4] complaints; the broad outage topic grew most (+28.5 pp [26.8, 30.3]), and among specific topics "Cannot pay or transfer" (+5.2 pp [4.3, 6.2]) (`phase7_spike_daily.png`, `phase7_spike_topics.png`).

6. **ShopeePay's complaint share doubled without a volume spike:** 23.1% [21.6, 24.7] in the week of Sep 21 (2,703 reviews, normal volume) vs a median of 11.2% for Jul 6 – Sep 7, and 22.1% [20.5, 23.7] in the partial week after, which makes it the newest sustained change in the data (`phase7_weekly_complaint_share.png`).

7. **Hidden complaints are common for OVO and rose during DANA's spikes:** the text model flags 16.9% [15.0, 19.0] of OVO's 4–5★ reviews as complaints vs 1.2% [1.1, 1.3] for ShopeePay, and DANA's rate rose from 4.7% [4.1, 5.3] in the week of Sep 14 to 16.9% [16.0, 17.9] in the week of Sep 21; these are lower bounds, since the model finds about half of 4–5★ complaints on gold (49% [31, 66]) (`phase7_hidden_complaints.png`).

8. **The main specific complaint differs by app:** money lost, missing or taken without consent is the largest specific topic everywhere and dominates OVO (32.8% [31.4, 34.3] of its unique non-short complaint texts), while DANA Cicil defines DANA (10.2% [9.9, 10.6]) and loans/paylater stand out for ShopeePay (9.9% [8.9, 10.9]) and GoPay (7.9% [7.4, 8.4]) (`phase7_topic_trends.png`).

Secondary (versions): DANA v2.145 shows 68.8% [68.0, 69.7] complaints, but 72% of its reviews were posted on the three spike days; excluding those days it is 30.3% [28.8, 31.9] vs 20.8% [19.9, 21.6] for v2.139, so the version gap is mostly spike timing, not a version effect (`phase7_versions.png`).

## Limitations

- **Topic fit is modest.** On a fresh 60-row check, 40% [26, 55] of fitted documents fit their topic ("yes"); specific topics 52%, the two broad topics 29%. Topic statements are weaker than complaint-share statements.
- **Short reviews carry no reliable topic.** Topic trends use fitted documents only (unique, non-short texts); short or duplicate rows assigned by transform fit their topic in 15% [5, 36] of cases and are excluded.
- **Star-rule false positives.** "Star 1–2 OR model" counts 1–2★ praise as complaints (4 of the 20 transform rows in the fresh check were 1–2★ praise). Gold precision is 0.973, so the effect is small but not zero; gold recall is 0.876, so complaint shares are slightly understated overall and 4–5★ complaints are understated most.
- **Denominators differ by measure.** Complaint shares use all reviews, duplicates included (33.1% overall; 51.4% on unique review texts, because duplicates are mostly short praise). Topic shares use unique, non-short complaint texts. Hidden complaints use all 4–5★ reviews.
- **Spike weeks dilute topic shares.** In a spike week the denominator jumps, so steady topics drop in share without fewer complaints (DANA Cicil falls to 3.4% in the week of Sep 21 vs 23% the week before).
- **Partial and lagged data.** The Sep 28 week has 5.7 days (the feed lags about 24 h) and is compared per day; the Jun 29 week is dropped. Dates are UTC.
- **Versions are installed versions,** not release timing, and new versions can coincide with spike days (see the secondary finding).
- **No causal claims.** Spike causes are inferred from review text only; read `reports/phase7_spike_examples.csv` (10 random complaint texts per spike week for the top 2 rising topics) before naming a cause.
