# Gold set labeling guidelines

You are labeling **600 reviews** (sheet `gold`) of Indonesian e-wallet
apps by **text only**. Star ratings, app names, and dates are hidden on purpose:
label what the text says, not what you guess the rating was.

Pick one value in the `label` column for every row:
`negative`, `neutral`, `positive`, or `invalid`. Use `notes` only with the
standard tags listed at the end of this file.

## Labels

### negative
A complaint, problem, frustration, or accusation, **even if it is polite**.
Includes bugs, failed transactions, lost balance, slow service, scam/fraud
accusations, and disappointment.

- **Requests framed as complaints are negative.** "kenapa limit saya turun",
  "tolong kembalikan saldo saya" describe a problem the user is unhappy about.

| Review | Why |
|---|---|
| Mohon info tim dana, apakah sedang error ya? top up gak masuk dan gak bisa log in lagi setelah d log out, mohon pencerahan nya | Polite and phrased as a question, but reports a failure |
| mohon maaf ka,, kenapa limit di akun dana sayah hilang,, padahal sayah sudah membayar tepat waktu,, mohon di perbaharui lagi ka apk nya | Request framed as a complaint (limit removed) |
| katanya dapat potongan, ternyata shope payung penipu | Accusation (penipu) |

### neutral
A question, a feature request, or a factual statement **without clear emotion**.
Also **mixed reviews where neither side dominates**.

- A plain feature request ("tolong tambahkan fitur ...") is neutral; if it
  comes with frustration about a problem, it is negative.

| Review | Why |
|---|---|
| bagaimana cara mengaktifkan fitur Dana Cicil? | Plain how-to question, no emotion |
| tambahin fitur login pakai email 🙏 | Feature request without a complaint |

### positive
Praise, satisfaction, gratitude, or recommendation.

| Review | Why |
|---|---|
| aplikasi ini sangat membantu transaksi sehari hari, proses transfer cepat dan mudah, trimakasih DANA | Praise + gratitude |
| dompet Digital yang paling aman dan cepat. sangat recommended buat dipakai sehari hari | Recommendation |
| Dana sangat membantu kelancaran usaha saya,trimksih | Satisfaction |

### invalid
Gibberish, unreadable text, or text with **no connection to any app or
service** (e.g. a comment about TV dramas, random characters, a greeting only).
Invalid rows are replaced from the `reserve` sheet, so use this label sparingly.

- A review about a **related service** (e.g. Gojek drivers or food orders in
  the GoPay app) is **not** invalid — see hard case 7.

| Review | Why |
|---|---|
| jhguj | Gibberish |
| saya suka melihat Drama Korea atau Drama Indonesia, cerita sungguh menarik perhatian para pecinta film atau Drama. Semoga tambah sukses. | Unrelated to any app (about TV dramas) |
| assalamualaikum wr wb | Greeting only, says nothing about the app |

## Rules for hard cases

Examples in this section are illustrative unless marked otherwise.

1. **Mixed reviews:** label the **dominant** sentiment. Use `neutral` only when
   praise and complaint are truly balanced.
   - "aplikasi bagus tapi sering error pas transfer, tolong diperbaiki" → the
     complaint dominates → `negative`.
   - "bagus si tapi iklannya diturunkan ya" (from the data) → mild praise and
     mild complaint are balanced → `neutral`.
2. **Praise + feature request:**
   - Strong praise + a feature wish without frustration → `positive`, tag
     `feature-request` ("sangat cocok dan bagus, tapi belum ada fitur investasi saham").
   - Feature request + frustration or a competitor comparison → `negative`
     ("kok belum ada juga, aplikasi lain udah dari dulu").
3. **Flat or lukewarm reviews** without praise or complaint ("biasa aja",
   "lumayan") → `neutral`.
4. **Polite tone does not change the label.** A courteous complaint is still `negative`.
5. **English or mixed language:** label normally.
6. **Emoji only or very short text:** label it if the meaning is clear
   ("mantap 👍" → `positive`); use `invalid` only if it is unreadable.
7. **Related services (off-product):** reviews in the GoPay app about Gojek
   services (drivers, food orders, rides), or similar cases in other apps, are
   labeled **normally by sentiment**, with tag `off-product`.
8. **Sarcasm:** label the intended meaning ("mantap, saldo hilang lagi" →
   `negative`) and add tag `sarcasm`.
9. **Do not look up** the original review, rating, or app. Label the text as shown.
10. **Rating talk** ("bintang 3 dulu ya", "coba dulu, nanti tambah bintang")
    without praise or complaint → `neutral`. If it includes praise or a
    complaint, label that sentiment instead. Rating talk alone is never `invalid`.
11. **Typos with a clear meaning** ("mantaeb") are labeled by their meaning.
12. **Short complaints** ("ribet", "lemot") are labeled normally; length alone
    is never a reason for `invalid`.

## Notes tags

Use only these tags in `notes`, separated by commas if more than one:

- `unsure` — you picked a label but could argue for another
- `off-product` — about a related service, not the e-wallet itself
- `feature-request` — asks for a new feature
- `sarcasm` — intended meaning differs from literal words
- `english` — mostly English text

## Workflow

- Label the `gold` sheet top to bottom. Do not reorder or delete rows.
- When you mark a row `invalid`, leave it in place. Each invalid row is replaced
  by a reserve row from the same hidden stratum, so the import step (3b) will
  list exactly which `reserve` rows (R....) to label. You may also label the
  whole `reserve` sheet up front; unused reserve labels are ignored.
- If you add a new rule while labeling, write it here and re-check earlier
  rows that it affects.
- Save the file as `.xlsx` with the same name.