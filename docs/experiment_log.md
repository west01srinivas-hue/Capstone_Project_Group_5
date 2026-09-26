# Experiment log

Kept as we build, for the final report's "methods tried / challenges / experiment logs" sections.
Model: Claude Haiku 4.5 (list price $1 / $5 per million input / output tokens). Budget is a hard $5.

## E1. Data source and sampling (2026-09-25)
- HF mirror `mteb/amazon_reviews_multi` cannot be loaded with `datasets` 5.x (legacy loading script). Fix: read the per-language `.jsonl` files directly.
- Files are sorted by star rating, so "first N rows" gave 100% 1-star reviews. Fix: stratified sample, 40 reviews per star per language (1,200 total). Labels are 0-4 (= 1-5 stars).
- The mirror has no product category or review date, so those schema fields are empty for MARC.
- Verified: no encoding damage in any language (the odd apostrophe seen earlier was only the Windows console).

## E2. Ingestion workflow in n8n (2026-09-26)
- Workflow 01 verified by a headless local run: 1,200 items, 200 per language, 240 per star, unique ids.

## E3. Translation: model comparison (2026-09-26)
Same prompt (`prompts/translate_v1.txt`), 8 reviews per call, 5 non-English languages, 40 reviews.

| | Groq Qwen 27B | Claude Haiku 4.5 |
|---|---|---|
| Valid JSON, ids match, language detected correctly | 40/40 | 40/40 |
| Latency per batch | 1.5 s to 58 s (erratic) | 5.6 s to 21.7 s |
| Cost | free tier | $0.0218 for 40 (about $0.54 per 1,000 reviews) |

Decision: Claude only. Matches the submitted proposal and avoids Groq's latency spikes.
Groq also no longer offers any Llama chat models (only gpt-oss and Qwen), which the earlier plan assumed.

## E4. Translation workflow 02 (2026-09-26)
- Local headless run and the user's n8n Cloud run agree: 120 reviews, 100 translated `ok`, 20 English `not_needed`, 0 failed, 0 language mismatches, 12,436 in / 9,426-9,531 out tokens, about $0.06.
- Failure test with an invalid key: run completes, all 100 marked `failed` with the API error recorded, nothing crashes.
- Note: n8n's "continue on error" shows green even when calls fail, so always read the summary node's `by_status`, not just the tick.

## E5. Sentiment classification (2026-09-26)
Ground truth from stars: 1-2 negative, 3 neutral, 4-5 positive. Split per language and star: 240 DEV (prompt tuning) and 960 TEST (reported once).
"Binary acc" = accuracy on positive/negative reviews only, where a neutral prediction counts as wrong (strict).
Baseline: TF-IDF character n-grams + logistic regression, one model per language, trained on 30,000 MARC validation-split reviews.

| Run | Split | 3-way acc | Macro-F1 | Binary acc | Cost |
|---|---|---|---|---|---|
| TF-IDF baseline | dev | 68.8% | 0.611 | 79.7% | free |
| Claude, prompt v1 | dev | 77.5% | 0.733 | 83.9% | $0.088 |
| Claude, prompt v2 | dev | 77.5% | 0.717 | 86.5% | $0.090 |
| TF-IDF baseline | **test** | 72.9% | 0.654 | 83.9% | free |
| **Claude, prompt v2** | **test** | **77.3%** | **0.697** | **88.5%** | $0.352 |

Prompt v2 (only change: clearer rules for minor caveats vs genuinely torn reviews) was chosen on dev because binary accuracy improved; the dev difference is within noise (240 reviews), so treat the choice as a judgment call, not proof.

Test, per language (binary acc): en 89.1, de 89.1, es 87.5, fr 95.3, ja 89.1, zh 81.2. Chinese is weakest for Claude; the baseline is stronger there (84.4).
Test confusion (rows = truth):

| truth \ predicted | negative | neutral | positive |
|---|---|---|---|
| negative (384) | 357 | 22 | 5 |
| neutral (192) | 101 | 62 | 29 |
| positive (384) | 17 | 44 | 323 |

Findings:
- Proposal target is 85%+ for binary and ternary. **Binary is met (88.5%, in line with the 88-92% literature range). 3-way is not (77.3%).**
- The gap is almost entirely the neutral class: recall is 32% (62 of 192); 101 neutral (3-star) reviews were called negative. Negative recall is 93%, positive 84%.
- Star ratings are a noisy proxy for text sentiment. A 3-star review like "I love the sound but it turns itself back on" is genuinely mixed, so some "errors" are label noise. This matches the published result that 5-star MARC classification tops out near 59-70%.
- Topic labels (not validated yet, no ground truth): quality 536, features 199, shipping 104, price 49, other 37, ux 19, support 16.

## E6. Trying to lift the neutral class (2026-09-26)
Prompt v3 = v2 plus an estimated 1-5 star rating per review. A small logistic-regression "stacker" combines Claude's label, confidence and star estimate with the TF-IDF baseline's probabilities.
Stacker training data: a separate 600-review sample from the MARC *train* split (100 per language, 20 per star; the baseline never saw it). Settings chosen by 5-fold cross-validation on that sample only. The 960 test reviews were scored once and never trained on.

Test set (960):

| Model | 3-way acc | Macro-F1 | Neutral recall | Binary acc |
|---|---|---|---|---|
| TF-IDF baseline | 72.9% | 0.654 | 29.2% | 83.9% |
| Claude v2 label | 77.3% | 0.697 | 32.3% | 88.5% |
| Claude v3 label | 77.7% | 0.707 | 35.4% | 88.3% |
| **Claude v3 star estimate -> label** | **78.1%** | **0.714** | 37.0% | **88.4%** |
| Stacker, Claude features | 78.0% | 0.699 | 30.7% | 89.8% |
| Stacker, Claude features, balanced | 78.0% | 0.727 | 45.3% | 86.2% |
| Stacker, Claude + baseline | 78.1% | 0.708 | 34.4% | 89.1% |
| Stacker, Claude + baseline, balanced (CV-selected) | 75.4% | 0.720 | 57.8% | 79.8% |

Findings:
- No method reaches 85% on 3-way. Everything lands at 75-78%. The stacker adds essentially nothing to accuracy; the baseline's probabilities add almost nothing on top of Claude.
- There is a trade-off, not a fix: "balanced" stackers find more neutrals (recall up to 58%) but push borderline positive/negative reviews into neutral, so binary accuracy falls from 88% to 80%, below the 85% target.
- Recommended production classifier: the plain Claude v3 star estimate mapped to a label. It is the simplest and cheapest (no stacker), and keeps binary accuracy at 88.4%. The balanced stacker can be offered as an optional "flag borderline reviews" mode.
- Star estimate vs true stars on test: MAE 0.472, exact match 58.0%, within one star 95.6% (Chinese weakest: MAE 0.63). Exact-match accuracy is in line with the published 59-70% for fine-grained MARC. **The proposal's star MAE <= 0.15 target is not realistic for this data; explain this in the report rather than chase it.**
- Star labels are a noisy proxy for text sentiment (E5), so the ceiling for any text-only model is well below 85% on 3-way.
- One stray topic label appeared ("compatibility", 1 review, outside the taxonomy); ignore or map to "features".

## Spend so far (approx.)
Translation tests $0.02 + $0.06 (local) + $0.06 (n8n Cloud); sentiment E5 $0.53; E6 $0.25 (train sample) + $0.40 (test, v3) = $0.65. Total about $1.33 of $5.

## Next
- Manual spot-check of topics on about 100 reviews (proposal metric: extraction precision).
- Clustering and trend detection (Stage 4).
