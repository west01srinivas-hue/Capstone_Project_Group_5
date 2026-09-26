# Experiment log

Kept as we build, for the final report's "methods tried / challenges / experiment logs" sections.
Model: Claude Haiku 4.5 (list price $1 / $5 per million input / output tokens). Budget is a hard $5.

## E1. Data source and sampling (2026-09-25)
- HF mirror `mteb/amazon_reviews_multi` cannot be loaded with `datasets` 5.x (legacy loading script). Fix: read the per-language `.jsonl` files directly.
- Files are sorted by star rating, so "first N rows" gave 100% 1-star reviews. Fix: stratified sample, 40 reviews per star per language (1,200 total). Labels are 0-4 (= 1-5 stars).
- The mirror has no product category or review date, so those schema fields are empty for MARC.
- Verified: no encoding damage in any language (the odd apostrophe seen earlier was only the Windows console).

**Switched to the Kaggle copy as the source of record (2026-09-26).** The team's downloaded Kaggle dataset `mexwell/amazon-reviews-multi` (train 330 MB, validation, test; 30,000 rows per test/validation file, 5,000 per language, 6,000 per star) was compared with the Hugging Face copy: all 1,200 sample reviews are in Kaggle's `test.csv` with identical review ids, star ratings and text (title + body), so every Claude result already paid for stays valid. Kaggle adds `product_category` (31 categories), `product_id` and `reviewer_id`, and still has no review date. `scripts/enrich_from_kaggle.py` joined the categories into both samples after verifying every row; the baseline now trains on Kaggle's `validation.csv` (72.9% 3-way, 0.655 macro-F1, 83.7% binary vs 72.9% / 0.654 / 83.9% on the Hugging Face copy, so the copies are equivalent). The n8n Cloud workflows still download from the Hugging Face copy, because n8n Cloud cannot read local files; it is the same corpus.

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

## E7. Issue clustering (2026-09-26)
Input: the 765 reviews Claude (prompt v3) classified negative, out of the 1,560 test and train-sample reviews. Embeddings: `paraphrase-multilingual-MiniLM-L12-v2` (runs locally, free). Clustering is unsupervised, so reusing the stacker's training reviews leaks nothing.

| Text that is embedded | k-means silhouette (k = 6..14) | HDBSCAN | Typical share of one Claude topic in a cluster |
|---|---|---|---|
| Native review text (6 languages) | 0.037 - 0.044 | 0 clusters, 100% noise | 42-86%, mostly 52-72% |
| Claude's short English "issue phrase" (`prompts/issue_v1.txt`) | 0.056 - 0.071 (best k = 13: 0.0706) | 3 clusters, 86% noise | 42-96%, mostly 65-85% |

- Embedding what went wrong (a short English issue phrase) instead of the raw review lifts silhouette by about 1.6x and makes clusters mix languages evenly, which is the goal of cross-language grouping. This is the proposal's "translate-then-analyze vs native" comparison; translate-then-analyze wins for clustering.
- Absolute silhouette is still low (0.07), as expected for short-phrase embeddings of very diverse product reviews. Treat it as a coarse measure; the more useful evidence is that the clusters read clearly (below). HDBSCAN is not usable on this data (mostly noise), so k-means with k = 13 is the choice.
- Cost: issue phrases $0.219 for 765 reviews (my estimate was $0.13; input was larger than assumed), cluster labels $0.008.

Cluster labels written by Claude (k = 13; size, dominant topic share):
delayed or missing shipments and poor support (65, shipping 85%); damaged packaging and product during shipping (56, shipping 70%); wrong item received or non-functional (69); missing parts and assembly defects (57); product fails or breaks shortly after purchase (45, quality 96%); defective or malfunctioning components (49); poor overall quality and durability (79); connectivity and performance issues (72); design flaws and uncomfortable fit (88); wrong size sent or undersized (42); poor content quality and printing errors (56); product does not work as advertised (39); overpriced and rapid price drops (48, price only 42%).
Still to do: a human spot-check of about 100 reviews for cluster and topic precision (proposal metric).

## E8. Trend detection with planted spikes (2026-09-26)
MARC has no review dates, so `src/analysis/trends.py` gives each review a simulated week (uniform over 12 weeks) and plants three spikes by moving extra reviews of a cluster into one week: delayed shipments 5x normal in week 7, damaged packaging 4x in week 9, price complaints 3x in week 10. Detection compares each cluster-week with the trailing 4 weeks (z-score with a Poisson floor on the spread, plus an EWMA control chart). Results are averaged over 200 random timelines.

Chosen setting (z >= 3.5 and at least 8 reviews in the week), measured on the same simulation it was tuned on, so treat as optimistic:

| Rule | Precision | Recall | Precision@3 | False alarms per 12 weeks (13 clusters) |
|---|---|---|---|---|
| z-score | 88.6% | 94.3% | 91.3% | 0.47 (0.53 on timelines with no spike) |
| EWMA | 90.6% | 86.0% | 91.3% | 0.34 |

Threshold sweep for the z-score rule (min 5 reviews): z 3.0 gives recall 97%, precision 77%, 1.07 false alarms; z 4.0 gives recall 90.7%, precision 91.6%, 0.32 false alarms; z 5.0 gives recall 75%, precision 97%. With z 3 the 5x and 4x spikes were always caught and the 3x spike 91% of the time.
Precision@5 (the proposal metric) cannot exceed 0.6 with 3 planted spikes, so Precision@3 is reported. Alerts are weekly here; real alert latency depends on the schedule of the Slack workflow.
Week-over-week deltas are in the detector output (e.g. delayed shipments: 3 reviews the week before, 27 in the spike week).

## Spend so far (approx.)
Translation tests $0.02 + $0.06 (local) + $0.06 (n8n Cloud); sentiment E5 $0.53; E6 $0.65; issue phrases and cluster labels $0.23. Total about $1.56 of $5.

## Next
- Human spot-check of clusters and topics (about 100 reviews).
- Stage 5: response generation (templates by sentiment and issue, guardrails, LLM-judge). Stage 6: dashboard (Google Sheets, Looker Studio), Slack alerts, FastAPI + deployment. Clustering needs a Python service (n8n Cloud cannot run embeddings), so it goes behind the planned FastAPI endpoints.
