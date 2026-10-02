# Customer Review Insights & Response Generator (Group 5)

End-to-end pipeline: aggregate multilingual reviews -> translate -> sentiment/topic analysis ->
issue clustering & trend detection -> personalized response drafts -> dashboard & Slack alerts.

LLM: Claude only (Haiku 4.5 by default; model name is set in `.env`).

## Repo layout

| Path | Purpose |
|---|---|
| `src/ingestion` | Pull/normalise reviews into the unified schema |
| `src/translation` | Language detection + LLM translation |
| `src/analysis` | Sentiment, topics, clustering, trend detection |
| `src/response` | Response-generation templates and guardrails |
| `src/api` | FastAPI service (`/analyze`, `/draft-response`, `/trends`) |
| `prompts/` | Versioned prompt templates |
| `n8n/` | Exported n8n workflow JSON |
| `scripts/` | One-off helpers (dataset download, key checks) |
| `data/` | Local data (`raw/` and `processed/` are git-ignored) |
| `notebooks/`, `tests/`, `docs/` | EDA, tests, report material |

## Data source

Kaggle: `mexwell/amazon-reviews-multi` (Multilingual Amazon Reviews Corpus, Keung et al. 2020): download it,
keep `train.csv`, `validation.csv`, `test.csv` in one folder, and set `KAGGLE_MARC_DIR` in `.env` to that folder.
The sample CSVs in `data/samples/` are checked against those files by `scripts/enrich_from_kaggle.py`.
The n8n workflows download the same corpus from its Hugging Face copy (n8n Cloud cannot read local files);
review ids, stars and text are identical.

## Unified review schema

`review_id, source, product_category, rating, review_text, detected_language, translated_text, review_date`

## Setup (Windows)

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env      # then fill in your keys
python scripts/download_marc_sample.py   # no keys needed
python scripts/check_keys.py             # tests the Claude key
```

## n8n workflows (`n8n/`)

| File | What it does |
|---|---|
| `01_ingestion_marc_to_unified_schema.json` | Fetches the MARC review files from Hugging Face, samples evenly per star rating, outputs the unified schema. No credentials needed. |
| `02_ingestion_and_translation.json` | Workflow 01 plus Claude translation of non-English reviews (batches of 8, retries, failures are recorded per review instead of stopping the run). |

| `03_full_pipeline.json` | The full pipeline: 02 plus sentiment/topic, issue category, reply drafts, status and a summary with accuracy, cost and timing. Needs the Claude credential on 4 HTTP nodes (Claude translate / classify / issue category / draft reply). Default about $0.30; `perStar` in `Config` scales it (40 = about $3). |
| `04_trend_detection_slack_alert.json` | Reads the dashboard sheet's Reviews tab (sheet shared as "anyone with the link can view"), finds negative-review spikes per issue category (z >= 3.5, >= 8 reviews, trailing 4 weeks) and posts a summary to Slack. Paste your Slack Incoming Webhook URL into the **Post to Slack** node. No Claude cost. |

Product categories: workflows 01 to 03 fill `product_category` from `data/samples/marc_test_categories.csv`, fetched from this repo's public GitHub raw URL (the Hugging Face copy has none). If that file is not on GitHub yet, or the fetch fails, the run still works and the category is null.

Import: n8n > Workflows > Create workflow > `...` menu > **Import from file**, then **Execute workflow**.

Claude credential (workflows 02 and 03): n8n > Credentials > Create > **Header Auth**, Name `x-api-key`,
Value = your Anthropic API key. Then open each **Claude ...** node (1 in workflow 02, 4 in workflow 03) and pick that credential.

Cost control: the `Config` node sets `perStar` (default 4, about 100 translations, roughly $0.06).
Set it to 40 for the full 1,200-review sample (roughly $0.55).

### n8n Cloud deployment (current account)

The workflows run on the team's n8n Cloud account `https://capstoneproject-group5.app.n8n.cloud/` (free trial, valid to about 16 Oct 2026; an earlier account was retired on 2 Oct). To set up a fresh account:

1. Import the four JSON files above (a new workflow each, **Import from file** or **Import from URL** with the GitHub raw link). Importing into an already-open workflow pastes the nodes next to the old ones, so import into an empty workflow.
2. Create the **Header Auth** credential (above) and select it on the 5 Claude nodes in workflows 02 and 03. Imported workflows never carry credentials.
3. Workflow 04: paste your Slack Incoming Webhook URL into **Post to Slack**, press Ctrl+S, and make the dashboard Google Sheet readable by "anyone with the link".
4. Run **03** (about $0.30; expect 120 reviews, all with a `product_category`) and **04** (expect 4 spikes and `slack_status: sent`).

Last verified on the current account on 2 Oct 2026: workflow 03 finished in 83 s for $0.30 with 78.3% 3-way and 88.5% positive-vs-negative accuracy; workflow 04 posted the 4 spikes to `#all-capstone-project`. No keys or webhook URLs are stored in this repository.


Stage 4 (issue clustering and trend detection): `python scripts/run_stage4.py issues`, then `cluster issue`, `label issue`, `trends issue`.

Stage 5 (response generation): `python scripts/eval_responses.py` (env `RESPONSE_VERSION=v1|v2`, default v1; v2 is the better prompt), the human review step (`data/samples/human_review_sheet_v2.csv`, `scripts/score_human_review.py`) is optional and was skipped; acceptance is reported from the LLM judge.

## Running the application

API service (needs `ANTHROPIC_API_KEY` in `.env`):

```bash
set SERVICE_API_KEY=choose-a-key   # optional; callers then send it as X-API-Key
set MAX_SPEND_USD=1.0              # hard cap on Claude spend for this service
uvicorn src.api.main:app --port 8000
# interactive docs: http://localhost:8000/docs
pytest tests -q                    # 11 tests, no Claude calls
```

Endpoints: `GET /health`, `POST /analyze`, `POST /draft-response`, `POST /trends`. Every Claude call is logged to `data/processed/llm_calls.jsonl`.

## Final report and presentation

`docs/final/` holds the Final Project Report (`.docx` and `.pdf`) and the final presentation (`.pptx`). The experiment log behind every number is `docs/experiment_log.md`.

## Demo app (Gradio) and deployment (Hugging Face Space)

`app.py` is a three-tab Gradio app that reuses the code in `src/`: saved examples (13 real reviews with their stored output, no Claude call), your own review (live Claude: classify, issue, drafted reply, optional Sonnet 5 judge, spend-capped) and spike detection (the workflow 04 detector on the dashboard data, no Claude call).

Run it locally (needs `pip install gradio`; the live tab needs `ANTHROPIC_API_KEY` in `.env`):

```bash
python app.py        # http://127.0.0.1:7860
pytest tests -q      # 16 tests, none call Claude
```

Live-tab limits: `MAX_SPEND_USD` (default 0.5 in the app), `DEMO_MAX_CALLS` per browser session (default 10), optional `DEMO_ACCESS_CODE`. The spend total is kept in a log file on the server, so it restarts from zero when a Space restarts; also set a spend limit in the Anthropic Console.

Live demo: https://huggingface.co/spaces/SrinivasanHF/review-insights-demo (direct app: https://srinivasanhf-review-insights-demo.hf.space). It runs on the free ZeroGPU tier; `app.py` contains a placeholder `@spaces.GPU` function because ZeroGPU Spaces refuse to start without one (the app itself uses no GPU).

Deploy to a Hugging Face Space (free CPU):

1. `python scripts/build_space.py` builds `space_build/` with only the files the demo needs (about 200 KB, no keys).
2. On huggingface.co create a new Space: SDK **Gradio**, hardware **CPU basic (free)**.
3. Space settings, Variables and secrets: add the secret `ANTHROPIC_API_KEY` (type it yourself; never commit it). Optional: `DEMO_ACCESS_CODE`, `MAX_SPEND_USD`.
4. Upload the contents of `space_build/`: either `python scripts/deploy_space.py` (needs `HF_TOKEN` with write access in `.env`; it uploads the folder with its structure) or, in a browser, Files, Add file, Upload files (one folder at a time).
5. Wait for the build, open the Space link, and try all three tabs.

