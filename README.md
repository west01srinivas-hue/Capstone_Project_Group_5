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

Import: n8n > Workflows > Create workflow > `...` menu > **Import from file**, then **Execute workflow**.

Claude credential (workflow 02 only): n8n > Credentials > Create > **Header Auth**, Name `x-api-key`,
Value = your Anthropic API key. Then open the **Claude translate** node and pick that credential.

Cost control: the `Config` node sets `perStar` (default 4, about 100 translations, roughly $0.06).
Set it to 40 for the full 1,200-review sample (roughly $0.55).

Stage 4 (issue clustering and trend detection): `python scripts/run_stage4.py issues`, then `cluster issue`, `label issue`, `trends issue`.

## Running the application

_To be completed as the pipeline is built (FastAPI/Streamlit launch commands)._
