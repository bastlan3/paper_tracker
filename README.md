# EBM Paper Tracker

Track research papers posted on arxiv with two-stage filtering based on the papers you choose, LLM summaries, and an interactive UMAP visualization.

## Quick Start

```bash
cd ebm-tracker
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Edit .env with your API keys (see below)

python run.py
# Open http://localhost:8000
```

## Configuration (.env)

**Required** — LLM provider for summaries (default: Mistral, free):
- `MISTRAL_API_KEY` — sign up at [console.mistral.ai](https://console.mistral.ai), free Experiment plan (phone verification only, no credit card). Uses `mistral-small-latest` (Apache 2.0, open-source).
- Alternatives: set `LLM_PROVIDER` to `anthropic` or `openai` with the corresponding API key.

**Optional** — notifications:
- **Telegram**: create a bot via @BotFather, get your chat ID via @userinfobot. Set `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID`.
- **Email**: use a Gmail app password (not your real password). Set `SMTP_USER`, `SMTP_PASSWORD`, `EMAIL_TO`.

**Tuning**:
- `SIMILARITY_THRESHOLD` (default 0.45): higher = stricter filtering, lower = more papers
- `DAILY_DIGEST_HOUR` (default 9): hour (UTC) for daily notifications
- Seed papers in `app/config.py` — add/remove arxiv IDs to shift what counts as "relevant"

## Architecture

- **Stage 1**: keyword search on arxiv (cs.LG, cs.AI, stat.ML)
- **Stage 2**: cosine similarity of Specter2 embeddings against seed paper set
- **Summaries**: LLM-generated 2-3 sentence summary + keyword tags
- **Viz**: UMAP 2D projection of paper embeddings, interactive Plotly scatter

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/api/papers` | GET | List papers (query params: `search`, `source`, `limit`, `offset`) |
| `/api/papers/add` | POST | Add paper by arxiv ID `{"arxiv_id": "2301.12345"}` |
| `/api/viz` | GET | UMAP visualization data |
| `/api/ingest` | POST | Trigger fetch + filter + summarize pipeline |
| `/api/umap/recompute` | POST | Recompute UMAP projection |
| `/api/stats` | GET | Paper counts |

## Keyboard Shortcuts

- `Ctrl+K` / `Cmd+K` — focus search
- `Escape` — close add paper form
- Click a paper card to highlight it on the UMAP scatter
- Click a dot on the scatter to open the paper
