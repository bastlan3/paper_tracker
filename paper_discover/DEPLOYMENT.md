# paper-discover — Deployment Guide

A multi-agent literature-discovery tool with calibrated coverage. This guide walks through running it on your own hardware end to end.

If anything below diverges from `pyproject.toml` or `paper_discover/config/models.yaml`, those files are the source of truth.

---

## 1. Prerequisites

| Component | Why | Notes |
|---|---|---|
| Python ≥ 3.11 | runtime | `python --version` |
| GPU with ≥ 12 GB VRAM | local LLM judging at usable speed | CPU works for tests but is impractical for a real run |
| Disk: ~10 GB free | models + per-run SQLite files | runs grow ~50 MB / 5 k papers |
| `sqlite3` (system binary) | debugging the run DB | already on most Linux/macOS |
| Network access | OpenAlex, Semantic Scholar, Crossref, Unpaywall | all free / open APIs |

**Optional services** (each is degraded-gracefully — pipeline runs without them):

- **Zotero** desktop with local API enabled (port 23119) — for read-only library dedup
- **Semantic Scholar API key** — bumps rate limits from 1/sec → 100/sec
- **Polite-pool email** for OpenAlex — bumps to 10/sec
- **Unpaywall email** — required by the Unpaywall API (free, just identifies you)

---

## 2. Install

```bash
git clone <repo-url> paper_tracker
cd paper_tracker

python -m venv .venv
source .venv/bin/activate

pip install -e .
# Optional extras (combine as needed):
pip install -e '.[vec]'   # M3 — persistent LanceDB embedding cache
pip install -e '.[web]'   # M8 — FastAPI dashboard + JSON API
pip install -e '.[mcp]'   # M9 — MCP stdio server
pip install -e '.[dev]'   # pytest tooling
```

Verify:

```bash
paper-discover --help
pytest paper_discover/tests/ -q     # all tests should pass
```

---

## 3. Local LLM serving (vLLM)

The pipeline needs at least the **main judge** model. The **skeptic** (Stage 5) is optional but improves coverage calibration.

### 3a. Main judge — vLLM on port 8000

```bash
pip install vllm
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-14B-Instruct \
  --port 8000 \
  --max-model-len 8192 \
  --guided-decoding-backend xgrammar
```

Test it:

```bash
curl http://localhost:8000/v1/models
```

### 3b. Skeptic — different model family on port 8001 (optional)

The skeptic should NOT share a base model with the judge — correlated biases defeat the point (FLAG F7). Pick from a different family:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3.3-8B-Instruct \
  --port 8001 \
  --max-model-len 8192 \
  --guided-decoding-backend xgrammar
```

Skip this and Stage 5 will short-circuit with a logged warning, and Stage 7 will note the missing signal in the coverage report.

### 3c. Embeddings + reranker

Run locally via `sentence-transformers` (default). First-time use downloads:

- `BAAI/bge-m3` (~2.3 GB) for embeddings
- `BAAI/bge-reranker-v2-m3` (~2.3 GB) for the T2 cross-encoder

No extra server is needed. To use a remote Infinity server, edit `paper_discover/config/models.yaml` and set `embedding.use_server: true`.

---

## 4. Configuration

### 4a. Environment variables

Put these in your shell profile, a `.env` file you `source`, or a `direnv` config — paper-discover reads them at runtime.

```bash
# vLLM endpoints
export PAPER_DISCOVER_VLLM_URL="http://localhost:8000/v1"
export PAPER_DISCOVER_SKEPTIC_URL="http://localhost:8001/v1"

# Model registry override (otherwise uses paper_discover/config/models.yaml)
# export PAPER_DISCOVER_MODELS_CONFIG="/abs/path/to/your/models.yaml"

# API politeness
export OPENALEX_EMAIL="you@example.com"
export S2_API_KEY="sk-..."           # optional but recommended
export UNPAYWALL_EMAIL="you@example.com"  # required by Unpaywall

# Zotero (optional — read-only library access)
export ZOTERO_LOCAL_PORT=23119
# Or use the cloud API:
# export ZOTERO_API_KEY="..."
# export ZOTERO_LIBRARY_ID="123456"
# export ZOTERO_LIBRARY_TYPE="user"   # or "group"

# Cloud LLM fallback (used only when local vLLM is unreachable)
# export DEEPSEEK_API_KEY="sk-..."
```

### 4b. `paper_discover/config/models.yaml`

The defaults work for the setup above. Edit if your endpoints, model names, or thresholds differ. Notable knobs:

- `local.judge_model` — must match a model loaded in your vLLM instance
- `local.judge_temperature: 0.0` — deterministic judging; only raise this if you understand the consequences
- `reranker.threshold: 0.3` — T2 cut-off; raise to be more aggressive (cuts more before LLM step)
- `skeptic.base_url` — separate vLLM port for the adversarial pass
- `cloud_fallback.enabled` — flip to `true` to allow DeepSeek/etc. when local is down

---

## 5. First run

### 5a. Plan mode (interactive — approves the search plan)

```bash
paper-discover plan \
  --intent "Do GLP-1 receptor agonists reduce CV mortality in CKD patients?" \
  --anchor 10.1056/NEJMoa2024816 \
  --anchor 10.1016/S0140-6736(20)30831-2 \
  --depth deep \
  --out runs/glp1_ckd
```

The planner LLM proposes intent, dimensions, queries, and may ask up to ~5 clarification questions. You **must approve** the plan before retrieval starts. The approved plan is saved to `runs/glp1_ckd/plan.json`.

### 5b. Deep run (full pipeline)

```bash
paper-discover run runs/glp1_ckd
```

This runs the LangGraph pipeline:

```
START → retrieval → judging → saturation → skeptic → coverage → report → END
```

Progress is logged to stderr; the SQLite database lives at `runs/glp1_ckd/run.db`. Inspect it any time:

```bash
sqlite3 runs/glp1_ckd/run.db "SELECT level, COUNT(*) FROM candidates GROUP BY level"
```

### 5c. Outputs

After the run, `runs/glp1_ckd/` contains:

- `plan.json` — the approved Stage 0 plan
- `run.db` — full audit trail (papers, candidates, queries, citations, signals, judge outputs, skeptic flags, anchor probes, coverage signals)
- `bibliography.md` — ranked annotated, grouped CORE → SUPPORTING → CONTEXT → ADJACENT
- `bibliography.bib` — BibTeX
- `bibliography.csl.json` — CSL JSON
- `summary.md` — run stats, coverage estimate with CI, flags
- `audit.jsonl` — replayable JSONL log

### 5d. List and inspect runs

```bash
paper-discover list
```

---

## 6. Daily-digest mode

Reuses a saved plan and only retrieves papers published since the last run.

```bash
# 1. Save an approved plan as a recurring search
paper-discover save runs/glp1_ckd/plan.json --name "GLP-1 / CKD daily" --cadence daily --db digest.db

# 2. List saved searches
paper-discover searches --db digest.db

# 3. Run all enabled searches (or one by ID)
paper-discover digest --db digest.db
paper-discover digest 01HXX...  --db digest.db
```

Each digest run skips Stages 4/5/6/7 (saturation, skeptic, hygiene, coverage) — it's just retrieval + judging restricted to publications since `last_run_at`. Outputs land under `digest_runs/<search_id>/<run_id>.md`.

To wire it to a scheduler:

```bash
# crontab -e
0 7 * * * /path/to/.venv/bin/paper-discover digest --db /path/to/digest.db
```

Delete a saved search:

```bash
paper-discover forget 01HXX... --db digest.db
```

---

## 7. Web dashboard (optional, [web] extra)

```bash
pip install -e '.[web]'
paper-discover serve --db runs/glp1_ckd/run.db --host 127.0.0.1 --port 8500
```

Open `http://127.0.0.1:8500/` for the run-history dashboard. JSON API:

| Endpoint | Returns |
|---|---|
| `GET /api/runs` | run list |
| `GET /api/runs/{id}` | run + coverage_signals |
| `GET /api/runs/{id}/papers?level=CORE` | bibliography |
| `GET /api/runs/{id}/concept_map` | citation graph |
| `GET /api/runs/{id}/prisma` | funnel counts |
| `GET /api/saved_searches` | digest searches |
| `POST /api/saved_searches` | create search |
| `DELETE /api/saved_searches/{id}` | delete |
| `GET /healthz` | liveness |

The dashboard runs against a single DB; for multi-tenant use, run one process per DB or front them with nginx.

---

## 8. MCP server (optional, [mcp] extra)

Lets other agents pull bibliographies, coverage estimates, and concept maps directly.

```bash
pip install -e '.[mcp]'
paper-discover mcp --db runs/glp1_ckd/run.db
```

Speaks MCP over stdio. Exposed tools (read-side only — pipeline runs stay on the CLI/scheduler):

`list_runs`, `get_run`, `get_bibliography`, `get_concept_map`, `get_prisma`, `list_saved_searches`, `save_plan`, `plan_from_pico`.

Wire it into Claude Desktop or any MCP client by pointing at the `paper-discover mcp` command in their config.

---

## 9. Seed modes (CLI plan)

`paper-discover plan` accepts any combination of:

| Flag | Meaning |
|---|---|
| `--seed "..."` | natural-language question |
| `--anchors a,b,c` | DOIs / arXiv IDs / Zotero keys / OpenAlex W-ids / S2 hex IDs |
| `--collection KEY` | Zotero collection key (theme inferred) |
| `--structured PATH` | PICO / boolean / filter JSON (skips planner LLM) |

Structured query examples (save as `query.json`, then `paper-discover plan --structured query.json`):

```json
{
  "format": "pico",
  "population": "adults with CKD",
  "intervention": "GLP-1 receptor agonists",
  "comparison": "placebo",
  "outcome": "cardiovascular mortality",
  "depth": "deep"
}
```

```json
{
  "format": "boolean",
  "intent": "long-COVID neurological symptoms",
  "boolean": "\"long covid\" AND (neurological OR cognitive) NOT pediatric"
}
```

```json
{
  "format": "filter",
  "concepts": ["diabetes", "metformin"],
  "scope": {"date_from": "2020-01-01", "languages": ["en"]}
}
```

---

## 10. Tests

```bash
pytest paper_discover/tests/ -q
# 173 passed
```

Coverage by milestone:

| File | What it tests |
|---|---|
| `test_db.py` | schema, WAL, FTS5, views, upserts, citation neighborhood |
| `test_level_rule.py` | deterministic level mapping + critical-dimension cap |
| `test_m2.py` | migration idempotency, Wilson CI, channel Jaccard, stratified skeptic sample |
| `test_m3.py` | cross-domain channel routing + decision rules, embedding cache fallback |
| `test_m4.py` | retraction parsing, errata, Unpaywall + Europe PMC, persist idempotency |
| `test_m5.py` | saved-search CRUD, incremental plan adaptation, digest rendering |
| `test_m6.py` | anchor classification, PICO/boolean/filter → plan |
| `test_m7.py` | concept map, PRISMA counts, gap-list rendering |
| `test_m8.py` | FastAPI endpoints (skipped if fastapi not installed) |
| `test_m9.py` | MCP tool handlers + dispatcher |

No LLM, network, or GPU required.

---

## 11. Operational notes

- **Resumability**: every stage reads/writes SQLite. If a run crashes, you can re-invoke the same `paper-discover run` and judging will pick up from `judge_status='pending'` candidates. Saturation is idempotent through the `saturation_log` primary key. Skeptic re-queues only flagged papers.
- **Determinism**: with `judge_temperature: 0.0`, the same inputs + same models produce the same bibliography. Audit JSONL captures every LLM call for replay.
- **Cost control**: depth budgets cap candidate counts and LLM call volume (`fast` < `standard` < `deep` < `unlimited`). For unbounded runs, use `--depth deep` and watch the live counts in `paper-discover list`.
- **Privacy (FLAG F11)**: Zotero integration only reads structured fields (DOI, title, authors, abstract, tags, collection). Notes and highlights are never sent to any LLM.

---

## 12. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `ConnectionError: connection refused on :8000` | vLLM not running | `python -m vllm.entrypoints.openai.api_server …` |
| `Stage 5 skipped: skeptic model unreachable` | port 8001 not running | start a different-family vLLM there, or accept the missing signal |
| `judge_parse_error` rows in DB | LLM returned non-JSON despite guided decoding | re-run with a stricter system message; one retry is automatic |
| OpenAlex 429 | rate-limited | set `OPENALEX_EMAIL` for the polite pool |
| S2 API gives 1 req/sec only | no API key | request a free key at semanticscholar.org/product/api |
| Embeddings recomputed every run | LanceDB not installed | `pip install -e '.[vec]'` and pass a path to `EmbeddingCache.open()` |
| `paper-discover plan` hangs | planner LLM too slow / no GPU | switch `planner_model` to a smaller model or use cloud fallback |

When in doubt: open the run DB directly with `sqlite3 runs/<run_id>/run.db` — every decision is auditable.
