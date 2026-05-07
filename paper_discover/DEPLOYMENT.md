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
# Optional persistent embedding cache (M3):
pip install -e '.[vec]'
# Dev tools:
pip install -e '.[dev]'
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

## 6. Daily-digest mode (planned, M5)

Reuses a saved plan and only retrieves papers published since the last run. Not yet wired — track in milestone M5.

---

## 7. Tests

```bash
pytest paper_discover/tests/ -v
```

The suite covers:

- DB schema, WAL, FTS5, views, migrations (test_db.py, test_m2.py)
- Deterministic level-rule mapping including the critical-dimension cap (test_level_rule.py)
- Saturation, skeptic sampling, coverage Wilson CI (test_m2.py)
- Cross-domain analogy routing and decision rules, embedding cache fallback (test_m3.py)

No LLM, network, or GPU is required for the test suite.

---

## 8. Operational notes

- **Resumability**: every stage reads/writes SQLite. If a run crashes, you can re-invoke the same `paper-discover run` and judging will pick up from `judge_status='pending'` candidates. Saturation is idempotent through the `saturation_log` primary key. Skeptic re-queues only flagged papers.
- **Determinism**: with `judge_temperature: 0.0`, the same inputs + same models produce the same bibliography. Audit JSONL captures every LLM call for replay.
- **Cost control**: depth budgets cap candidate counts and LLM call volume (`fast` < `standard` < `deep` < `unlimited`). For unbounded runs, use `--depth deep` and watch the live counts in `paper-discover list`.
- **Privacy (FLAG F11)**: Zotero integration only reads structured fields (DOI, title, authors, abstract, tags, collection). Notes and highlights are never sent to any LLM.

---

## 9. Troubleshooting

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
