-- paper-discover SQLite schema
-- Applied once via candidates/db.py on first open.
-- WAL + synchronous=NORMAL set at connection time (not here).

-- ── Global paper cache ───────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS papers (
  paper_id        TEXT PRIMARY KEY,
  doi             TEXT UNIQUE,
  openalex_id     TEXT UNIQUE,
  s2_id           TEXT UNIQUE,
  arxiv_id        TEXT UNIQUE,
  pmid            TEXT UNIQUE,
  title           TEXT NOT NULL,
  title_norm      TEXT NOT NULL,
  authors_json    TEXT,
  first_author    TEXT,
  year            INTEGER,
  venue           TEXT,
  abstract        TEXT,
  oa_url          TEXT,
  is_preprint     INTEGER NOT NULL DEFAULT 0,
  retracted       INTEGER NOT NULL DEFAULT 0,
  retraction_notice_json TEXT,
  fetched_at      TEXT,
  metadata_source TEXT
);

CREATE INDEX IF NOT EXISTS idx_papers_year   ON papers(year);
CREATE INDEX IF NOT EXISTS idx_papers_doi    ON papers(doi);
CREATE INDEX IF NOT EXISTS idx_papers_titlen ON papers(title_norm);
CREATE INDEX IF NOT EXISTS idx_papers_oa     ON papers(openalex_id);
CREATE INDEX IF NOT EXISTS idx_papers_s2     ON papers(s2_id);

-- FTS5 mirror for lexical retrieval
CREATE VIRTUAL TABLE IF NOT EXISTS papers_fts USING fts5(
  paper_id UNINDEXED,
  title,
  abstract,
  content='papers',
  content_rowid='rowid',
  tokenize='porter ascii'
);

CREATE TRIGGER IF NOT EXISTS papers_fts_ins AFTER INSERT ON papers BEGIN
  INSERT INTO papers_fts(rowid, paper_id, title, abstract)
  VALUES (new.rowid, new.paper_id, new.title, COALESCE(new.abstract, ''));
END;

CREATE TRIGGER IF NOT EXISTS papers_fts_del BEFORE DELETE ON papers BEGIN
  DELETE FROM papers_fts WHERE rowid = old.rowid;
END;

CREATE TRIGGER IF NOT EXISTS papers_fts_upd AFTER UPDATE ON papers BEGIN
  DELETE FROM papers_fts WHERE rowid = old.rowid;
  INSERT INTO papers_fts(rowid, paper_id, title, abstract)
  VALUES (new.rowid, new.paper_id, new.title, COALESCE(new.abstract, ''));
END;

-- ── Citation graph ───────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS citations (
  src_paper_id TEXT NOT NULL,
  dst_paper_id TEXT NOT NULL,
  PRIMARY KEY (src_paper_id, dst_paper_id),
  FOREIGN KEY (src_paper_id) REFERENCES papers(paper_id),
  FOREIGN KEY (dst_paper_id) REFERENCES papers(paper_id)
);

CREATE INDEX IF NOT EXISTS idx_cite_src ON citations(src_paper_id);
CREATE INDEX IF NOT EXISTS idx_cite_dst ON citations(dst_paper_id);

-- ── Runs ─────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS runs (
  run_id          TEXT PRIMARY KEY,
  started_at      TEXT NOT NULL,
  finished_at     TEXT,
  mode            TEXT NOT NULL CHECK(mode IN ('plan','deep','digest')),
  plan_json       TEXT NOT NULL,
  status          TEXT NOT NULL DEFAULT 'running'
                       CHECK(status IN ('running','done','failed')),
  coverage_p      REAL,
  coverage_ci_lo  REAL,
  coverage_ci_hi  REAL,
  notes           TEXT
);

-- ── Anchors per run ──────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS run_anchors (
  run_id     TEXT NOT NULL,
  paper_id   TEXT NOT NULL,
  PRIMARY KEY (run_id, paper_id),
  FOREIGN KEY (run_id)   REFERENCES runs(run_id),
  FOREIGN KEY (paper_id) REFERENCES papers(paper_id)
);

-- ── Queries issued ───────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS queries (
  query_id      TEXT PRIMARY KEY,
  run_id        TEXT NOT NULL,
  family        TEXT NOT NULL
                     CHECK(family IN ('lexical','semantic','concept_translation',
                                      'citation','cocite','zotero')),
  source        TEXT NOT NULL,
  query_text    TEXT NOT NULL,
  dimensions_json TEXT,
  issued_at     TEXT NOT NULL,
  result_count  INTEGER,
  status        TEXT NOT NULL DEFAULT 'ok'
                     CHECK(status IN ('ok','rate_limited','error')),
  FOREIGN KEY (run_id) REFERENCES runs(run_id)
);

CREATE INDEX IF NOT EXISTS idx_queries_run ON queries(run_id);

-- ── Candidates ───────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS candidates (
  run_id               TEXT NOT NULL,
  paper_id             TEXT NOT NULL,
  first_seen_at        TEXT NOT NULL,
  last_seen_at         TEXT NOT NULL,
  seen_count           INTEGER NOT NULL DEFAULT 1,
  seen_by_json         TEXT NOT NULL,
  hop_distance_to_anchor INTEGER,
  signals_json         TEXT,
  judge_status         TEXT NOT NULL DEFAULT 'pending'
                            CHECK(judge_status IN
                              ('pending','judged','cut','flagged','parse_error')),
  judge_tier           TEXT CHECK(judge_tier IN ('T1','T2','T3','T4','T5')),
  level                TEXT CHECK(level IN
                         ('CORE','SUPPORTING','CONTEXT','ADJACENT','CUT')),
  judge_score_json     TEXT,
  judge_confidence     REAL,
  judged_by            TEXT,
  judged_at            TEXT,
  flags_json           TEXT,
  evidence_span        TEXT,
  PRIMARY KEY (run_id, paper_id),
  FOREIGN KEY (run_id)   REFERENCES runs(run_id),
  FOREIGN KEY (paper_id) REFERENCES papers(paper_id)
);

CREATE INDEX IF NOT EXISTS idx_cand_status ON candidates(run_id, judge_status);
CREATE INDEX IF NOT EXISTS idx_cand_level  ON candidates(run_id, level);
CREATE INDEX IF NOT EXISTS idx_cand_seen   ON candidates(run_id, seen_count DESC);

-- ── Candidate × Query source map ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS candidate_sources (
  run_id    TEXT NOT NULL,
  paper_id  TEXT NOT NULL,
  query_id  TEXT NOT NULL,
  rank      INTEGER,
  PRIMARY KEY (run_id, paper_id, query_id),
  FOREIGN KEY (run_id, paper_id) REFERENCES candidates(run_id, paper_id),
  FOREIGN KEY (query_id)         REFERENCES queries(query_id)
);

-- ── Embeddings ───────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS embeddings (
  paper_id    TEXT NOT NULL,
  model       TEXT NOT NULL,
  vector      BLOB NOT NULL,
  dim         INTEGER NOT NULL,
  computed_at TEXT NOT NULL,
  PRIMARY KEY (paper_id, model),
  FOREIGN KEY (paper_id) REFERENCES papers(paper_id)
);

-- ── Skeptic flags ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS skeptic_flags (
  run_id          TEXT NOT NULL,
  paper_id        TEXT NOT NULL,
  flagged_at      TEXT NOT NULL,
  skeptic_model   TEXT NOT NULL,
  skeptic_reason  TEXT,
  resolution      TEXT CHECK(resolution IN ('overturned','sustained','pending')),
  PRIMARY KEY (run_id, paper_id),
  FOREIGN KEY (run_id, paper_id) REFERENCES candidates(run_id, paper_id)
);

-- ── Anchor-injection calibration probes ──────────────────────────────────────
CREATE TABLE IF NOT EXISTS anchor_probes (
  run_id         TEXT NOT NULL,
  paper_id       TEXT NOT NULL,
  expected_level TEXT NOT NULL,
  actual_level   TEXT,
  passed         INTEGER,
  PRIMARY KEY (run_id, paper_id),
  FOREIGN KEY (run_id, paper_id) REFERENCES candidates(run_id, paper_id)
);

-- ── Saved searches (digest mode) ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS saved_searches (
  search_id  TEXT PRIMARY KEY,
  name       TEXT NOT NULL,
  plan_json  TEXT NOT NULL,
  created_at TEXT NOT NULL,
  last_run_at TEXT,
  cadence    TEXT NOT NULL DEFAULT 'daily',
  enabled    INTEGER NOT NULL DEFAULT 1
);

-- ── Zotero snapshot (per run, for dedup) ─────────────────────────────────────
CREATE TABLE IF NOT EXISTS zotero_items (
  run_id     TEXT NOT NULL,
  zotero_key TEXT NOT NULL,
  paper_id   TEXT,
  collection TEXT,
  tags_json  TEXT,
  PRIMARY KEY (run_id, zotero_key),
  FOREIGN KEY (run_id) REFERENCES runs(run_id)
);

-- ── Views ────────────────────────────────────────────────────────────────────
CREATE VIEW IF NOT EXISTS v_bibliography AS
SELECT
  c.run_id, p.paper_id, p.title, p.authors_json, p.year, p.venue, p.doi,
  p.oa_url, p.is_preprint, p.retracted,
  c.level, c.judge_confidence, c.evidence_span,
  c.flags_json, c.seen_count, c.seen_by_json
FROM candidates c
JOIN papers p USING (paper_id)
WHERE c.level IS NOT NULL
  AND c.level != 'CUT'
ORDER BY
  CASE c.level
    WHEN 'CORE'       THEN 1
    WHEN 'SUPPORTING' THEN 2
    WHEN 'CONTEXT'    THEN 3
    WHEN 'ADJACENT'   THEN 4
  END,
  c.judge_confidence DESC,
  p.year DESC;

CREATE VIEW IF NOT EXISTS v_discovery_curve AS
SELECT
  run_id,
  DATE(judged_at) AS judged_date,
  COUNT(*) FILTER (WHERE level NOT IN ('CUT') AND level IS NOT NULL) AS cumulative_keeps
FROM candidates
WHERE judged_at IS NOT NULL
GROUP BY run_id, DATE(judged_at)
ORDER BY run_id, judged_date;
