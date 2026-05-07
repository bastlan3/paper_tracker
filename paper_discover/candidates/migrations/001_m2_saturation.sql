-- M2 migration: tables needed for saturation, skeptic, and coverage stages.
-- Applied idempotently on DB open (CREATE TABLE IF NOT EXISTS).

-- Per-iteration saturation tracking (drives the discovery curve).
CREATE TABLE IF NOT EXISTS saturation_log (
  run_id          TEXT NOT NULL,
  iteration       INTEGER NOT NULL,
  new_candidates  INTEGER NOT NULL,   -- papers added as candidates this iteration
  new_keeps       INTEGER NOT NULL,   -- papers promoted above CUT this iteration
  total_keeps     INTEGER NOT NULL,   -- cumulative keeps after this iteration
  converged       INTEGER NOT NULL DEFAULT 0,  -- 1 if this iteration triggered stop
  logged_at       TEXT NOT NULL,
  PRIMARY KEY (run_id, iteration),
  FOREIGN KEY (run_id) REFERENCES runs(run_id)
);

-- Track which papers the skeptic was given and what it decided.
-- (skeptic_flags already exists in schema.sql; this is additive.)

-- Per-run coverage signal breakdown for auditability.
CREATE TABLE IF NOT EXISTS coverage_signals (
  run_id               TEXT PRIMARY KEY,
  saturation_signal    REAL,   -- 0–1; from discovery curve flatness
  skeptic_signal       REAL,   -- 0–1; 1 - skeptic overturn rate (NULL if no skeptic)
  channel_jaccard      REAL,   -- 0–1; avg pairwise Jaccard of kept-paper channel sets
  anchor_accuracy      REAL,   -- 0–1; fraction anchors correctly CORE/SUPPORTING
  coverage_p           REAL,   -- weighted aggregate
  coverage_ci_lo       REAL,
  coverage_ci_hi       REAL,
  methodology_json     TEXT,   -- full input values for the audit report
  computed_at          TEXT,
  FOREIGN KEY (run_id) REFERENCES runs(run_id)
);

-- saturation_iteration column on candidates (records which sat. iteration added this paper).
-- ALTER TABLE IF NOT EXISTS column syntax not available in SQLite < 3.37;
-- use a safe conditional approach.
-- We skip the ALTER if the column already exists (the pragma trick).
