"""SQLite storage: raw articles -> extracted deals -> investors per deal."""
import json
import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS articles (
    id            TEXT PRIMARY KEY,          -- "<source>:<native id or url>"
    source        TEXT NOT NULL,             -- egirisim | webrazzi
    url           TEXT NOT NULL UNIQUE,
    title         TEXT NOT NULL,
    published_at  TEXT NOT NULL,             -- ISO 8601
    body          TEXT NOT NULL,             -- plain text
    fetched_at    TEXT NOT NULL DEFAULT (datetime('now')),
    extracted     INTEGER NOT NULL DEFAULT 0 -- 0 = not processed by LLM yet
);

CREATE TABLE IF NOT EXISTS deals (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    article_id      TEXT NOT NULL REFERENCES articles(id),  -- one article can report several deals
    company         TEXT NOT NULL,
    description     TEXT,
    sector          TEXT,
    hq_country      TEXT,
    turkey_link     TEXT,   -- hq_turkey | turkish_founders | turkish_investor | none
    stage           TEXT,   -- pre_seed | seed | series_a | ... | unknown
    amount          REAL,   -- in `currency`, NULL if undisclosed
    currency        TEXT,
    amount_usd      REAL,   -- rough conversion, NULL if undisclosed
    valuation_usd   REAL,
    announced_date  TEXT,
    confidence      REAL,   -- model's own 0-1 confidence
    raw_json        TEXT NOT NULL,
    model           TEXT NOT NULL,
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE (article_id, company)
);

CREATE TABLE IF NOT EXISTS deal_investors (
    deal_id   INTEGER NOT NULL REFERENCES deals(id) ON DELETE CASCADE,
    investor  TEXT NOT NULL,   -- name as written in the article
    role      TEXT NOT NULL,   -- lead | participant
    PRIMARY KEY (deal_id, investor)
);

-- Model comparison: a fixed random test set, and every model's raw answer per article.
-- Kept apart from `deals` so eval runs never touch the main dataset.
CREATE TABLE IF NOT EXISTS eval_set (
    article_id  TEXT PRIMARY KEY REFERENCES articles(id),
    position    INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS model_runs (
    article_id  TEXT,
    model       TEXT,
    deal_json   TEXT,   -- one deal as JSON; NULL = model said "no deal"; {"error": ...} on failure
    created_at  TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_articles_extracted ON articles(extracted);
CREATE INDEX IF NOT EXISTS idx_deal_investors_investor ON deal_investors(investor);
"""


def connect(path: str | Path = "data/dealflow.db") -> sqlite3.Connection:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    return conn


def upsert_article(conn, a: dict) -> bool:
    """Insert an article; returns True if it was new."""
    cur = conn.execute(
        """INSERT OR IGNORE INTO articles (id, source, url, title, published_at, body)
           VALUES (:id, :source, :url, :title, :published_at, :body)""",
        a,
    )
    return cur.rowcount == 1


def pending_articles(conn, limit: int | None = None):
    sql = "SELECT * FROM articles WHERE extracted = 0 ORDER BY published_at DESC"
    if limit:
        sql += f" LIMIT {int(limit)}"
    return conn.execute(sql).fetchall()


def mark_extracted(conn, article_id: str, status: int = 1):
    """status: 1 = funding deal saved, 2 = not a funding round, 3 = skipped by prefilter, -1 = error"""
    conn.execute("UPDATE articles SET extracted = ? WHERE id = ?", (status, article_id))


def save_deal(conn, article_id: str, d: dict, model: str) -> int:
    cur = conn.execute(
        """INSERT OR REPLACE INTO deals
           (article_id, company, description, sector, hq_country, turkey_link, stage,
            amount, currency, amount_usd, valuation_usd, announced_date, confidence, raw_json, model)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            article_id, d["company"], d.get("description"), d.get("sector"), d.get("hq_country"),
            d.get("turkey_link"), d.get("stage"), d.get("amount"), d.get("currency"),
            d.get("amount_usd"), d.get("valuation_usd"), d.get("announced_date"),
            d.get("confidence"), json.dumps(d, ensure_ascii=False), model,
        ),
    )
    deal_id = cur.lastrowid
    conn.execute("DELETE FROM deal_investors WHERE deal_id = ?", (deal_id,))
    for name in d.get("lead_investors", []):
        conn.execute("INSERT OR IGNORE INTO deal_investors VALUES (?,?,?)", (deal_id, name.strip(), "lead"))
    for name in d.get("other_investors", []):
        conn.execute("INSERT OR IGNORE INTO deal_investors VALUES (?,?,?)", (deal_id, name.strip(), "participant"))
    return deal_id
