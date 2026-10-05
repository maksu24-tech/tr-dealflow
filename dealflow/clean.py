"""Post-extraction cleaning: investor name aliases and duplicate rounds.

Both steps are idempotent and keep the raw model output in deals.raw_json.
Run: python -m dealflow clean
"""
import unicodedata

# alias -> canonical name. Built by reading the full investor list by hand.
INVESTOR_ALIASES = {
    "Ahu Büyükkuşoğlu Serter": "Ahu Serter",
    "Ak Portföy GSYF": "Ak Portföy",
    "Akkök": "Akkök Holding",
    "Balderton Growth Fund": "Balderton",
    "DST": "DST Global",
    "Insha GSYF": "Insha Ventures",
    "Meritech": "Meritech Capital",
    "NVentures": "NVIDIA",
    "Point72 Private Investments": "Point72",
    "Point72 Ventures": "Point72",
    "Re-Pie": "RePie Portföy",
    "Re-Pie Portföy": "RePie Portföy",
    "Sharks&Partners": "Sharks & Partners",
    "SGlobe GSYF": "SGlobe",
    "ÜNLÜ Ventures": "Ünlü & Co",
    "İTÜ ARI Teknokent GSYF": "İTÜ ARI Teknokent",
    "Kfund": "K Fund",
    "Maxis Girişim Sermayesi": "Maxis Ventures",
    "Tera GSYO": "Tera Girişim",
    "Oxtech GSYF": "Deniz Portföy",
}

# (duplicate deal, kept deal, reason). Found by listing companies with >1 record
# and reading both articles. Different rounds of the same company stay separate.
DUPLICATES = [
    ("Bluecore Energy", "2026-09-28", "egirisim", "same round reported by Egirişim and Webrazzi"),
    ("GetirAraç", "2025-10-07", None, "follow-up article on the same acquisition"),
    ("Milvus Robotics", "2025-05-27", None, "Series A later extended from 4.5M to 5.5M"),
    ("Lucida AI", "2026-04-15", None, "seed round later extended from 5.4M to 7M"),
    ("Eloquent AI", "2025-09-10", None, "seed round later extended from 7.4M to 8.4M"),
    ("Winfluencer", "2025-04-29", None, "one round announced investor by investor"),
    ("Winfluencer", "2025-08-19", None, "one round announced investor by investor"),
]


def _fold(s: str) -> str:
    return unicodedata.normalize("NFKC", s).strip().casefold()


def ensure_columns(conn):
    cols = {r[1] for r in conn.execute("PRAGMA table_info(deals)")}
    if "duplicate_of" not in cols:
        conn.execute("ALTER TABLE deals ADD COLUMN duplicate_of INTEGER REFERENCES deals(id)")


def normalise_investors(conn) -> int:
    changed = 0
    for alias, canon in INVESTOR_ALIASES.items():
        rows = conn.execute("SELECT deal_id, role FROM deal_investors WHERE investor = ?", (alias,)).fetchall()
        for deal_id, role in rows:
            conn.execute("DELETE FROM deal_investors WHERE deal_id = ? AND investor = ?", (deal_id, alias))
            existing = conn.execute("SELECT role FROM deal_investors WHERE deal_id = ? AND investor = ?",
                                    (deal_id, canon)).fetchone()
            if existing is None:
                conn.execute("INSERT INTO deal_investors VALUES (?,?,?)", (deal_id, canon, role))
            elif role == "lead":
                conn.execute("UPDATE deal_investors SET role='lead' WHERE deal_id=? AND investor=?", (deal_id, canon))
            changed += 1
    return changed


def mark_duplicates(conn) -> list[tuple]:
    """Point each duplicate at the most recent other record of the same company and copy its investors over."""
    ensure_columns(conn)
    done = []
    for company, date, source, reason in DUPLICATES:
        sql = """SELECT d.id FROM deals d JOIN articles a ON a.id = d.article_id
                 WHERE d.company = ? AND d.announced_date = ?"""
        params = [company, date]
        if source:
            sql += " AND a.source = ?"
            params.append(source)
        dup = conn.execute(sql, params).fetchone()
        if not dup:
            continue
        keep = conn.execute("""SELECT id FROM deals WHERE company = ? AND id != ? AND duplicate_of IS NULL
                               ORDER BY announced_date DESC, id LIMIT 1""", (company, dup[0])).fetchone()
        if not keep:
            continue
        conn.execute("UPDATE deals SET duplicate_of = ? WHERE id = ?", (keep[0], dup[0]))
        for inv, role in conn.execute("SELECT investor, role FROM deal_investors WHERE deal_id = ?", (dup[0],)).fetchall():
            conn.execute("INSERT OR IGNORE INTO deal_investors VALUES (?,?,?)", (keep[0], inv, role))
        done.append((company, dup[0], keep[0], reason))
    return done


def duplicate_candidates(conn):
    """Same company name (case/accents folded) seen more than once: review these by hand."""
    rows = conn.execute("SELECT id, company, announced_date, stage, amount_usd FROM deals WHERE duplicate_of IS NULL").fetchall()
    groups = {}
    for r in rows:
        groups.setdefault(_fold(r[1]), []).append(r)
    return {k: v for k, v in groups.items() if len(v) > 1}


def run(conn):
    n = normalise_investors(conn)
    d = mark_duplicates(conn)
    conn.commit()
    return n, d
