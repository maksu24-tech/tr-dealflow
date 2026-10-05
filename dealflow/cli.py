"""Command line entry point.

    python -m dealflow fetch --since 2025-01-01
    python -m dealflow extract --limit 20
    python -m dealflow stats
    python -m dealflow sample-for-labeling --n 150
"""
import argparse
import csv
import random
import sys

from . import db, sources
from .extract import MODEL, PROVIDER, extract_deals


def cmd_fetch(args):
    conn = db.connect(args.db)
    batches = []
    print("Fetching Egirişim ...")
    batches += sources.fetch_egirisim(since=args.since)
    print("Fetching Webrazzi ...")
    try:
        batches += sources.fetch_webrazzi()
    except Exception as e:  # one source failing should not kill the run
        print(f"  webrazzi failed: {e}", file=sys.stderr)
    new = sum(db.upsert_article(conn, a) for a in batches)
    conn.commit()
    print(f"Done: {len(batches)} fetched, {new} new articles stored.")


def cmd_extract(args):
    conn = db.connect(args.db)
    todo = db.pending_articles(conn, args.limit)
    print(f"{len(todo)} articles to process with {PROVIDER}:{MODEL}")
    saved = skipped = empty = errors = 0
    for i, a in enumerate(todo, 1):
        if not sources.looks_like_funding(a):
            db.mark_extracted(conn, a["id"], 3)
            skipped += 1
            continue
        try:
            deals = extract_deals(a)
        except Exception as e:
            print(f"  [{i}] ERROR {a['title'][:60]}: {e}", file=sys.stderr)
            db.mark_extracted(conn, a["id"], -1)
            errors += 1
            continue
        for d in deals:
            db.save_deal(conn, a["id"], d, f"{PROVIDER}:{MODEL}")
        db.mark_extracted(conn, a["id"], 1 if deals else 2)
        saved += len(deals)
        empty += not deals
        conn.commit()
        label = ", ".join(f"{d['company']} ({d['stage']}, {d['amount_usd'] or '?'} USD)" for d in deals) or "no deal"
        print(f"  [{i}/{len(todo)}] {a['title'][:60]} -> {label}")
    conn.commit()
    print(f"Saved {saved} deals | {empty} non-deal articles | {skipped} prefiltered | {errors} errors")


def cmd_stats(args):
    conn = db.connect(args.db)
    q = lambda sql: conn.execute(sql).fetchall()
    print("Articles by status:", {r[0]: r[1] for r in q("SELECT extracted, COUNT(*) FROM articles GROUP BY 1")})
    has_dup = "duplicate_of" in {r[1] for r in conn.execute("PRAGMA table_info(deals)")}
    live = "duplicate_of IS NULL" if has_dup else "1=1"
    print("Deals:", q(f"SELECT COUNT(*) FROM deals WHERE {live}")[0][0])
    print("\nTürkiye-linked deals by stage:")
    for r in q("""SELECT stage, COUNT(*) n, ROUND(SUM(amount_usd)/1e6,1) musd FROM deals
                  WHERE turkey_link != 'none' AND {live} GROUP BY stage ORDER BY n DESC""".format(live=live)):
        print(f"  {r['stage']:<14} {r['n']:>4} deals  {r['musd'] or 0:>8} M USD")
    print("\nMost active investors in Türkiye-linked deals:")
    for r in q("""SELECT investor, COUNT(*) n, SUM(role='lead') leads FROM deal_investors di
                  JOIN deals d ON d.id = di.deal_id WHERE d.turkey_link != 'none' AND d.{live}
                  GROUP BY investor ORDER BY n DESC LIMIT 15""".format(live=live)):
        print(f"  {r['investor']:<35} {r['n']:>3} deals ({r['leads']} led)")


def cmd_sample(args):
    """Export a random sample for hand-labelling (Day 2 accuracy eval)."""
    conn = db.connect(args.db)
    rows = conn.execute("""SELECT a.id, a.url, a.title, d.company, d.stage, d.amount, d.currency,
                                  d.turkey_link, d.sector, d.raw_json
                           FROM articles a LEFT JOIN deals d ON d.article_id = a.id
                           WHERE a.extracted IN (1, 2)""").fetchall()
    random.seed(42)
    sample = random.sample(rows, min(args.n, len(rows)))
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["article_id", "url", "title", "pred_company", "pred_stage", "pred_amount", "pred_currency",
                    "pred_turkey_link", "pred_sector",
                    "true_is_deal", "true_company", "true_stage", "true_amount", "true_currency", "true_turkey_link",
                    "notes"])
        for r in sample:
            w.writerow([r["id"], r["url"], r["title"], r["company"], r["stage"], r["amount"], r["currency"],
                        r["turkey_link"], r["sector"], "", "", "", "", "", "", ""])
    print(f"Wrote {len(sample)} rows to {args.out}. Fill the true_* columns by reading each article.")


def cmd_eval_run(args):
    """Run one model on the fixed test set only. Results go to model_runs, never to deals."""
    import json
    import time
    from . import extract
    if args.model:
        extract.MODEL = args.model
    name = f"{extract.PROVIDER}:{extract.MODEL}"
    conn = db.connect(args.db)
    todo = conn.execute("""SELECT a.* FROM eval_set e JOIN articles a ON a.id = e.article_id
                           WHERE a.id NOT IN (SELECT article_id FROM model_runs
                                              WHERE model = ? AND (deal_json IS NULL OR deal_json NOT LIKE '{"error"%'))
                           ORDER BY e.position""", (name,)).fetchall()
    total = conn.execute("SELECT COUNT(*) FROM eval_set").fetchone()[0]
    print(f"Test set: {total} articles, {len(todo)} left for {name}", flush=True)
    for i, a in enumerate(todo, 1):
        print(f"  [{i}/{len(todo)}] {a['title'][:60]} ...", end=" ", flush=True)
        t0 = time.time()
        conn.execute("DELETE FROM model_runs WHERE article_id = ? AND model = ?", (a["id"], name))
        try:
            deals = extract.extract_deals(a)
            for d in deals:
                conn.execute("INSERT INTO model_runs (article_id, model, deal_json) VALUES (?,?,?)",
                             (a["id"], name, json.dumps(d, ensure_ascii=False)))
            if not deals:
                conn.execute("INSERT INTO model_runs (article_id, model, deal_json) VALUES (?,?,NULL)", (a["id"], name))
            print(f"{', '.join(d['company'] for d in deals) or 'no deal'} ({time.time() - t0:.0f}s)", flush=True)
        except Exception as e:
            conn.execute("INSERT INTO model_runs (article_id, model, deal_json) VALUES (?,?,?)",
                         (a["id"], name, json.dumps({"error": str(e)})))
            print(f"ERROR: {e}", flush=True)
        conn.commit()
    print("Done. Send data/dealflow.db back for scoring.")


def cmd_clean(args):
    from . import clean
    conn = db.connect(args.db)
    n, dups = clean.run(conn)
    print(f"Renamed {n} investor aliases.")
    for company, dup, keep, reason in dups:
        print(f"  duplicate: {company} #{dup} -> #{keep} ({reason})")
    left = clean.duplicate_candidates(conn)
    print(f"{len(left)} companies still have several rounds (check they are genuinely different):")
    for rows in left.values():
        print("  " + " | ".join(f"{r[1]} {r[2]} {r[3]}" for r in rows))


def cmd_score(args):
    import json
    from . import score as S
    conn = db.connect(args.db)
    labels = S.load_labels(args.labels)
    models = [m for (m,) in conn.execute("SELECT DISTINCT model FROM model_runs WHERE article_id IN (SELECT article_id FROM eval_set)")]
    overrides = json.load(open(args.overrides)) if args.overrides else None
    table, rows = S.score(conn, labels, models, overrides)
    print(f"{'field':16}" + "".join(f"{m[:28]:>30}" for m in models))
    for f in S.FIELDS:
        cells = []
        for m in models:
            a, b = table[m][f]
            cells.append(f"{a}/{b} ({100 * a / b:.0f}%)" if b else "-")
        print(f"{f:16}" + "".join(f"{c:>30}" for c in cells))
    S.write_rows(rows, args.out)
    print(f"Per-article results written to {args.out}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="dealflow")
    p.add_argument("--db", default="data/dealflow.db")
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch"); f.add_argument("--since", default="2025-01-01"); f.set_defaults(fn=cmd_fetch)
    e = sub.add_parser("extract"); e.add_argument("--limit", type=int); e.set_defaults(fn=cmd_extract)
    s = sub.add_parser("stats"); s.set_defaults(fn=cmd_stats)
    l = sub.add_parser("sample-for-labeling"); l.add_argument("--n", type=int, default=150)
    l.add_argument("--out", default="data/labels.csv"); l.set_defaults(fn=cmd_sample)
    sc = sub.add_parser("score", help="score models against hand labels")
    sc.add_argument("--labels", required=True); sc.add_argument("--overrides")
    sc.add_argument("--out", default="data/eval_rows.csv"); sc.set_defaults(fn=cmd_score)
    cl = sub.add_parser("clean", help="merge investor aliases and duplicate rounds"); cl.set_defaults(fn=cmd_clean)
    ev = sub.add_parser("eval-run", help="run a model on the fixed test set only")
    ev.add_argument("--model", help="e.g. gemma3:4b"); ev.set_defaults(fn=cmd_eval_run)
    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
