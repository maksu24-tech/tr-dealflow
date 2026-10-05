"""Offline tests: parsing, prefilter, normalisation and DB, with Claude mocked.

Run:  python -m unittest discover tests
"""
import json
import unittest
from pathlib import Path
from types import SimpleNamespace

from dealflow import db, sources
from dealflow import extract
from dealflow.extract import extract_deals, normalise

FIX = Path(__file__).parent / "fixtures"


class FakeHTTP:
    """Mimics requests.post for the Ollama / Gemini REST APIs."""
    def __init__(self, body):
        self.body, self.calls = body, []

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return SimpleNamespace(status_code=200, raise_for_status=lambda: None, json=lambda: self.body)


class FakeClient:
    """Mimics anthropic.Anthropic().messages.create returning a forced tool call."""
    def __init__(self, payload):
        self.payload = payload
        self.calls = []
        self.messages = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(content=[SimpleNamespace(type="tool_use", input=self.payload)])


CIRCLE_GAMES = {"deals": [{
    "company": " Circle Games ", "description": "Istanbul mobile puzzle game studio.", "sector": "gaming",
    "hq_country": "TR", "turkey_link": "hq_turkey", "stage": "series_a", "amount": 25000000, "currency": "usd",
    "valuation": None, "lead_investors": ["Tencent"], "other_investors": ["BITKRAFT Ventures", "e2vc", "Tencent"],
    "confidence": 0.95,
}]}


class TestSources(unittest.TestCase):
    def test_parse_egirisim(self):
        posts = json.loads((FIX / "egirisim_posts.json").read_text())
        arts = sources.parse_egirisim_posts(posts)
        self.assertEqual(arts[0]["id"], "egirisim:122016")
        self.assertIn("Circle Games", arts[0]["title"])
        self.assertNotIn("<strong>", arts[0]["body"])
        self.assertNotIn("Bakan", arts[1]["body"])  # figure captions stripped
        self.assertIn("Terminal İstanbul’da", arts[1]["title"])  # HTML entities decoded

    def test_parse_rss_with_category_filter(self):
        xml = (FIX / "webrazzi_feed.xml").read_text()
        self.assertEqual(len(sources.parse_rss(xml, "webrazzi")), 2)
        only = sources.parse_rss(xml, "webrazzi", only_category="Yatırım")
        self.assertEqual(len(only), 1)
        self.assertEqual(only[0]["published_at"], "2026-09-25T12:00:00+00:00")

    def test_prefilter(self):
        posts = sources.parse_egirisim_posts(json.loads((FIX / "egirisim_posts.json").read_text()))
        self.assertTrue(sources.looks_like_funding(posts[0]))   # Circle Games round
        self.assertFalse(sources.looks_like_funding(posts[1]))  # event news


class TestExtract(unittest.TestCase):
    def test_normalise(self):
        d = normalise(CIRCLE_GAMES["deals"][0], "2026-09-24T13:20:55")
        self.assertEqual(d["company"], "Circle Games")
        self.assertEqual(d["currency"], "USD")
        self.assertEqual(d["amount_usd"], 25000000)
        self.assertEqual(d["announced_date"], "2026-09-24")
        self.assertNotIn("Tencent", d["other_investors"])  # lead not double counted

    def test_bad_enums_and_undisclosed_amount(self):
        d = normalise({"company": "X", "stage": "mezzanine", "sector": "space", "turkey_link": "maybe",
                       "amount": None, "currency": None}, "")
        self.assertEqual((d["stage"], d["sector"], d["turkey_link"]), ("unknown", "other", "none"))
        self.assertIsNone(d["amount_usd"])

    def article(self):
        return sources.parse_egirisim_posts(json.loads((FIX / "egirisim_posts.json").read_text()))[0]

    def test_anthropic_forces_tool(self):
        client = FakeClient(CIRCLE_GAMES)
        deals = extract_deals(self.article(), call=lambda a: extract.call_anthropic(a, client=client))
        self.assertEqual(deals[0]["stage"], "series_a")
        self.assertEqual(client.calls[0]["tool_choice"], {"type": "tool", "name": "record_deals"})

    def test_ollama_sends_schema(self):
        http = FakeHTTP({"message": {"content": json.dumps(CIRCLE_GAMES)}})
        deals = extract_deals(self.article(), call=lambda a: extract.call_ollama(a, http=http))
        url, kw = http.calls[0]
        self.assertTrue(url.endswith("/api/chat"))
        self.assertEqual(kw["json"]["format"], extract.DEALS_SCHEMA)
        self.assertFalse(kw["json"]["stream"])
        self.assertEqual(deals[0]["company"], "Circle Games")

    def test_gemini_schema_is_openapi(self):
        import os
        os.environ.setdefault("GEMINI_API_KEY", "test")
        http = FakeHTTP({"candidates": [{"content": {"parts": [{"text": json.dumps(CIRCLE_GAMES)}]}}]})
        deals = extract_deals(self.article(), call=lambda a: extract.call_gemini(a, http=http))
        schema = http.calls[0][1]["json"]["generationConfig"]["responseSchema"]
        amount = schema["properties"]["deals"]["items"]["properties"]["amount"]
        self.assertEqual(amount, {"type": "number", "nullable": True, "description": amount["description"]})
        self.assertNotIn("['", json.dumps(schema))
        self.assertEqual(deals[0]["amount_usd"], 25000000)

    def test_empty_company_dropped(self):
        deals = extract_deals(self.article(), call=lambda a: {"deals": [{"company": ""}]})
        self.assertEqual(deals, [])


class TestDB(unittest.TestCase):
    def test_roundtrip(self):
        conn = db.connect(":memory:")
        art = sources.parse_egirisim_posts(json.loads((FIX / "egirisim_posts.json").read_text()))[0]
        self.assertTrue(db.upsert_article(conn, art))
        self.assertFalse(db.upsert_article(conn, art))  # dedupe
        d = normalise(CIRCLE_GAMES["deals"][0], art["published_at"])
        db.save_deal(conn, art["id"], d, "test-model")
        db.save_deal(conn, art["id"], d, "test-model")  # re-run is idempotent
        db.mark_extracted(conn, art["id"], 1)
        self.assertEqual(conn.execute("SELECT COUNT(*) FROM deals").fetchone()[0], 1)
        roles = dict(conn.execute("SELECT investor, role FROM deal_investors").fetchall())
        self.assertEqual(roles, {"Tencent": "lead", "BITKRAFT Ventures": "participant", "e2vc": "participant"})
        self.assertEqual(db.pending_articles(conn), [])


class TestClean(unittest.TestCase):
    def test_aliases_and_duplicates(self):
        from dealflow import clean
        conn = db.connect(":memory:")
        for i, (src, date) in enumerate([("egirisim", "2026-09-28"), ("webrazzi", "2026-09-28")]):
            conn.execute("INSERT INTO articles (id, source, url, title, published_at, body) VALUES (?,?,?,?,?,?)",
                         (f"{src}:{i}", src, f"u{i}", "t", date, "b"))
            d = normalise({"company": "Bluecore Energy", "stage": "seed", "amount": 5e7, "currency": "USD",
                           "lead_investors": ["Silverton Partners"],
                           "other_investors": ["Sharks&Partners"] if i == 0 else ["Re-Pie"]}, date)
            db.save_deal(conn, f"{src}:{i}", d, "m")
        n1, d1 = clean.run(conn)
        n2, d2 = clean.run(conn)           # idempotent
        self.assertEqual((n1, n2), (2, 0))
        self.assertEqual(len(d1), 1)
        kept = conn.execute("SELECT id FROM deals WHERE duplicate_of IS NULL").fetchall()
        self.assertEqual(len(kept), 1)
        invs = {r[0] for r in conn.execute("SELECT investor FROM deal_investors WHERE deal_id=?", kept[0])}
        self.assertEqual(invs, {"Silverton Partners", "Sharks & Partners", "RePie Portföy"})


if __name__ == "__main__":
    unittest.main()
