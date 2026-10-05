"""Fetch funding news from public Turkish startup-news sources.

- Egirişim: WordPress REST API, "Yatırım" category (id 32). Paginates back in time.
- Webrazzi: RSS feed of the "Yatırım" category. RSS only returns recent posts, so
  run `fetch` regularly (Day 3: GitHub Actions cron) to build history over time.

Be polite: identify yourself, sleep between requests, never hammer the sites.
"""
import html
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

import requests
from bs4 import BeautifulSoup

USER_AGENT = "tr-dealflow/0.1 (student research project; contact via GitHub)"
SLEEP_SECONDS = 1.5

EGIRISIM_API = "https://egirisim.com/wp-json/wp/v2/posts"
EGIRISIM_FUNDING_CATEGORY = 32  # "Yatırım"
WEBRAZZI_FEEDS = [
    "https://webrazzi.com/kategori/yatirim/feed/",
    "https://webrazzi.com/feed/",  # fallback: main feed, filtered to category "Yatırım"
]

NS = {"content": "http://purl.org/rss/1.0/modules/content/"}


def html_to_text(raw: str) -> str:
    soup = BeautifulSoup(raw or "", "html.parser")
    for tag in soup(["script", "style", "iframe", "figure", "img"]):
        tag.decompose()
    text = soup.get_text("\n")
    text = html.unescape(text)
    text = re.sub(r"[ \t ]+", " ", text)
    return re.sub(r"\n\s*\n+", "\n\n", text).strip()


def _session() -> requests.Session:
    s = requests.Session()
    s.headers["User-Agent"] = USER_AGENT
    return s


# ---------- Egirişim (WordPress REST API) ----------

def parse_egirisim_posts(posts: list[dict]) -> list[dict]:
    out = []
    for p in posts:
        out.append({
            "id": f"egirisim:{p['id']}",
            "source": "egirisim",
            "url": p["link"],
            "title": html_to_text(p["title"]["rendered"]),
            "published_at": p.get("date_gmt") or p["date"],
            "body": html_to_text(p["content"]["rendered"]),
        })
    return out


def fetch_egirisim(since: str, max_pages: int = 50) -> list[dict]:
    """All posts in the funding category published after `since` (YYYY-MM-DD)."""
    s = _session()
    articles, page = [], 1
    while page <= max_pages:
        r = s.get(EGIRISIM_API, params={
            "categories": EGIRISIM_FUNDING_CATEGORY,
            "per_page": 100,
            "page": page,
            "after": f"{since}T00:00:00",
            "_fields": "id,date,date_gmt,link,title,content",
        }, timeout=30)
        if r.status_code == 400:  # WordPress returns 400 past the last page
            break
        r.raise_for_status()
        batch = parse_egirisim_posts(r.json())
        articles.extend(batch)
        total_pages = int(r.headers.get("X-WP-TotalPages", page))
        print(f"  egirisim page {page}/{total_pages}: {len(batch)} posts")
        if page >= total_pages:
            break
        page += 1
        time.sleep(SLEEP_SECONDS)
    return articles


# ---------- Webrazzi (RSS) ----------

def parse_rss(xml_text: str, source: str, only_category: str | None = None) -> list[dict]:
    root = ET.fromstring(xml_text)
    out = []
    for item in root.iter("item"):
        cats = [c.text or "" for c in item.findall("category")]
        if only_category and only_category not in cats:
            continue
        link = (item.findtext("link") or "").strip()
        body_html = item.findtext("content:encoded", namespaces=NS) or item.findtext("description") or ""
        pub = item.findtext("pubDate")
        published = parsedate_to_datetime(pub).astimezone(timezone.utc).isoformat() if pub else ""
        out.append({
            "id": f"{source}:{link}",
            "source": source,
            "url": link,
            "title": html_to_text(item.findtext("title") or ""),
            "published_at": published,
            "body": html_to_text(body_html),
        })
    return out


def fetch_webrazzi(max_pages: int = 10) -> list[dict]:
    s = _session()
    category_feed, main_feed = WEBRAZZI_FEEDS
    articles = []
    for page in range(1, max_pages + 1):
        url = category_feed if page == 1 else f"{category_feed}?paged={page}"
        r = s.get(url, timeout=30)
        if r.status_code != 200:
            break
        batch = parse_rss(r.text, "webrazzi")
        if not batch:
            break
        articles.extend(batch)
        print(f"  webrazzi page {page}: {len(batch)} posts")
        time.sleep(SLEEP_SECONDS)
    if not articles:  # category feed unavailable -> use main feed
        r = s.get(main_feed, timeout=30)
        r.raise_for_status()
        articles = parse_rss(r.text, "webrazzi", only_category="Yatırım")
        print(f"  webrazzi main feed: {len(articles)} funding posts")
    return articles


# ---------- cheap prefilter before spending LLM tokens ----------

# v2: v1 missed "X'e yatırım yaptı", "X'ten ilk yatırım", "ortak oldu" and all acquisitions (exits).
FUNDING_PATTERNS = re.compile(
    r"yatırım (aldı|turu|turunda|yaptı)|yatırım(ı|lar)?:|'?[td][ae]n (ilk |yeni |stratejik |üç |iki )*yatırım"
    r"|stratejik yatırım|ortak oldu|satın al|bünyesine kattı|çatısı altına|hissesi|exit"
    r"|tohum|seri [a-e]\b|melek yatırım|milyon (dolar|euro|avro|tl|sterlin)|bin (dolar|euro)|değerleme",
    re.IGNORECASE,
)


def looks_like_funding(article: dict) -> bool:
    text = f"{article['title']}\n{article['body'][:1500]}"
    return bool(FUNDING_PATTERNS.search(text))


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
