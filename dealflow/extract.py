"""Turn a news article into structured deal records with an LLM.

Providers (pick with DEALFLOW_PROVIDER):
  ollama    - local model, free, default
  gemini    - Gemini API with a free Google AI Studio key
  anthropic - Claude API, paid
Every provider is constrained to the same JSON schema, then the result is
normalised in plain Python (never trust the LLM with arithmetic).
"""
import json
import os
import time

import requests

PROVIDER = os.getenv("DEALFLOW_PROVIDER", "ollama")
DEFAULT_MODELS = {"ollama": "qwen3:4b", "gemini": "gemini-flash-latest", "anthropic": "claude-haiku-4-5-20251001"}
MODEL = os.getenv("DEALFLOW_MODEL", DEFAULT_MODELS.get(PROVIDER, ""))
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
# 4096 tokens fits a 6000-char article + prompt; larger contexts use more RAM (tight on 8 GB machines)
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "4096"))

STAGES = ["pre_seed", "seed", "series_a", "series_b", "series_c", "series_d_plus",
          "bridge", "growth", "debt", "grant", "acquisition", "unknown"]
TURKEY_LINKS = ["hq_turkey", "turkish_founders", "turkish_investor", "none"]
SECTORS = ["fintech", "gaming", "ecommerce_retail", "b2b_saas", "ai_ml", "health", "mobility_logistics",
           "energy_climate", "edtech", "deeptech_hardware", "food_agri", "media_consumer", "other"]

# Approximate FX to USD. Update these to current rates before running the analysis.
USD_PER = {"USD": 1.0, "EUR": 1.15, "GBP": 1.33, "TRY": 0.024}

RECORD_DEALS_TOOL = {
    "name": "record_deals",
    "description": "Record every startup funding round reported in the article. Return an empty list if the "
                   "article does not announce a specific funding round (e.g. event news, fund launches, product news).",
    "input_schema": {
        "type": "object",
        "properties": {
            "deals": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "company": {"type": "string", "description": "Startup that raised money."},
                        "description": {"type": "string", "description": "What the company does, one short English sentence."},
                        "sector": {"type": "string", "enum": SECTORS},
                        "hq_country": {"type": "string", "description": "ISO country code of HQ, e.g. TR, US, GB. Empty if unknown."},
                        "turkey_link": {"type": "string", "enum": TURKEY_LINKS,
                                        "description": "hq_turkey if HQ in Türkiye; turkish_founders if founded by Turks abroad; "
                                                       "turkish_investor if only a Türkiye-based investor participates; none otherwise."},
                        "stage": {"type": "string", "enum": STAGES},
                        "amount": {"type": ["number", "null"], "description": "Round size as a plain number (2.75 million -> 2750000). null if undisclosed."},
                        "currency": {"type": ["string", "null"], "description": "ISO code: USD, EUR, GBP, TRY."},
                        "valuation": {"type": ["number", "null"], "description": "Post-money valuation if stated, same currency."},
                        "lead_investors": {"type": "array", "items": {"type": "string"}},
                        "other_investors": {"type": "array", "items": {"type": "string"},
                                            "description": "Participating investors, excluding leads. Skip vague mentions like 'melek yatırımcılar'."},
                        "confidence": {"type": "number", "description": "0-1: how sure you are the fields are correct."},
                    },
                    "required": ["company", "sector", "turkey_link", "stage", "amount", "currency",
                                 "lead_investors", "other_investors", "confidence"],
                },
            }
        },
        "required": ["deals"],
    },
}

SYSTEM = """You extract venture funding data from Turkish and English startup news.
Rules:
- Only record rounds that the article announces as happening now. Ignore past rounds mentioned as background
  (e.g. "Temmuz 2025'te 7,25 milyon dolarlık tohum yatırımı almıştı").
- Turkish number format: "2,75 milyon" = 2750000; "32.25 milyon" = 32250000.
- "liderliğinde" / "led by" => lead investor. "Tura ... katıldı" => participant.
- Use investor names exactly as written, without descriptions ("mevcut yatırımcısı e2vc" -> "e2vc").
- "tohum öncesi" = pre_seed, "tohum" = seed, "Seri A" = series_a, "köprü" = bridge.
- Private-equity or development-fund stakes in mature companies are still recorded, with stage "growth"."""


def build_prompt(article) -> str:
    return (f"Published: {article['published_at']}\nURL: {article['url']}\n"
            f"Title: {article['title']}\n\n{article['body'][:6000]}")


def normalise(deal: dict, published_at: str) -> dict:
    """Validate enums and compute USD fields in Python."""
    d = dict(deal)
    d["company"] = d["company"].strip()
    if d.get("stage") not in STAGES:
        d["stage"] = "unknown"
    if d.get("sector") not in SECTORS:
        d["sector"] = "other"
    if d.get("turkey_link") not in TURKEY_LINKS:
        d["turkey_link"] = "none"
    cur = (d.get("currency") or "").upper() or None
    d["currency"] = cur
    rate = USD_PER.get(cur) if cur else None
    d["amount_usd"] = round(d["amount"] * rate) if d.get("amount") and rate else None
    d["valuation_usd"] = round(d["valuation"] * rate) if d.get("valuation") and rate else None
    d["announced_date"] = (published_at or "")[:10]
    lead = [x.strip() for x in d.get("lead_investors", []) if x.strip()]
    d["lead_investors"] = lead
    d["other_investors"] = [x.strip() for x in d.get("other_investors", []) if x.strip() and x.strip() not in lead]
    return d


# ---------- providers: all return {"deals": [...]} in the same schema ----------

def to_openapi_schema(schema):
    """Gemini's responseSchema is an OpenAPI subset: ["number","null"] -> type number + nullable."""
    if isinstance(schema, dict):
        out = {}
        for k, v in schema.items():
            if k == "type" and isinstance(v, list):
                out["type"] = next(t for t in v if t != "null")
                if "null" in v:
                    out["nullable"] = True
            else:
                out[k] = to_openapi_schema(v)
        return out
    if isinstance(schema, list):
        return [to_openapi_schema(x) for x in schema]
    return schema


DEALS_SCHEMA = RECORD_DEALS_TOOL["input_schema"]
JSON_INSTRUCTION = "\n\nAnswer only with JSON matching the given schema. Return {\"deals\": []} if no round is announced."


def call_ollama(article, http=requests) -> dict:
    """Local model through Ollama (free). Uses structured outputs: `format` = JSON schema."""
    r = http.post(f"{OLLAMA_URL}/api/chat", json={
        "model": MODEL,
        "stream": False,
        "format": DEALS_SCHEMA,
        "options": {"temperature": 0, "num_ctx": OLLAMA_NUM_CTX},
        "messages": [
            {"role": "system", "content": SYSTEM + JSON_INSTRUCTION},
            {"role": "user", "content": build_prompt(article)},
        ],
    }, timeout=300)
    r.raise_for_status()
    return json.loads(r.json()["message"]["content"])


def call_gemini(article, http=requests, retries=3) -> dict:
    """Gemini API with a free AI Studio key. JSON mode + response schema."""
    key = os.environ["GEMINI_API_KEY"]
    r = http.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent",
        headers={"x-goog-api-key": key},
        json={
            "systemInstruction": {"parts": [{"text": SYSTEM}]},
            "contents": [{"role": "user", "parts": [{"text": build_prompt(article)}]}],
            "generationConfig": {
                "temperature": 0,
                "responseMimeType": "application/json",
                "responseSchema": to_openapi_schema(DEALS_SCHEMA),
            },
        },
        timeout=120,
    )
    if r.status_code == 429 and retries > 0:  # free-tier rate limit: wait and retry
        time.sleep(60)
        return call_gemini(article, http, retries - 1)
    r.raise_for_status()
    return json.loads(r.json()["candidates"][0]["content"]["parts"][0]["text"])


def call_anthropic(article, client=None) -> dict:
    """Claude API (paid). Forced tool call so the answer must match the schema."""
    if client is None:
        import anthropic  # imported lazily so the other providers don't need the SDK
        client = anthropic.Anthropic()
    resp = client.messages.create(
        model=MODEL, max_tokens=2000, system=SYSTEM,
        tools=[RECORD_DEALS_TOOL], tool_choice={"type": "tool", "name": "record_deals"},
        messages=[{"role": "user", "content": build_prompt(article)}],
    )
    block = next(b for b in resp.content if getattr(b, "type", None) == "tool_use")
    return block.input if isinstance(block.input, dict) else json.loads(block.input)


PROVIDERS = {"ollama": call_ollama, "gemini": call_gemini, "anthropic": call_anthropic}


def extract_deals(article, call=None) -> list[dict]:
    """One model call per article; `call` is injectable for tests."""
    raw = (call or PROVIDERS[PROVIDER])(article)
    return [normalise(d, article["published_at"]) for d in raw.get("deals", []) if d.get("company")]
