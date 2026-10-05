# Türkiye Early-Stage Funding Map

A structured database of Turkish startup funding rounds and acquisitions, built from news articles with an LLM, and a short analysis of what it shows.

**Interactive map (Turkish):** https://maksu24-tech.github.io/tr-dealflow/

## Key finding: most rounds are reported without the basics

Between January 2025 and September 2026, 274 funding rounds linked to Türkiye were reported in the startup press. Of those:

| Missing in the news article | Rounds | Share |
|---|---|---|
| Round stage (seed, Series A, ...) | 165 | 60% |
| Amount | 107 | 39% |
| Both | 84 | 31% |
| Lead investor | 78 | 28% |

Rounds of Türkiye-headquartered companies are the least transparent: 67% have no stage. For Turkish-founded companies headquartered abroad, the figure is 40%. For 44 of the 51 acquisitions, the price was not disclosed.

Coverage is also incomplete. APY Ventures says it made 24 investments in 2025; the press reported 15 of them. Free sources either give only aggregate figures (quarterly ecosystem reports) or are scattered legal records such as the trade registry gazette. Deal-level data is paid. This project is a small example of the missing layer: free, structured and up to date.

Two further observations, with weaker evidence (small samples, a single main news source):

- **Turkish founders increasingly start companies abroad.** Their share of Türkiye-linked rounds went from 15% in 2025 to 31% in 2026. Over the same period, news volume fell by 25% and rounds of Türkiye-headquartered companies fell by 48%.
- **Gaming works like its own ecosystem.** It is the largest sector among Türkiye-headquartered rounds (34 of 181). Among investors with at least two gaming rounds, 9 of 27 invest only in gaming; in fintech the figure is 4 of 30.

## Data

| File | Contents |
|---|---|
| `data/deals.csv` | 349 deals: date, company, one-line description, sector, HQ country, Türkiye link, stage, amount (original currency and USD), source article URL |
| `data/deal_investors.csv` | 977 investor–deal pairs, with role (lead / participant) |

Article texts are not included; every row links to its source. Türkiye link categories: `hq_turkey` (headquartered in Türkiye), `turkish_founders` (headquartered abroad, Turkish founders), `turkish_investor` (foreign company, Turkish investor in the round), `none`.

## How it works

```
Egirişim (WordPress API) ─┐                         ┌─ deals (company, stage, amount, sector)
                          ├─► articles ─► LLM ─────┤
Webrazzi (RSS) ───────────┘   (SQLite)   extraction └─ deal_investors (lead / participant)
```

1. `fetch` downloads investment news from Egirişim and Webrazzi into SQLite.
2. A keyword prefilter skips articles that are clearly not about a deal.
3. `extract` sends each remaining article to an LLM that must answer in one fixed JSON schema (company, stage, amount, investors, sector, Türkiye link).
4. Currency conversion and investor-name merging run in Python, not in the model. `clean` merges alias spellings of the same fund and marks rounds that were reported twice.

To check accuracy, I hand-labelled a random set of 50 articles. On that set the extraction model (Claude) got the round stage right in 41 of 44 cases and the amount in 42 of 45. A small local model (Gemma 3 4B via Ollama) was much weaker: 11 of 44 and 25 of 45. The published dataset therefore uses Claude's extraction.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m dealflow fetch --since 2025-01-01
python -m dealflow extract --limit 20      # default provider: Ollama (free, local)
python -m dealflow clean
python -m dealflow stats
python -m unittest discover tests          # offline tests
```

Providers: `ollama` (default, local), `gemini` (free tier, needs `GEMINI_API_KEY`) or `anthropic` (paid, needs `ANTHROPIC_API_KEY`), set with `DEALFLOW_PROVIDER`. See `.env.example`.

## Limitations

- 382 of the 397 articles come from Egirişim. Webrazzi's RSS feed only exposes recent posts.
- Egirişim shows a short subtitle above each article that the WordPress API does not return. At least one labelled article gave its stage only there, so the 60% missing-stage figure is an upper bound.
- Totals in USD are dominated by a single deal (Dream Games, $2B). The analysis uses deal counts and medians instead.

## About

Built by Mehmet Eren Aksu (Koç University) with Claude as a coding assistant. I chose the scope and the questions, hand-labelled the test set and reviewed the results.
