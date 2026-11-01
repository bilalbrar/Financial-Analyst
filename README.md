# JARVIS-Fi — Step 1: News Ingestion Pipeline

This module ingests global financial news from the GDELT DOC 2.0 API, enriches
and deduplicates articles, scores their sentiment, and stores everything in a
local SQLite database ready for downstream ML and graph analysis.

## Setup

```bash
git clone <repo>
cd jarvis-fi
pip install -r requirements.txt
cp .env.example .env          # edit DB_PATH / LOG_LEVEL as needed
python scripts/run_backfill.py --days 7 --sector energy
```

## Run the live scheduler

```bash
python scripts/run_live.py
```

Jobs:
- **Hourly** — fetches articles for all sectors from the past 2 hours
- **Daily 00:05 UTC** — fetches tone timeline for the past 7 days, upserts signals
- **Every 15 min** — writes row counts to `data/health.log`

Logs are written to `data/scheduler.log` and stdout.

## Run backfill

```bash
# All sectors, last 90 days
python scripts/run_backfill.py

# One sector, custom range
python scripts/run_backfill.py --days 30 --sector banking

# Preview without writing
python scripts/run_backfill.py --days 7 --dry-run
```

## Run tests

```bash
pytest tests/ -v
```

## DB schema

| Table            | Description                                              |
|------------------|----------------------------------------------------------|
| `articles`       | One row per unique article (deduped by SHA-256 of URL)  |
| `tone_signals`   | Daily tone score per sector (upserted, no duplicates)   |
| `ingestion_runs` | Audit log of every backfill / scheduled run             |

## What's Next

**Step 2 — Neo4j graph ingestion**: model articles, entities (companies,
countries, people), and sectors as nodes; publish/mention/relates_to edges;
enable graph queries like "which companies co-appear with 'oil sanctions'
in the 30 days before an energy price spike?".

## Financial Guru dashboard (news × market)

Links the news/graph data to Alpha Vantage price series, finds where news
flow actually moves markets, and serves a local Plotly Dash dashboard.

```bash
# 1. cache prices + rebuild the news+price join (rate-limited, free tier)
python scripts/refresh_market.py            # --force refetch | --no-fetch rebuild only | --bonus BRENT/NATGAS

# 2. launch the dashboard
python dashboard/app.py                      # -> http://127.0.0.1:8050
```

Tabs: **Market Overview** (cross-sector amplification + link strength + signal
feed), **Sector Deep-Dive** (price chart with news-spike overlay, correlations,
top driving entities, headlines), **Why Did It Move?** (click a big move → the
news behind it), **Ask the Guru** (Groq LLM Q&A — `intelligence/guru.py` grounds
`GROQ_LARGE_MODEL` in the computed market+graph context; falls back to a
deterministic brief if no `GROQ_API_KEY`).

Each sector maps to an instrument and to knowledge-graph "driver" entities in
`config/market_map.py`. Prices cache into `prices`; the daily news+price join
lives in `sector_daily`. Signals are **decision-support** (volatility/risk +
attribution), not buy/sell advice — news flow tracks volatility better than
direction. Energy (WTI, driven by Iran/Hormuz/OPEC) is the strongest link:
r≈0.37 and ~2.5× volatility on news-spike days.

> Alpha Vantage free tier: `outputsize=full` is premium for `TIME_SERIES_DAILY`
> (ETFs use `compact` = 100 days); WTI/BRENT/NATURAL_GAS commodity endpoints are
> unrestricted. Quota ≈ 25 calls/day, so prices are cached and reused.
