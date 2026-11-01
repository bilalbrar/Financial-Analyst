"""Refresh the local market cache: fetch daily prices for every sector's
instrument, then rebuild the news+price join (sector_daily) for all sectors.

Designed for the Alpha Vantage *free tier* — calls are spaced out by the client
and the daily quota is small, so prices are cached and only refetched when stale.

Usage:
    python scripts/refresh_market.py                # fetch stale + rebuild
    python scripts/refresh_market.py --force        # refetch everything
    python scripts/refresh_market.py --no-fetch     # rebuild from cache only
    python scripts/refresh_market.py --bonus        # also fetch BRENT/NATURAL_GAS
"""
import argparse
import io
import logging
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings
from config.market_map import BONUS_COMMODITIES, SECTOR_INSTRUMENTS
from graph.client import GraphClient
from ingestion.market_client import MarketClient, MarketRateLimit
from signals.market_correlation import build_sector_daily
from storage.database import Database

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("refresh_market")

START = "2025-06-01"
STALE_DAYS = 10  # commodity/market data lags a few days; avoid burning quota refetching


def _is_fresh(db: Database, key: str) -> bool:
    last = db.latest_price_date(key)
    if not last:
        return False
    last_d = datetime.strptime(last[:10], "%Y-%m-%d").date()
    return (date.today() - last_d).days <= STALE_DAYS


def fetch_prices(db: Database, force: bool, bonus: bool) -> dict:
    client = MarketClient()
    summary = {"fetched": [], "cached": [], "skipped": [], "errors": []}

    targets = [(s, inst) for s, inst in SECTOR_INSTRUMENTS.items()]
    for sector, inst in targets:
        key = inst["key"]
        if not force and _is_fresh(db, key):
            summary["cached"].append(key)
            logger.info("%s (%s): cache fresh, skipping fetch", sector, key)
            continue
        try:
            rows = client.fetch_daily(inst)
            rows = [(d, c) for d, c in rows if d >= START]
            n = db.upsert_prices(key, rows)
            summary["fetched"].append(f"{key}({n})")
        except MarketRateLimit as exc:
            logger.error("Rate limit hit on %s: %s", key, exc)
            summary["errors"].append(f"{key}: rate-limit")
            summary["skipped"] += [i["key"] for s, i in targets if i["key"] not in
                                   summary["fetched"] + summary["cached"] + [key]]
            logger.error("Stopping price fetch — daily quota likely exhausted.")
            break
        except Exception as exc:  # noqa: BLE001
            logger.error("Fetch failed for %s: %s", key, exc)
            summary["errors"].append(f"{key}: {exc}")

    if bonus:
        for key, fn in BONUS_COMMODITIES.items():
            if not force and _is_fresh(db, key):
                summary["cached"].append(key)
                continue
            try:
                rows = [(d, c) for d, c in client.fetch_commodity(fn) if d >= START]
                summary["fetched"].append(f"{key}({db.upsert_prices(key, rows)})")
            except MarketRateLimit as exc:
                logger.error("Rate limit on bonus %s: %s", key, exc)
                summary["errors"].append(f"{key}: rate-limit")
                break
            except Exception as exc:  # noqa: BLE001
                summary["errors"].append(f"{key}: {exc}")
    return summary


def rebuild_sector_daily(db: Database) -> dict:
    graph = GraphClient(
        settings.NEO4J_URI, settings.NEO4J_USER,
        settings.NEO4J_PASSWORD, settings.NEO4J_DATABASE,
    )
    result = {}
    try:
        for sector in SECTOR_INSTRUMENTS:
            rows = build_sector_daily(db, graph, sector, START)
            if rows:
                db.replace_sector_daily(sector, rows)
            result[sector] = len(rows)
            logger.info("Rebuilt sector_daily[%s]: %d rows", sector, len(rows))
    finally:
        graph.close()
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="refetch all prices")
    ap.add_argument("--no-fetch", action="store_true", help="rebuild from cache only")
    ap.add_argument("--bonus", action="store_true", help="also fetch BRENT/NATURAL_GAS")
    args = ap.parse_args()

    with Database(settings.DB_PATH) as db:
        db.init_schema()

        if not args.no_fetch:
            print("\n=== Fetching prices (rate-limited) ===")
            s = fetch_prices(db, force=args.force, bonus=args.bonus)
            print("fetched:", s["fetched"])
            print("cached :", s["cached"])
            if s["errors"]:
                print("errors :", s["errors"])

        print("\n=== Rebuilding sector_daily ===")
        rebuilt = rebuild_sector_daily(db)
        print("rows per sector:", rebuilt)

        print("\n=== Price cache summary ===")
        for sector, inst in SECTOR_INSTRUMENTS.items():
            key = inst["key"]
            df = db.get_prices(key)
            rng = f"{df['price_date'].min()}..{df['price_date'].max()}" if not df.empty else "EMPTY"
            print(f"  {sector:14s} {key:5s} {len(df):4d} pts  {rng}")


if __name__ == "__main__":
    main()
