"""One-time historical backfill script for JARVIS-Fi.

Usage:
    python scripts/run_backfill.py
    python scripts/run_backfill.py --days 365
    python scripts/run_backfill.py --days 365 --sector energy
    python scripts/run_backfill.py --days 365 --dry-run

GDELT DOC 2.0 limits article_search to ~90-day windows. When --days exceeds
--chunk-days (default 90), the range is automatically split into sequential
chunks so the full period is covered.
"""
import argparse
import json
import logging
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tabulate import tabulate

from config import settings
from config.themes import FINANCIAL_THEMES
from ingestion.article_processor import process_articles
from ingestion.gdelt_client import GDELTClient
from ingestion.sentiment import tone_series_to_signal
from storage.database import Database

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s %(levelname)-8s %(name)s - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("backfill")

_SLEEP_BETWEEN_KEYWORDS = 5   # seconds between keyword queries
_RATE_LIMIT_SLEEP = 30        # extra sleep when a rate limit is hit


def _exc_desc(exc: Exception) -> str:
    """Return a non-empty description for exceptions that have no str() message.

    gdeltdoc raises HTTPError subclasses with only `response=` set, so
    str(exc) is ''. This falls back to type name + HTTP status + body snippet.
    """
    msg = str(exc)
    if msg:
        return msg
    resp = getattr(exc, "response", None)
    if resp is not None:
        body = (resp.text or "")[:200].strip()
        return f"{type(exc).__name__} HTTP {resp.status_code}" + (f": {body}" if body else "")
    return repr(exc)



def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="JARVIS-Fi historical backfill")
    parser.add_argument(
        "--days", type=int, default=settings.BACKFILL_DAYS,
        help="Number of past days to backfill (default: %(default)s)",
    )
    parser.add_argument(
        "--chunk-days", type=int, default=settings.BACKFILL_QUERY_DAYS,
        help="Days per GDELT query window — smaller = more total articles (default: %(default)s)",
    )
    parser.add_argument(
        "--sector", type=str, default=None,
        help="Run only this sector (default: all sectors)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print planned queries without writing to DB",
    )
    return parser.parse_args()


def _date_windows(days: int, chunk_days: int) -> list[tuple[str, str]]:
    """Split a lookback period into (start, end) chunks of at most chunk_days.

    Chunks are ordered oldest-first so new articles are inserted in
    chronological order.
    """
    now = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    overall_start = now - timedelta(days=days)

    windows: list[tuple[str, str]] = []
    chunk_start = overall_start
    while chunk_start < now:
        chunk_end = min(chunk_start + timedelta(days=chunk_days), now)
        windows.append((
            chunk_start.strftime("%Y-%m-%d"),
            chunk_end.strftime("%Y-%m-%d"),
        ))
        chunk_start = chunk_end
    return windows


def _fetch_tone_with_backoff(client: GDELTClient, keyword: str, start: str, end: str):
    """Fetch tone timeline, sleeping longer on repeated rate limits."""
    from gdeltdoc.errors import RateLimitError
    for attempt in range(5):
        try:
            return client.fetch_tone_timeline([keyword], start, end)
        except RateLimitError:
            wait = _RATE_LIMIT_SLEEP * (attempt + 1)
            print(f"    [rate limit] tone timeline, sleeping {wait}s ...")
            time.sleep(wait)
    return None  # give up gracefully after 5 attempts


def _run_sector(
    sector: str,
    keywords: list[str],
    start_date: str,
    end_date: str,
    client: GDELTClient,
    db: Database,
    dry_run: bool,
) -> dict:
    """Backfill one sector; return a summary dict."""
    sector_fetched = sector_new = sector_dupes = 0

    for keyword in keywords:
        if dry_run:
            print(f"  [DRY-RUN] Would fetch: sector={sector!r} keyword={keyword!r} "
                  f"range={start_date} to {end_date}")
            continue

        run_id = db.log_run_start("backfill", sector, start_date, end_date)
        kw_fetched = kw_new = kw_dupes = 0

        try:
            raw_df = client.fetch_articles([keyword], start_date, end_date)
            processed = process_articles(raw_df, sector, [keyword])

            for _, row in processed.iterrows():
                inserted = db.insert_article(row.to_dict())
                kw_fetched += 1
                if inserted:
                    kw_new += 1
                else:
                    kw_dupes += 1

            # Tone timeline — graceful on rate limit
            tone_df = _fetch_tone_with_backoff(client, keyword, start_date, end_date)
            if tone_df is not None:
                signal = tone_series_to_signal(tone_df)
                if signal["mean"] is not None:
                    db.upsert_tone_signal(
                        {
                            "sector": sector,
                            "keywords": json.dumps([keyword]),
                            "signal_date": end_date,
                            "tone_value": signal["mean"],
                            "tone_class": signal["latest_classification"],
                            "volume_count": len(tone_df) if not tone_df.empty else 0,
                        }
                    )

            db.log_run_complete(run_id, kw_fetched, kw_new, kw_dupes)
            print(
                f"  [{sector}] {keyword:<35} -> "
                f"{kw_fetched} fetched, {kw_new} new, {kw_dupes} dupes"
            )

        except Exception as exc:  # noqa: BLE001
            desc = _exc_desc(exc)
            db.log_run_failed(run_id, desc)
            logger.error("[%s] %r failed: %s", sector, keyword, desc)
            print(f"  [{sector}] {keyword:<35} -> ERROR: {desc}")

        sector_fetched += kw_fetched
        sector_new += kw_new
        sector_dupes += kw_dupes

        time.sleep(_SLEEP_BETWEEN_KEYWORDS)

    return {"sector": sector, "fetched": sector_fetched, "new": sector_new, "dupes": sector_dupes}


def main() -> None:
    """Entry point for the backfill script."""
    args = _parse_args()
    windows = _date_windows(args.days, args.chunk_days)
    overall_start, overall_end = windows[0][0], windows[-1][1]

    if args.sector and args.sector not in FINANCIAL_THEMES:
        print(f"Unknown sector {args.sector!r}. Valid: {list(FINANCIAL_THEMES)}")
        sys.exit(1)

    sectors = (
        {args.sector: FINANCIAL_THEMES[args.sector]}
        if args.sector
        else FINANCIAL_THEMES
    )

    print(
        f"\nJARVIS-Fi Backfill  |  range: {overall_start} to {overall_end}  "
        f"|  {len(windows)} chunk(s) of <={args.chunk_days}d  "
        f"|  sectors: {list(sectors)}  |  dry_run={args.dry_run}\n"
    )

    client = GDELTClient(
        request_delay=settings.GDELT_REQUEST_DELAY_SECONDS,
        language=settings.GDELT_LANGUAGE,
    )

    # Accumulate totals across all chunks keyed by sector.
    totals: dict[str, dict] = {
        s: {"sector": s, "fetched": 0, "new": 0, "dupes": 0} for s in sectors
    }

    if args.dry_run:
        db = None
    else:
        db = Database(settings.DB_PATH)
        db.connect()
        db.init_schema()

    try:
        for chunk_num, (start_date, end_date) in enumerate(windows, 1):
            print(f"\n{'='*60}")
            print(f"CHUNK {chunk_num}/{len(windows)}  |  {start_date} to {end_date}")
            print(f"{'='*60}")

            for sector, keywords in sectors.items():
                print(f"\n  SECTOR: {sector.upper()}  ({len(keywords)} keywords)")
                row = _run_sector(sector, keywords, start_date, end_date, client, db, args.dry_run)
                totals[sector]["fetched"] += row["fetched"]
                totals[sector]["new"] += row["new"]
                totals[sector]["dupes"] += row["dupes"]

        if not args.dry_run:
            for sector in sectors:
                total_in_db = len(db.get_articles_by_sector(sector, limit=99999))
                t = totals[sector]
                print(
                    f"\n  [{sector}] TOTAL: {t['fetched']} fetched, "
                    f"{t['new']} new, {t['dupes']} dupes | DB total: {total_in_db}"
                )
    finally:
        if db:
            db.close()

    print(f"\n{'='*60}")
    print("BACKFILL COMPLETE - SUMMARY")
    print(f"{'='*60}")
    print(
        tabulate(
            [[r["sector"], r["fetched"], r["new"], r["dupes"]] for r in totals.values()],
            headers=["Sector", "Fetched", "New", "Dupes"],
            tablefmt="simple",
        )
    )

    if not args.dry_run:
        with Database(settings.DB_PATH) as fin_db:
            fin_db.init_schema()
            counts = fin_db.row_counts()
        print(f"\nDB row counts: {counts}")

        with Database(settings.DB_PATH) as fin_db:
            fin_db.init_schema()
            conn = fin_db.conn
            print("\nArticles by sector:")
            rows = conn.execute(
                "SELECT sector, COUNT(*) AS cnt FROM articles GROUP BY sector"
            ).fetchall()
            print(tabulate(rows, headers=["sector", "count"], tablefmt="simple"))

            print("\nTone signals by sector:")
            rows = conn.execute(
                "SELECT sector, COUNT(*) AS cnt FROM tone_signals GROUP BY sector"
            ).fetchall()
            print(tabulate(rows, headers=["sector", "count"], tablefmt="simple"))


if __name__ == "__main__":
    main()
