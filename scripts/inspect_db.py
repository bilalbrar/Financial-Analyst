"""Quick DB inspection — run at any time, even while backfill is running."""
import io
import sys
from pathlib import Path

# Force UTF-8 output so non-ASCII article titles don't crash on Windows consoles
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tabulate import tabulate

from config import settings
from storage.database import Database

DB = settings.DB_PATH


def _q(conn, sql, *params):
    return conn.execute(sql, params).fetchall()


def main():
    print(f"\nDB path : {DB}")
    print(f"DB size : {Path(DB).stat().st_size / 1024:.1f} KB\n")

    with Database(DB) as db:
        db.init_schema()
        conn = db.conn

        # ── row counts ────────────────────────────────────────────────
        counts = db.row_counts()
        print("=== Table row counts ===")
        print(tabulate(counts.items(), headers=["table", "rows"], tablefmt="simple"))

        # ── articles by sector ────────────────────────────────────────
        print("\n=== Articles by sector ===")
        rows = _q(conn, "SELECT sector, COUNT(*) FROM articles GROUP BY sector ORDER BY 2 DESC")
        print(tabulate(rows, headers=["sector", "articles"], tablefmt="simple"))

        # ── articles by language ──────────────────────────────────────
        print("\n=== Articles by language (top 10) ===")
        rows = _q(conn, "SELECT language, COUNT(*) FROM articles GROUP BY language ORDER BY 2 DESC LIMIT 10")
        print(tabulate(rows, headers=["language", "count"], tablefmt="simple"))

        # ── top domains ───────────────────────────────────────────────
        print("\n=== Top 10 source domains ===")
        rows = _q(conn, "SELECT domain, COUNT(*) FROM articles GROUP BY domain ORDER BY 2 DESC LIMIT 10")
        print(tabulate(rows, headers=["domain", "count"], tablefmt="simple"))

        # ── tone signals ──────────────────────────────────────────────
        print("\n=== Latest tone signals per sector ===")
        rows = _q(conn, """
            SELECT sector, signal_date, ROUND(tone_value, 3) AS tone, tone_class, volume_count
            FROM tone_signals
            WHERE (sector, signal_date) IN (
                SELECT sector, MAX(signal_date) FROM tone_signals GROUP BY sector
            )
            ORDER BY sector
        """)
        print(tabulate(rows, headers=["sector", "date", "tone", "class", "volume"], tablefmt="simple"))

        # ── ingestion run summary ─────────────────────────────────────
        print("\n=== Ingestion run summary ===")
        rows = _q(conn, """
            SELECT run_type, sector, status, COUNT(*) AS runs,
                   SUM(articles_fetched) AS fetched, SUM(articles_new) AS new
            FROM ingestion_runs
            GROUP BY run_type, sector, status
            ORDER BY sector, status
        """)
        print(tabulate(rows, headers=["type", "sector", "status", "runs", "fetched", "new"], tablefmt="simple"))

        # ── recent 5 articles ─────────────────────────────────────────
        print("\n=== 5 most recent articles ===")
        rows = _q(conn, """
            SELECT sector, domain, SUBSTR(title, 1, 60) AS title, published_at
            FROM articles ORDER BY fetched_at DESC LIMIT 5
        """)
        print(tabulate(rows, headers=["sector", "domain", "title", "published_at"], tablefmt="simple"))


if __name__ == "__main__":
    main()
