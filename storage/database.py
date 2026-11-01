"""SQLite database layer for JARVIS-Fi."""
import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

_SCHEMA_PATH = Path(__file__).parent / "schema.sql"


class Database:
    """SQLite database wrapper for JARVIS-Fi."""

    def __init__(self, db_path: str) -> None:
        """Initialise with path to the SQLite file.

        Args:
            db_path: Filesystem path to the SQLite database file.
        """
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: Optional[sqlite3.Connection] = None

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def connect(self) -> "Database":
        """Open the database connection and apply WAL mode."""
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        logger.debug("Connected to database at %s", self._path)
        return self

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "Database":
        return self.connect()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    @property
    def conn(self) -> sqlite3.Connection:
        """Return active connection, raising if not connected."""
        if self._conn is None:
            raise RuntimeError("Database not connected — call connect() first.")
        return self._conn

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def init_schema(self) -> None:
        """Create all tables and indexes if they do not already exist."""
        sql = _SCHEMA_PATH.read_text(encoding="utf-8")
        self.conn.executescript(sql)
        self.conn.commit()
        logger.info("Schema initialised from %s", _SCHEMA_PATH)

    # ------------------------------------------------------------------
    # Articles
    # ------------------------------------------------------------------

    def insert_article(self, article: dict) -> bool:
        """Insert an article; skip if the id already exists.

        Args:
            article: Dict with keys matching the articles table columns.

        Returns:
            True if the row was inserted (new), False if it was a duplicate.
        """
        sql = """
            INSERT OR IGNORE INTO articles
                (id, url, title, domain, language, source_country,
                 published_at, sector, keywords_matched, fetched_at, title_word_count)
            VALUES
                (:id, :url, :title, :domain, :language, :source_country,
                 :published_at, :sector, :keywords_matched, :fetched_at, :title_word_count)
        """
        cursor = self.conn.execute(sql, article)
        self.conn.commit()
        inserted = cursor.rowcount == 1
        if inserted:
            logger.debug("Inserted article %s", article.get("id"))
        return inserted

    def get_articles_by_sector(self, sector: str, limit: int = 100) -> list[dict]:
        """Return articles for a sector, newest first.

        Args:
            sector: Sector name (e.g. 'energy').
            limit: Maximum rows to return.

        Returns:
            List of row dicts.
        """
        sql = """
            SELECT * FROM articles
            WHERE sector = ?
            ORDER BY published_at DESC
            LIMIT ?
        """
        rows = self.conn.execute(sql, (sector, limit)).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Tone signals
    # ------------------------------------------------------------------

    def insert_tone_signal(self, signal: dict) -> None:
        """Insert a tone signal row (fails on duplicate sector+signal_date).

        Args:
            signal: Dict with keys matching the tone_signals table columns.
        """
        sql = """
            INSERT INTO tone_signals
                (sector, keywords, signal_date, tone_value, tone_class, volume_count)
            VALUES
                (:sector, :keywords, :signal_date, :tone_value, :tone_class, :volume_count)
        """
        self.conn.execute(sql, signal)
        self.conn.commit()

    def upsert_tone_signal(self, signal: dict) -> None:
        """Insert or replace a tone signal on (sector, signal_date) conflict.

        Args:
            signal: Dict with keys matching the tone_signals table columns.
        """
        sql = """
            INSERT INTO tone_signals
                (sector, keywords, signal_date, tone_value, tone_class, volume_count)
            VALUES
                (:sector, :keywords, :signal_date, :tone_value, :tone_class, :volume_count)
            ON CONFLICT(sector, signal_date) DO UPDATE SET
                keywords     = excluded.keywords,
                tone_value   = excluded.tone_value,
                tone_class   = excluded.tone_class,
                volume_count = excluded.volume_count,
                created_at   = CURRENT_TIMESTAMP
        """
        self.conn.execute(sql, signal)
        self.conn.commit()

    def get_tone_series(self, sector: str, days: int = 30) -> pd.DataFrame:
        """Return tone signals for a sector as a DataFrame.

        Args:
            sector: Sector name.
            days: How many past days to retrieve.

        Returns:
            DataFrame with tone signal rows ordered by signal_date asc.
        """
        sql = """
            SELECT signal_date, tone_value, tone_class, volume_count
            FROM tone_signals
            WHERE sector = ?
              AND signal_date >= date('now', ? || ' days')
            ORDER BY signal_date ASC
        """
        rows = self.conn.execute(sql, (sector, f"-{days}")).fetchall()
        if not rows:
            return pd.DataFrame(columns=["signal_date", "tone_value", "tone_class", "volume_count"])
        return pd.DataFrame([dict(r) for r in rows])

    # ------------------------------------------------------------------
    # Ingestion run logging
    # ------------------------------------------------------------------

    def log_run_start(
        self, run_type: str, sector: str, start_date: str, end_date: str
    ) -> int:
        """Record a new ingestion run in running state.

        Args:
            run_type: 'backfill' or 'scheduled'.
            sector: Sector name being ingested.
            start_date: Query start date string.
            end_date: Query end date string.

        Returns:
            The auto-incremented run id.
        """
        sql = """
            INSERT INTO ingestion_runs (run_type, sector, start_date, end_date, status)
            VALUES (?, ?, ?, ?, 'running')
        """
        cursor = self.conn.execute(sql, (run_type, sector, start_date, end_date))
        self.conn.commit()
        return cursor.lastrowid  # type: ignore[return-value]

    def log_run_complete(
        self,
        run_id: int,
        articles_fetched: int,
        articles_new: int,
        articles_dupes: int,
    ) -> None:
        """Mark a run as complete with article counts.

        Args:
            run_id: Run id returned by log_run_start.
            articles_fetched: Total articles fetched from API.
            articles_new: Articles actually inserted.
            articles_dupes: Articles skipped as duplicates.
        """
        sql = """
            UPDATE ingestion_runs
            SET status = 'complete',
                articles_fetched = ?,
                articles_new = ?,
                articles_dupes = ?,
                completed_at = ?
            WHERE id = ?
        """
        now = datetime.now(timezone.utc).isoformat()
        self.conn.execute(sql, (articles_fetched, articles_new, articles_dupes, now, run_id))
        self.conn.commit()

    def log_run_failed(self, run_id: int, error_message: str) -> None:
        """Mark a run as failed with an error message.

        Args:
            run_id: Run id returned by log_run_start.
            error_message: Description of the failure.
        """
        sql = """
            UPDATE ingestion_runs
            SET status = 'failed',
                error_message = ?,
                completed_at = ?
            WHERE id = ?
        """
        now = datetime.now(timezone.utc).isoformat()
        self.conn.execute(sql, (error_message, now, run_id))
        self.conn.commit()

    # ------------------------------------------------------------------
    # Signals
    # ------------------------------------------------------------------

    def insert_signal(self, signal: dict) -> None:
        """Insert a detected signal row.

        Args:
            signal: Dict with keys: signal_type, sector, entity_name, entity_type,
                    severity, value, baseline, description, window_days.
        """
        sql = """
            INSERT INTO signals
                (signal_type, sector, entity_name, entity_type,
                 severity, value, baseline, description, window_days)
            VALUES
                (:signal_type, :sector, :entity_name, :entity_type,
                 :severity, :value, :baseline, :description, :window_days)
        """
        self.conn.execute(sql, signal)
        self.conn.commit()

    def get_recent_signals(self, hours: int = 24) -> list[dict]:
        """Return signals detected within the last N hours.

        Args:
            hours: Lookback window in hours.

        Returns:
            List of signal dicts ordered newest first.
        """
        sql = """
            SELECT * FROM signals
            WHERE detected_at >= datetime('now', ? || ' hours')
            ORDER BY detected_at DESC
        """
        rows = self.conn.execute(sql, (f"-{hours}",)).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Insights
    # ------------------------------------------------------------------

    def insert_insight(self, insight: dict) -> int:
        """Insert a row into the insights table.

        Args:
            insight: Dict with keys: type, sector, title, content, context_json.

        Returns:
            Auto-incremented row id.
        """
        sql = """
            INSERT INTO insights (type, sector, title, content, context_json)
            VALUES (:type, :sector, :title, :content, :context_json)
        """
        cursor = self.conn.execute(sql, insight)
        self.conn.commit()
        return cursor.lastrowid  # type: ignore[return-value]

    def get_insights_by_type(
        self, type_: str, sector: str | None, limit: int = 10
    ) -> list[dict]:
        """Return insights filtered by type (and optionally sector), newest first.

        Args:
            type_: Insight type string (e.g. 'agent_response').
            sector: Optional sector filter. None returns all sectors.
            limit: Maximum rows to return.
        """
        if sector:
            sql = (
                "SELECT * FROM insights WHERE type = ? AND sector = ? "
                "ORDER BY created_at DESC LIMIT ?"
            )
            rows = self.conn.execute(sql, (type_, sector, limit)).fetchall()
        else:
            sql = "SELECT * FROM insights WHERE type = ? ORDER BY created_at DESC LIMIT ?"
            rows = self.conn.execute(sql, (type_, limit)).fetchall()
        return [dict(r) for r in rows]

    def get_anomalies(self, days: int = 60) -> list[dict]:
        """Return high/medium severity signals from the last N days.

        Args:
            days: Lookback window in days.

        Returns:
            List of signal dicts ordered newest first.
        """
        sql = """
            SELECT * FROM signals
            WHERE detected_at >= datetime('now', ? || ' days')
              AND severity IN ('high', 'medium')
            ORDER BY detected_at DESC
        """
        rows = self.conn.execute(sql, (f"-{days}",)).fetchall()
        return [dict(r) for r in rows]

    def row_counts(self) -> dict[str, int]:
        """Return row counts for all main tables.

        Returns:
            Dict mapping table name to count.
        """
        tables = ("articles", "tone_signals", "ingestion_runs", "signals", "insights")
        return {
            t: self.conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]  # noqa: S608
            for t in tables
        }

    # ------------------------------------------------------------------
    # Prices (cached Alpha Vantage daily series)
    # ------------------------------------------------------------------

    def upsert_prices(self, instrument: str, rows: list[tuple[str, float]]) -> int:
        """Insert or replace daily closes for an instrument.

        Args:
            instrument: Instrument key (e.g. 'WTI', 'SOXX').
            rows: List of (price_date_str, close) tuples.

        Returns:
            Number of rows written.
        """
        sql = """
            INSERT INTO prices (instrument, price_date, close)
            VALUES (?, ?, ?)
            ON CONFLICT(instrument, price_date) DO UPDATE SET
                close = excluded.close,
                fetched_at = CURRENT_TIMESTAMP
        """
        params = [(instrument, d, c) for d, c in rows]
        self.conn.executemany(sql, params)
        self.conn.commit()
        return len(params)

    def get_prices(self, instrument: str, since: str | None = None) -> pd.DataFrame:
        """Return cached daily closes for an instrument as a DataFrame.

        Args:
            instrument: Instrument key.
            since: Optional ISO date string lower bound (inclusive).

        Returns:
            DataFrame with columns [price_date, close] ordered ascending.
        """
        if since:
            sql = ("SELECT price_date, close FROM prices WHERE instrument = ? "
                   "AND price_date >= ? ORDER BY price_date ASC")
            rows = self.conn.execute(sql, (instrument, since)).fetchall()
        else:
            sql = ("SELECT price_date, close FROM prices WHERE instrument = ? "
                   "ORDER BY price_date ASC")
            rows = self.conn.execute(sql, (instrument,)).fetchall()
        if not rows:
            return pd.DataFrame(columns=["price_date", "close"])
        return pd.DataFrame([dict(r) for r in rows])

    def latest_price_date(self, instrument: str) -> str | None:
        """Return the most recent cached date for an instrument, or None."""
        row = self.conn.execute(
            "SELECT MAX(price_date) FROM prices WHERE instrument = ?", (instrument,)
        ).fetchone()
        return row[0] if row else None

    # ------------------------------------------------------------------
    # Sector daily (pre-computed news+price join)
    # ------------------------------------------------------------------

    def replace_sector_daily(self, sector: str, rows: list[dict]) -> int:
        """Replace all sector_daily rows for a sector with a fresh computation.

        Args:
            sector: Sector name.
            rows: List of dicts with keys trade_date, instrument, close,
                  ret_pct, abs_ret_pct, article_vol, entity_vol.

        Returns:
            Number of rows written.
        """
        self.conn.execute("DELETE FROM sector_daily WHERE sector = ?", (sector,))
        sql = """
            INSERT INTO sector_daily
                (sector, trade_date, instrument, close, ret_pct, abs_ret_pct,
                 article_vol, entity_vol)
            VALUES
                (:sector, :trade_date, :instrument, :close, :ret_pct, :abs_ret_pct,
                 :article_vol, :entity_vol)
        """
        for r in rows:
            r.setdefault("sector", sector)
        self.conn.executemany(sql, rows)
        self.conn.commit()
        return len(rows)

    def get_sector_daily(self, sector: str) -> pd.DataFrame:
        """Return the pre-computed daily news+price frame for a sector."""
        sql = "SELECT * FROM sector_daily WHERE sector = ? ORDER BY trade_date ASC"
        rows = self.conn.execute(sql, (sector,)).fetchall()
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame([dict(r) for r in rows])
