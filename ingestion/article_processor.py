"""Parse and enrich raw GDELT article DataFrames."""
import hashlib
import json
import logging
import sqlite3
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# GDELT seendate formats — the API returns either form depending on version
_GDELT_DATE_FMTS = ["%Y%m%d%H%M%S", "%Y%m%dT%H%M%SZ", "%Y%m%dT%H%M%S"]


def _sha256(text: str) -> str:
    """Return the SHA-256 hex digest of a string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _parse_gdelt_date(raw: str) -> Optional[datetime]:
    """Parse a GDELT seendate string to a timezone-aware datetime.

    Handles both legacy YYYYMMDDHHMMSS and modern YYYYMMDDTHHMMSSz formats.

    Args:
        raw: Date string returned by the GDELT API.

    Returns:
        UTC-aware datetime, or None if parsing fails.
    """
    cleaned = str(raw).strip()
    for fmt in _GDELT_DATE_FMTS:
        try:
            dt = datetime.strptime(cleaned, fmt)
            return dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    logger.warning("Could not parse GDELT date: %r", raw)
    return None


def is_duplicate(article_id: str, db_conn: sqlite3.Connection) -> bool:
    """Check whether an article id already exists in the database.

    Args:
        article_id: SHA-256 hash of the article URL.
        db_conn: Open sqlite3 connection.

    Returns:
        True if the article already exists, False otherwise.
    """
    row = db_conn.execute(
        "SELECT 1 FROM articles WHERE id = ? LIMIT 1", (article_id,)
    ).fetchone()
    return row is not None


def process_articles(
    raw_df: pd.DataFrame,
    sector: str,
    keywords_matched: list[str],
) -> pd.DataFrame:
    """Transform a raw GDELT article DataFrame into the enriched schema.

    Args:
        raw_df: DataFrame returned by GDELTClient.fetch_articles().
        sector: Which FINANCIAL_THEMES sector this batch belongs to.
        keywords_matched: The query keywords used to fetch this batch.

    Returns:
        Enriched DataFrame ready for insertion into the articles table.
        Returns an empty DataFrame if raw_df is empty.
    """
    if raw_df is None or raw_df.empty:
        return pd.DataFrame()

    fetched_at = datetime.now(timezone.utc).isoformat()
    keywords_json = json.dumps(keywords_matched)

    records = []
    for _, row in raw_df.iterrows():
        url = str(row.get("url", "")).strip()
        if not url:
            continue

        article_id = _sha256(url)
        title = str(row.get("title", "")).strip()
        published_at = _parse_gdelt_date(row.get("seendate", ""))

        records.append(
            {
                "id": article_id,
                "url": url,
                "title": title,
                "domain": str(row.get("domain", "")).strip() or None,
                "language": str(row.get("language", "en")).strip() or "en",
                "source_country": str(row.get("sourcecountry", "")).strip() or None,
                "published_at": published_at.isoformat() if published_at else None,
                "sector": sector,
                "keywords_matched": keywords_json,
                "fetched_at": fetched_at,
                "title_word_count": len(title.split()) if title else 0,
            }
        )

    if not records:
        return pd.DataFrame()

    return pd.DataFrame(records)
