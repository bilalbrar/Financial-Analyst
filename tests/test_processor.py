"""Tests for ingestion/article_processor.py."""
import hashlib
import sqlite3

import pandas as pd
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ingestion.article_processor import (
    _parse_gdelt_date,
    _sha256,
    is_duplicate,
    process_articles,
)


@pytest.fixture()
def raw_df():
    return pd.DataFrame(
        [
            {
                "url": "https://reuters.com/article/test-1",
                "url_mobile": "",
                "title": "Oil price surges on OPEC cut",
                "seendate": "20240115T093000Z",
                "socialimage": "",
                "domain": "reuters.com",
                "language": "English",
                "sourcecountry": "United States",
            }
        ]
    )


@pytest.fixture()
def mem_db():
    """In-memory SQLite DB with the articles table."""
    conn = sqlite3.connect(":memory:")
    conn.execute(
        """
        CREATE TABLE articles (
            id TEXT PRIMARY KEY,
            url TEXT NOT NULL UNIQUE,
            title TEXT,
            domain TEXT,
            language TEXT,
            source_country TEXT,
            published_at DATETIME,
            sector TEXT,
            keywords_matched TEXT,
            fetched_at DATETIME,
            title_word_count INTEGER
        )
        """
    )
    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# SHA-256
# ---------------------------------------------------------------------------

def test_article_id_is_valid_sha256(raw_df):
    """article_id field is a 64-char hex SHA-256 digest."""
    processed = process_articles(raw_df, "energy", ["oil price"])
    article_id = processed.iloc[0]["id"]
    assert len(article_id) == 64
    assert all(c in "0123456789abcdef" for c in article_id)


def test_article_id_matches_sha256_of_url(raw_df):
    """article_id is the SHA-256 of the URL."""
    url = "https://reuters.com/article/test-1"
    expected = hashlib.sha256(url.encode("utf-8")).hexdigest()
    processed = process_articles(raw_df, "energy", ["oil price"])
    assert processed.iloc[0]["id"] == expected


# ---------------------------------------------------------------------------
# Duplicate detection
# ---------------------------------------------------------------------------

def test_is_duplicate_returns_false_for_unknown(mem_db):
    """is_duplicate returns False when the id is not in the DB."""
    assert is_duplicate("nonexistent_id", mem_db) is False


def test_is_duplicate_returns_true_for_known(mem_db, raw_df):
    """is_duplicate returns True after the article is inserted."""
    processed = process_articles(raw_df, "energy", ["oil"])
    row = processed.iloc[0].to_dict()
    mem_db.execute(
        "INSERT INTO articles (id, url, title, domain, language, source_country, "
        "published_at, sector, keywords_matched, fetched_at, title_word_count) "
        "VALUES (:id, :url, :title, :domain, :language, :source_country, "
        ":published_at, :sector, :keywords_matched, :fetched_at, :title_word_count)",
        row,
    )
    mem_db.commit()
    assert is_duplicate(row["id"], mem_db) is True


# ---------------------------------------------------------------------------
# Date parsing
# ---------------------------------------------------------------------------

def test_published_at_parses_gdelt_date_format(raw_df):
    """published_at correctly parses the GDELT YYYYMMDDHHMMSS format."""
    processed = process_articles(raw_df, "energy", ["oil"])
    published_at = processed.iloc[0]["published_at"]
    assert published_at is not None
    assert "2024-01-15" in published_at


def test_parse_gdelt_date_returns_utc():
    """_parse_gdelt_date returns a UTC-aware datetime for both date formats."""
    from datetime import timezone
    for raw in ("20240115093000", "20240115T093000Z"):
        dt = _parse_gdelt_date(raw)
        assert dt is not None, f"Failed to parse {raw!r}"
        assert dt.tzinfo == timezone.utc
        assert dt.year == 2024
        assert dt.month == 1
        assert dt.day == 15


def test_parse_gdelt_date_returns_none_for_invalid():
    """_parse_gdelt_date returns None for an unparseable value."""
    assert _parse_gdelt_date("not-a-date") is None
    assert _parse_gdelt_date("") is None


# ---------------------------------------------------------------------------
# Empty input
# ---------------------------------------------------------------------------

def test_process_articles_returns_empty_for_empty_input():
    """process_articles returns an empty DataFrame for empty input."""
    result = process_articles(pd.DataFrame(), "energy", ["oil"])
    assert result.empty


def test_process_articles_title_word_count(raw_df):
    """title_word_count matches the number of words in the title."""
    processed = process_articles(raw_df, "energy", ["oil"])
    title = raw_df.iloc[0]["title"]
    assert processed.iloc[0]["title_word_count"] == len(title.split())
