"""Tests for storage/database.py using an in-memory SQLite database."""
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from storage.database import Database

_TODAY = datetime.now(timezone.utc).strftime("%Y-%m-%d")


@pytest.fixture()
def db(tmp_path):
    """Return an initialised Database backed by a temp file."""
    path = tmp_path / "test.db"
    database = Database(str(path))
    database.connect()
    database.init_schema()
    yield database
    database.close()


@pytest.fixture()
def sample_article():
    return {
        "id": "abc123def456" + "0" * 52,
        "url": "https://reuters.com/article/test",
        "title": "Test article",
        "domain": "reuters.com",
        "language": "en",
        "source_country": "US",
        "published_at": f"{_TODAY}T09:30:00+00:00",
        "sector": "energy",
        "keywords_matched": '["oil price"]',
        "fetched_at": f"{_TODAY}T10:00:00+00:00",
        "title_word_count": 2,
    }


@pytest.fixture()
def sample_tone_signal():
    return {
        "sector": "energy",
        "keywords": '["oil price"]',
        "signal_date": _TODAY,
        "tone_value": -3.5,
        "tone_class": "negative",
        "volume_count": 42,
    }


# ---------------------------------------------------------------------------
# insert_article
# ---------------------------------------------------------------------------

def test_insert_article_returns_true_for_new(db, sample_article):
    """insert_article returns True when the article is new."""
    assert db.insert_article(sample_article) is True


def test_insert_article_returns_false_for_duplicate(db, sample_article):
    """insert_article returns False when the article already exists."""
    db.insert_article(sample_article)
    assert db.insert_article(sample_article) is False


def test_insert_article_row_is_retrievable(db, sample_article):
    """Inserted article can be retrieved by sector query."""
    db.insert_article(sample_article)
    rows = db.get_articles_by_sector("energy", limit=10)
    assert len(rows) == 1
    assert rows[0]["id"] == sample_article["id"]


# ---------------------------------------------------------------------------
# upsert_tone_signal
# ---------------------------------------------------------------------------

def test_upsert_tone_signal_inserts_new(db, sample_tone_signal):
    """upsert_tone_signal inserts when no conflict exists."""
    db.upsert_tone_signal(sample_tone_signal)
    df = db.get_tone_series("energy", days=30)
    assert not df.empty


def test_upsert_tone_signal_no_error_on_conflict(db, sample_tone_signal):
    """upsert_tone_signal does not raise on duplicate (sector, signal_date)."""
    db.upsert_tone_signal(sample_tone_signal)
    updated = {**sample_tone_signal, "tone_value": 2.0, "tone_class": "positive"}
    db.upsert_tone_signal(updated)  # must not raise

    df = db.get_tone_series("energy", days=30)
    assert float(df.iloc[0]["tone_value"]) == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# get_tone_series
# ---------------------------------------------------------------------------

def test_get_tone_series_returns_dataframe(db, sample_tone_signal):
    """get_tone_series returns a DataFrame with the expected columns."""
    db.upsert_tone_signal(sample_tone_signal)
    df = db.get_tone_series("energy", days=30)
    assert isinstance(df, pd.DataFrame)
    for col in ("signal_date", "tone_value", "tone_class", "volume_count"):
        assert col in df.columns


def test_get_tone_series_empty_for_missing_sector(db):
    """get_tone_series returns an empty DataFrame for an unknown sector."""
    df = db.get_tone_series("nonexistent", days=30)
    assert isinstance(df, pd.DataFrame)
    assert df.empty


# ---------------------------------------------------------------------------
# Run logging
# ---------------------------------------------------------------------------

def test_log_run_start_returns_int(db):
    """log_run_start returns an integer run id."""
    run_id = db.log_run_start("backfill", "energy", "2024-01-01", "2024-01-15")
    assert isinstance(run_id, int)
    assert run_id > 0


def test_log_run_complete_updates_status(db):
    """log_run_complete sets status to 'complete'."""
    run_id = db.log_run_start("backfill", "energy", "2024-01-01", "2024-01-15")
    db.log_run_complete(run_id, articles_fetched=10, articles_new=8, articles_dupes=2)
    row = db.conn.execute(
        "SELECT status, articles_new FROM ingestion_runs WHERE id = ?", (run_id,)
    ).fetchone()
    assert row["status"] == "complete"
    assert row["articles_new"] == 8


def test_log_run_failed_updates_status(db):
    """log_run_failed sets status to 'failed' with an error message."""
    run_id = db.log_run_start("backfill", "energy", "2024-01-01", "2024-01-15")
    db.log_run_failed(run_id, "Connection refused")
    row = db.conn.execute(
        "SELECT status, error_message FROM ingestion_runs WHERE id = ?", (run_id,)
    ).fetchone()
    assert row["status"] == "failed"
    assert "Connection refused" in row["error_message"]
