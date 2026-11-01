"""Tests for ingestion/gdelt_client.py."""
import time
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import requests

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ingestion.gdelt_client import GDELTClient


@pytest.fixture()
def client():
    """Return a GDELTClient with a short request delay for tests."""
    return GDELTClient(request_delay=0.0)


@pytest.fixture()
def sample_article_df():
    return pd.DataFrame(
        [
            {
                "url": "https://reuters.com/article/1",
                "url_mobile": "",
                "title": "Oil price hits record",
                "seendate": "20240101120000",
                "socialimage": "",
                "domain": "reuters.com",
                "language": "English",
                "sourcecountry": "United States",
            }
        ]
    )


# ---------------------------------------------------------------------------
# fetch_articles
# ---------------------------------------------------------------------------

def test_fetch_articles_returns_dataframe(client, sample_article_df):
    """fetch_articles returns a DataFrame with the expected columns."""
    with patch.object(client._gd, "article_search", return_value=sample_article_df):
        result = client.fetch_articles(["oil price"], "2024-01-01", "2024-01-02")

    assert isinstance(result, pd.DataFrame)
    assert "url" in result.columns
    assert "title" in result.columns
    assert len(result) == 1


def test_fetch_articles_returns_empty_on_none(client):
    """fetch_articles returns an empty DataFrame when the API returns None."""
    with patch.object(client._gd, "article_search", return_value=None):
        result = client.fetch_articles(["oil price"], "2024-01-01", "2024-01-02")

    assert isinstance(result, pd.DataFrame)
    assert result.empty


# ---------------------------------------------------------------------------
# Retry on RequestException
# ---------------------------------------------------------------------------

def test_fetch_articles_retries_on_request_exception(client, sample_article_df):
    """fetch_articles retries up to 3 times on RequestException."""
    call_count = 0

    def flaky(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise requests.exceptions.ConnectionError("timeout")
        return sample_article_df

    with patch.object(client._gd, "article_search", side_effect=flaky):
        result = client.fetch_articles(["crude oil"], "2024-01-01", "2024-01-02")

    assert call_count == 3
    assert not result.empty


def test_fetch_articles_raises_after_max_retries(client):
    """fetch_articles raises after exhausting all retries."""
    with patch.object(
        client._gd,
        "article_search",
        side_effect=requests.exceptions.ConnectionError("always fails"),
    ):
        with pytest.raises(requests.exceptions.ConnectionError):
            client.fetch_articles(["crude oil"], "2024-01-01", "2024-01-02")


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------

def test_rate_limiting_adds_delay_between_calls(sample_article_df):
    """Successive calls respect the request delay."""
    delay = 0.1
    client = GDELTClient(request_delay=delay)

    with patch.object(client._gd, "article_search", return_value=sample_article_df):
        t0 = time.monotonic()
        client.fetch_articles(["crude oil"], "2024-01-01", "2024-01-02")
        client.fetch_articles(["natural gas"], "2024-01-01", "2024-01-02")
        elapsed = time.monotonic() - t0

    assert elapsed >= delay
