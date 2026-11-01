"""Minimal Alpha Vantage REST client for cached daily price series.

Deliberately thin: it fetches a daily close series for one instrument and parses
the two free-tier response shapes (commodity endpoints vs TIME_SERIES_DAILY).
Free-tier throttling (the "Information"/"Note" JSON keys) is surfaced as an
explicit MarketRateLimit error so callers can stop instead of caching garbage.
"""
import logging
import time

import requests

from config import settings

logger = logging.getLogger(__name__)

_BASE_URL = "https://www.alphavantage.co/query"


class MarketRateLimit(RuntimeError):
    """Raised when Alpha Vantage returns a throttle/quota message instead of data."""


class MarketClient:
    """Fetches daily close series from Alpha Vantage, one instrument at a time."""

    def __init__(self, api_key: str | None = None, rate_delay: float | None = None) -> None:
        self._key = api_key or settings.ALPHAVANTAGE_API_KEY
        self._delay = rate_delay if rate_delay is not None else settings.ALPHAVANTAGE_RATE_DELAY_SECONDS
        self._last_call = 0.0
        if not self._key:
            raise ValueError("No Alpha Vantage API key configured (AlphaVantage_API_KEY).")

    def _get(self, params: dict) -> dict:
        # Space out calls to respect the free-tier rate limit.
        wait = self._delay - (time.monotonic() - self._last_call)
        if wait > 0:
            time.sleep(wait)
        params["apikey"] = self._key
        resp = requests.get(_BASE_URL, params=params, timeout=30)
        self._last_call = time.monotonic()
        resp.raise_for_status()
        data = resp.json()
        # Free-tier throttle / quota messages
        if "Information" in data or "Note" in data:
            raise MarketRateLimit(str(data.get("Information") or data.get("Note"))[:200])
        if "Error Message" in data:
            raise ValueError(f"Alpha Vantage error: {data['Error Message'][:200]}")
        return data

    def fetch_daily(self, instrument: dict) -> list[tuple[str, float]]:
        """Fetch a daily (date, close) series for one instrument definition.

        Args:
            instrument: A dict from config.market_map.SECTOR_INSTRUMENTS.

        Returns:
            List of (YYYY-MM-DD, close) tuples, oldest first.
        """
        fn = instrument["av_function"]
        kind = instrument["kind"]
        if kind == "commodity":
            params = {"function": fn, "interval": "daily"}
        else:
            params = {
                "function": fn,
                "symbol": instrument["symbol"],
                "outputsize": settings.ALPHAVANTAGE_OUTPUTSIZE,
            }

        data = self._get(params)
        rows = _parse_commodity(data) if kind == "commodity" else _parse_timeseries(data)
        logger.info("Fetched %d daily points for %s", len(rows), instrument["key"])
        return rows

    def fetch_commodity(self, function: str) -> list[tuple[str, float]]:
        """Fetch a free commodity series (WTI/BRENT/NATURAL_GAS) by function name."""
        data = self._get({"function": function, "interval": "daily"})
        return _parse_commodity(data)


def _parse_commodity(data: dict) -> list[tuple[str, float]]:
    out: list[tuple[str, float]] = []
    for row in data.get("data", []):
        val = row.get("value")
        if val in (None, ".", ""):
            continue
        try:
            out.append((row["date"], float(val)))
        except (ValueError, KeyError):
            continue
    out.sort(key=lambda r: r[0])
    return out


def _parse_timeseries(data: dict) -> list[tuple[str, float]]:
    ts = data.get("Time Series (Daily)", {})
    out = [(d, float(v["4. close"])) for d, v in ts.items() if "4. close" in v]
    out.sort(key=lambda r: r[0])
    return out
