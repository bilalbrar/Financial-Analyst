"""GDELT DOC 2.0 API client with rate limiting and exponential-backoff retry."""
import functools
import logging
import threading
import time
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
import requests
from gdeltdoc import Filters, GdeltDoc
from gdeltdoc.errors import RateLimitError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)

# gdeltdoc calls requests.get with no timeout, which can hang indefinitely.
# Patch it at module load so every call gets a 60-second read/connect deadline.
_GDELT_TIMEOUT = 60
requests.get = functools.partial(requests.get, timeout=_GDELT_TIMEOUT)


def _retry_decorator():
    """Return a tenacity retry decorator for GDELT requests."""
    return retry(
        retry=retry_if_exception_type((requests.exceptions.RequestException, RateLimitError)),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=10, max=60),
        reraise=True,
    )


class GDELTClient:
    """Wrapper around the gdeltdoc library with rate limiting and retry logic."""

    def __init__(
        self,
        request_delay: float = 2.0,
        language: str = "English",
    ) -> None:
        """Initialise the GDELT client.

        Args:
            request_delay: Minimum seconds between outbound requests.
        """
        self._gd = GdeltDoc()
        self._request_delay = request_delay
        self._language = language
        self._lock = threading.Lock()
        self._last_request_time: float = 0.0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _throttle_and_call(self, fn, *args, **kwargs):
        """Enforce per-request rate limiting and log every call.

        Args:
            fn: Callable to invoke (a gdeltdoc method).
            *args: Positional arguments forwarded to fn.
            **kwargs: Keyword arguments forwarded to fn.

        Returns:
            Whatever fn returns.
        """
        with self._lock:
            elapsed = time.monotonic() - self._last_request_time
            if elapsed < self._request_delay:
                time.sleep(self._request_delay - elapsed)

            ts = datetime.now(timezone.utc).isoformat()
            start = time.monotonic()
            try:
                result = fn(*args, **kwargs)
            finally:
                # Stamp the time even on failure so the next retry (or next
                # keyword) still respects the minimum inter-request delay.
                self._last_request_time = time.monotonic()

            duration_ms = (time.monotonic() - start) * 1000
            size = len(result) if result is not None and hasattr(result, "__len__") else "n/a"
            logger.info(
                "GDELT request | ts=%s | fn=%s | response_size=%s | elapsed_ms=%.0f",
                ts,
                getattr(fn, "__name__", str(fn)),
                size,
                duration_ms,
            )
            return result

    @staticmethod
    def _build_filters(
        keywords: list[str],
        start_date: str,
        end_date: str,
        language: Optional[str] = "English",
    ) -> Filters:
        """Construct a gdeltdoc Filters object.

        Args:
            keywords: One or more keywords (ORed together when multiple).
            start_date: Start date in YYYY-MM-DD format.
            end_date: End date in YYYY-MM-DD format.
            language: GDELT language filter (e.g. "English"). None means no filter.

        Returns:
            Configured Filters instance.
        """
        if len(keywords) == 1:
            keyword_str = keywords[0]
        else:
            keyword_str = " OR ".join(f'"{k}"' for k in keywords)
        # GDELT rejects single-word queries shorter than 5 characters
        if " " not in keyword_str and len(keyword_str) < 5:
            raise ValueError(
                f"Keyword {keyword_str!r} is too short for the GDELT API "
                f"(single-word queries require at least 5 characters)"
            )
        return Filters(
            keyword=keyword_str,
            start_date=start_date,
            end_date=end_date,
            language=language or None,
        )

    @staticmethod
    def _safe_df(result: Optional[pd.DataFrame]) -> pd.DataFrame:
        """Return result or empty DataFrame if None/empty."""
        if result is None or (isinstance(result, pd.DataFrame) and result.empty):
            return pd.DataFrame()
        return result

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_articles(
        self,
        keywords: list[str],
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """Fetch articles matching keywords over a date range.

        Args:
            keywords: Keywords to search for.
            start_date: Start date in YYYY-MM-DD format.
            end_date: End date in YYYY-MM-DD format.

        Returns:
            DataFrame with columns: url, url_mobile, title, seendate,
            socialimage, domain, language, sourcecountry.
            The GDELT API returns at most 250 articles per query; use
            smaller date windows to maximise total yield.
        """
        @_retry_decorator()
        def _call():
            filters = self._build_filters(keywords, start_date, end_date, self._language)
            return self._throttle_and_call(self._gd.article_search, filters)

        return self._safe_df(_call())

    def fetch_tone_timeline(
        self,
        keywords: list[str],
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """Fetch tone timeline for keywords over a date range.

        Args:
            keywords: Keywords to search for.
            start_date: Start date in YYYY-MM-DD format.
            end_date: End date in YYYY-MM-DD format.

        Returns:
            DataFrame with columns: datetime, Value (average tone score).
        """
        @_retry_decorator()
        def _call():
            filters = self._build_filters(keywords, start_date, end_date, self._language)
            return self._throttle_and_call(self._gd.timeline_search, "timelinetone", filters)

        return self._safe_df(_call())

    def fetch_volume_timeline(
        self,
        keywords: list[str],
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """Fetch volume timeline for keywords over a date range.

        Args:
            keywords: Keywords to search for.
            start_date: Start date in YYYY-MM-DD format.
            end_date: End date in YYYY-MM-DD format.

        Returns:
            DataFrame with volume-over-time data.
        """
        @_retry_decorator()
        def _call():
            filters = self._build_filters(keywords, start_date, end_date)
            return self._throttle_and_call(self._gd.timeline_search, "timelinevolAngle", filters)

        return self._safe_df(_call())

    def fetch_by_domain(
        self,
        domain: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """Fetch articles from a specific source domain.

        Args:
            domain: Source domain to filter by (e.g. 'reuters.com').
            start_date: Start date in YYYY-MM-DD format.
            end_date: End date in YYYY-MM-DD format.

        Returns:
            DataFrame with article data from that domain.
        """
        @_retry_decorator()
        def _call():
            filters = Filters(domain=domain, start_date=start_date, end_date=end_date)
            return self._throttle_and_call(self._gd.article_search, filters)

        return self._safe_df(_call())
