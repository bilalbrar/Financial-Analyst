"""Tone scoring utilities for GDELT tone timeline data."""
import logging

import pandas as pd

logger = logging.getLogger(__name__)


def classify_tone(tone_value: float) -> str:
    """Classify a GDELT tone score into a sentiment label.

    GDELT tone ranges roughly from -100 (very negative) to +100 (very
    positive), with 0 being neutral.

    Args:
        tone_value: Numeric tone score.

    Returns:
        One of: 'very_negative', 'negative', 'neutral', 'positive',
        'very_positive'.
    """
    if tone_value < -5:
        return "very_negative"
    if tone_value < -1:
        return "negative"
    if tone_value <= 1:
        return "neutral"
    if tone_value <= 5:
        return "positive"
    return "very_positive"


def compute_tone_delta(current: float, previous: float) -> float:
    """Return the signed change between two tone values.

    Args:
        current: Most recent tone score.
        previous: Prior tone score.

    Returns:
        Difference (current - previous).
    """
    return current - previous


def tone_series_to_signal(df: pd.DataFrame) -> dict:
    """Summarise a tone timeline DataFrame into a signal dict.

    Args:
        df: DataFrame with at least a 'Value' column containing tone scores,
            as returned by GDELTClient.fetch_tone_timeline().

    Returns:
        Dict with keys: mean, std, min, max, trend, latest_classification.
        Returns a dict of None values if df is empty.
    """
    empty = {
        "mean": None,
        "std": None,
        "min": None,
        "max": None,
        "trend": None,
        "latest_classification": None,
    }

    if df is None or df.empty:
        return empty

    # gdeltdoc timelinetone returns 'Average Tone'; fall back for any future renames
    tone_col_candidates = ["Average Tone", "Value"]
    col = next((c for c in tone_col_candidates if c in df.columns), None)
    if col is None:
        logger.warning("tone_series_to_signal: no recognised tone column in %s", df.columns.tolist())
        return empty
    series = pd.to_numeric(df[col], errors="coerce").dropna()

    if series.empty:
        return empty

    mean_val = float(series.mean())
    std_val = float(series.std()) if len(series) > 1 else 0.0
    min_val = float(series.min())
    max_val = float(series.max())
    latest = float(series.iloc[-1])

    # Trend: compare last third of series against first third
    n = len(series)
    if n >= 3:
        first_mean = series.iloc[: n // 3].mean()
        last_mean = series.iloc[-(n // 3) :].mean()
        delta = last_mean - first_mean
        if delta > 0.5:
            trend = "rising"
        elif delta < -0.5:
            trend = "falling"
        else:
            trend = "flat"
    else:
        trend = "flat"

    return {
        "mean": mean_val,
        "std": std_val,
        "min": min_val,
        "max": max_val,
        "trend": trend,
        "latest_classification": classify_tone(latest),
    }
