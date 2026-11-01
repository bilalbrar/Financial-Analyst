"""Signal detection: queries Neo4j + SQLite and returns structured signal dicts."""
import logging
from datetime import datetime, timedelta, timezone

from config.themes import FINANCIAL_THEMES
from graph.neo4j_client import Neo4jClient
from storage.database import Database

logger = logging.getLogger(__name__)

# Thresholds
_TONE_DELTA_MEDIUM = 2.0
_TONE_DELTA_HIGH = 5.0
_SPIKE_RATIO_MEDIUM = 2.0   # current > 2x baseline
_SPIKE_RATIO_HIGH = 4.0
_CROSS_SECTOR_MIN = 2       # entity must appear in this many sectors

# ------------------------------------------------------------------
# Cypher queries
# ------------------------------------------------------------------

# Entity mentions in last N days per sector
_ENTITY_COUNTS = """
MATCH (a:Article)-[:BELONGS_TO]->(sec:Sector {name: $sector})
WHERE a.published_at >= $since
MATCH (a)-[:MENTIONS]->(e:Entity)
RETURN e.name AS entity, e.type AS entity_type, count(a) AS mentions
ORDER BY mentions DESC
LIMIT 50
"""

# Entities appearing in 2+ sectors recently
_CROSS_SECTOR = """
MATCH (a:Article)-[:BELONGS_TO]->(sec:Sector)
WHERE a.published_at >= $since
MATCH (a)-[:MENTIONS]->(e:Entity)
WITH e.name AS entity, e.type AS entity_type,
     collect(DISTINCT sec.name) AS sectors, count(a) AS mentions
WHERE size(sectors) >= $min_sectors
RETURN entity, entity_type, sectors, mentions
ORDER BY mentions DESC
LIMIT 30
"""

# Article volume per sector in a date window
_SECTOR_VOLUME = """
MATCH (a:Article)-[:BELONGS_TO]->(sec:Sector {name: $sector})
WHERE a.published_at >= $since
RETURN count(a) AS volume
"""


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d")


def _signal(signal_type, sector, entity_name, entity_type, severity, value, baseline, description, window_days):
    return {
        "signal_type": signal_type,
        "sector": sector,
        "entity_name": entity_name,
        "entity_type": entity_type,
        "severity": severity,
        "value": float(value) if value is not None else None,
        "baseline": float(baseline) if baseline is not None else None,
        "description": description,
        "window_days": window_days,
    }


# ------------------------------------------------------------------
# Signal detectors
# ------------------------------------------------------------------

def _tone_alerts(db: Database) -> list[dict]:
    """Flag sectors where the latest tone is very_negative or very_positive,
    and where it has shifted significantly from the prior reading."""
    results = []
    for sector in FINANCIAL_THEMES:
        df = db.get_tone_series(sector, days=30)
        if df.empty or len(df) < 1:
            continue

        latest_row = df.iloc[-1]
        tone_class = latest_row["tone_class"]
        latest_val = float(latest_row["tone_value"])

        # Absolute class alert
        if tone_class in ("very_negative", "very_positive"):
            severity = "high" if abs(latest_val) > 5 else "medium"
            results.append(_signal(
                "tone_alert", sector, None, None, severity,
                latest_val, None,
                f"{sector} tone is {tone_class} ({latest_val:.2f})",
                1,
            ))

        # Delta vs previous reading
        if len(df) >= 2:
            prior_val = float(df.iloc[-2]["tone_value"])
            delta = latest_val - prior_val
            if abs(delta) >= _TONE_DELTA_HIGH:
                severity = "high"
            elif abs(delta) >= _TONE_DELTA_MEDIUM:
                severity = "medium"
            else:
                continue
            direction = "rose" if delta > 0 else "fell"
            results.append(_signal(
                "tone_delta", sector, None, None, severity,
                latest_val, prior_val,
                f"{sector} tone {direction} by {abs(delta):.2f} (now {latest_val:.2f})",
                1,
            ))

    return results


def _entity_spikes(neo4j: Neo4jClient) -> list[dict]:
    """Compare entity mention counts in the last 7 days vs the prior 7 days.
    Flag entities whose recent count is significantly higher than baseline."""
    results = []
    now = datetime.now(timezone.utc)
    recent_since = _iso(now - timedelta(days=7))
    baseline_since = _iso(now - timedelta(days=14))

    for sector in FINANCIAL_THEMES:
        try:
            # Recent window (last 7 days)
            with neo4j._driver.session(database=neo4j._database) as session:
                recent_rows = session.run(
                    _ENTITY_COUNTS, sector=sector, since=recent_since
                ).data()
                baseline_rows = session.run(
                    _ENTITY_COUNTS, sector=sector, since=baseline_since
                ).data()

            baseline_map = {r["entity"]: r["mentions"] for r in baseline_rows}

            for row in recent_rows:
                entity = row["entity"]
                recent = row["mentions"]
                # baseline is prior 14 days total / 2 to get 7-day equivalent
                prior_total = baseline_map.get(entity, 0)
                baseline = prior_total / 2.0

                if baseline < 1:
                    continue  # not enough history to judge

                ratio = recent / baseline
                if ratio >= _SPIKE_RATIO_HIGH:
                    severity = "high"
                elif ratio >= _SPIKE_RATIO_MEDIUM:
                    severity = "medium"
                else:
                    continue

                results.append(_signal(
                    "entity_spike", sector, entity, row["entity_type"], severity,
                    recent, baseline,
                    f"{entity} mentions in {sector} spiked {ratio:.1f}x vs prior week "
                    f"({recent} vs {baseline:.1f})",
                    7,
                ))
        except Exception as exc:  # noqa: BLE001
            logger.error("entity_spike [%s]: %s", sector, exc)

    return results


def _cross_sector_entities(neo4j: Neo4jClient) -> list[dict]:
    """Detect entities appearing across multiple sectors — contagion signals."""
    results = []
    since = _iso(datetime.now(timezone.utc) - timedelta(days=7))

    try:
        with neo4j._driver.session(database=neo4j._database) as session:
            rows = session.run(
                _CROSS_SECTOR, since=since, min_sectors=_CROSS_SECTOR_MIN
            ).data()

        for row in rows:
            sectors = row["sectors"]
            n_sectors = len(sectors)
            severity = "high" if n_sectors >= 4 else "medium" if n_sectors >= 3 else "low"
            results.append(_signal(
                "cross_sector", None, row["entity"], row["entity_type"], severity,
                float(n_sectors), float(_CROSS_SECTOR_MIN),
                f"{row['entity']} appeared across {n_sectors} sectors: {', '.join(sectors)} "
                f"({row['mentions']} mentions)",
                7,
            ))
    except Exception as exc:  # noqa: BLE001
        logger.error("cross_sector: %s", exc)

    return results


def _volume_spikes(neo4j: Neo4jClient) -> list[dict]:
    """Flag sectors where article volume in the last 7 days is much higher than the prior week."""
    results = []
    now = datetime.now(timezone.utc)
    recent_since = _iso(now - timedelta(days=7))
    baseline_since = _iso(now - timedelta(days=14))

    for sector in FINANCIAL_THEMES:
        try:
            with neo4j._driver.session(database=neo4j._database) as session:
                recent = session.run(_SECTOR_VOLUME, sector=sector, since=recent_since).single()["volume"]
                baseline_total = session.run(_SECTOR_VOLUME, sector=sector, since=baseline_since).single()["volume"]

            baseline = baseline_total / 2.0  # prior 7-day equivalent
            if baseline < 1:
                continue

            ratio = recent / baseline
            if ratio >= _SPIKE_RATIO_HIGH:
                severity = "high"
            elif ratio >= _SPIKE_RATIO_MEDIUM:
                severity = "medium"
            else:
                continue

            results.append(_signal(
                "volume_spike", sector, None, None, severity,
                float(recent), baseline,
                f"{sector} article volume {ratio:.1f}x above prior week ({recent} vs {baseline:.0f})",
                7,
            ))
        except Exception as exc:  # noqa: BLE001
            logger.error("volume_spike [%s]: %s", sector, exc)

    return results


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

def generate_all(db: Database, neo4j: Neo4jClient) -> list[dict]:
    """Run all detectors and return the combined signal list."""
    signals: list[dict] = []
    signals += _tone_alerts(db)
    signals += _entity_spikes(neo4j)
    signals += _cross_sector_entities(neo4j)
    signals += _volume_spikes(neo4j)
    logger.info(
        "Signal detection complete: %d signals (%d tone, %d entity_spike, %d cross_sector, %d volume)",
        len(signals),
        sum(1 for s in signals if s["signal_type"] in ("tone_alert", "tone_delta")),
        sum(1 for s in signals if s["signal_type"] == "entity_spike"),
        sum(1 for s in signals if s["signal_type"] == "cross_sector"),
        sum(1 for s in signals if s["signal_type"] == "volume_spike"),
    )
    return signals
