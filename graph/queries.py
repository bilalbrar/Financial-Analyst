"""Cypher query library for the JARVIS-Fi graph layer."""
import logging

from graph.client import GraphClient

logger = logging.getLogger(__name__)

# Adapted to use Article/Entity/Source/Sector schema (the actual graph schema).

_SECTOR_SUMMARY = """
MATCH (a:Article)-[:BELONGS_TO]->(sec:Sector {name: $sector})
OPTIONAL MATCH (a)-[:MENTIONS]->(e:Entity)
OPTIONAL MATCH (a)-[:PUBLISHED_BY]->(src:Source)
WITH count(DISTINCT a) AS event_count,
     collect(DISTINCT e.name)[..5] AS top_entities,
     collect(DISTINCT src.domain)[..5] AS top_sources
RETURN event_count, top_entities, top_sources
"""

_ENTITY_ARTICLES = """
MATCH (a:Article)-[:MENTIONS]->(e:Entity)
WHERE any(name IN $entities WHERE toLower(e.name) CONTAINS toLower(name))
MATCH (a)-[:BELONGS_TO]->(s:Sector)
RETURN DISTINCT a.id AS id,
       a.title AS event_title,
       a.published_at AS event_date,
       a.url AS url,
       a.domain AS domain,
       s.name AS sector,
       5 AS market_relevance,
       null AS causal_claim,
       ['Article: ' + a.title, 'Sector: ' + s.name] AS path_labels,
       2 AS path_length
ORDER BY a.published_at DESC
LIMIT $limit
"""

_SECTOR_ARTICLES = """
MATCH (a:Article)-[:BELONGS_TO]->(s:Sector)
WHERE s.name IN $sectors
  AND a.published_at >= $cutoff_date
RETURN a.id AS id,
       a.title AS event_title,
       a.published_at AS event_date,
       a.url AS url,
       a.domain AS domain,
       s.name AS sector,
       5 AS market_relevance,
       null AS causal_claim
ORDER BY a.published_at DESC
LIMIT $limit
"""


def get_sector_graph_summary(client: GraphClient, sector: str) -> dict:
    """Return a summary of Article/Entity/Source counts for a sector.

    Args:
        client: Active GraphClient instance.
        sector: Sector name to summarise.

    Returns:
        Dict with keys: sector, event_count, top_entities, top_sources.
    """
    empty = {"sector": sector, "event_count": 0, "top_entities": [], "top_sources": []}
    try:
        rows = client.run_query(_SECTOR_SUMMARY, sector=sector)
        if not rows:
            return empty
        r = rows[0]
        return {
            "sector": sector,
            "event_count": r.get("event_count", 0),
            "top_entities": r.get("top_entities", []),
            "top_sources": r.get("top_sources", []),
        }
    except Exception as exc:  # noqa: BLE001
        logger.error("get_sector_graph_summary [%s]: %s", sector, exc)
        return empty


def get_entity_articles(
    client: GraphClient, entities: list[str], limit: int = 8
) -> list[dict]:
    """Return articles mentioning any of the given entity names.

    Args:
        client: Active GraphClient instance.
        entities: List of entity name strings to search for.
        limit: Maximum results to return.

    Returns:
        List of article record dicts.
    """
    if not entities:
        return []
    try:
        return client.run_query(_ENTITY_ARTICLES, entities=entities, limit=limit)
    except Exception as exc:  # noqa: BLE001
        logger.error("get_entity_articles: %s", exc)
        return []


def get_sector_articles(
    client: GraphClient,
    sectors: list[str],
    cutoff_date: str,
    limit: int = 10,
) -> list[dict]:
    """Return recent articles for the given sectors.

    Args:
        client: Active GraphClient instance.
        sectors: List of sector names.
        cutoff_date: ISO date string; only articles on or after this date.
        limit: Maximum results.

    Returns:
        List of article record dicts.
    """
    if not sectors:
        return []
    try:
        return client.run_query(
            _SECTOR_ARTICLES, sectors=sectors, cutoff_date=cutoff_date, limit=limit
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("get_sector_articles: %s", exc)
        return []
