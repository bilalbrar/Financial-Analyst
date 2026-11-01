"""Focused Neo4j write helpers used by the live scheduler."""
import logging

from graph.neo4j_client import Neo4jClient
from graph.ner import NERProcessor

logger = logging.getLogger(__name__)

_MERGE_ARTICLES = """
UNWIND $rows AS row
MERGE (sec:Sector {name: row.sector})
MERGE (src:Source {domain: row.domain})
MERGE (a:Article {id: row.id})
  ON CREATE SET
    a.url          = row.url,
    a.title        = row.title,
    a.published_at = row.published_at,
    a.sector       = row.sector,
    a.word_count   = row.title_word_count
MERGE (a)-[:BELONGS_TO]->(sec)
MERGE (a)-[:PUBLISHED_BY]->(src)
"""

_MERGE_ENTITIES = """
UNWIND $rows AS row
MATCH (a:Article {id: row.article_id})
UNWIND row.entities AS ent
MERGE (e:Entity {name: ent.name, type: ent.type})
MERGE (a)-[:MENTIONS]->(e)
"""

_MERGE_TONE = """
UNWIND $rows AS row
MERGE (sec:Sector {name: row.sector})
MERGE (t:ToneSignal {sector: row.sector, date: row.signal_date})
  ON CREATE SET
    t.tone_value   = row.tone_value,
    t.tone_class   = row.tone_class,
    t.volume_count = row.volume_count
ON MATCH SET
    t.tone_value   = row.tone_value,
    t.tone_class   = row.tone_class,
    t.volume_count = row.volume_count
MERGE (sec)-[:HAS_SIGNAL]->(t)
"""


def push_articles(
    articles: list[dict],
    client: Neo4jClient,
    ner: NERProcessor,
) -> int:
    """Push a batch of new articles (with NER) to Neo4j. Returns entity mention count."""
    if not articles:
        return 0

    rows = [
        {
            "id": a["id"],
            "url": a.get("url", ""),
            "title": a.get("title", ""),
            "domain": a.get("domain") or "unknown",
            "published_at": str(a.get("published_at", "")),
            "sector": a["sector"],
            "title_word_count": a.get("title_word_count", 0),
        }
        for a in articles
    ]

    client.run(_MERGE_ARTICLES, rows=rows)

    titles = [r["title"] for r in rows]
    entity_lists = ner.extract_batch(titles)
    ent_rows = [
        {"article_id": r["id"], "entities": ents}
        for r, ents in zip(rows, entity_lists)
        if ents
    ]
    if ent_rows:
        client.run(_MERGE_ENTITIES, rows=ent_rows)

    total = sum(len(r["entities"]) for r in ent_rows)
    logger.debug("Pushed %d articles, %d entity mentions to Neo4j", len(articles), total)
    return total


def push_tone_signal(signal: dict, client: Neo4jClient) -> None:
    """Upsert a single tone signal node in Neo4j."""
    client.run(_MERGE_TONE, rows=[signal])
    logger.debug("Pushed tone signal for sector=%s date=%s", signal["sector"], signal["signal_date"])
