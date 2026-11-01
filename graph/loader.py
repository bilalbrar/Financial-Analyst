"""Load articles from SQLite into Neo4j with NER-derived entity nodes."""
import logging
from itertools import islice
from typing import Iterator

from graph.neo4j_client import Neo4jClient
from graph.ner import NERProcessor
from storage.database import Database

logger = logging.getLogger(__name__)

_BATCH_SIZE = 500

# ------------------------------------------------------------------
# Cypher templates (all idempotent via MERGE)
# ------------------------------------------------------------------

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

_MERGE_TONE_SIGNALS = """
UNWIND $rows AS row
MERGE (sec:Sector {name: row.sector})
MERGE (t:ToneSignal {sector: row.sector, date: row.signal_date})
  ON CREATE SET
    t.tone_value   = row.tone_value,
    t.tone_class   = row.tone_class,
    t.volume_count = row.volume_count
MERGE (sec)-[:HAS_SIGNAL]->(t)
"""


def _batched(items: list, n: int) -> Iterator[list]:
    it = iter(items)
    while batch := list(islice(it, n)):
        yield batch


def load(
    db_path: str,
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str,
    neo4j_database: str = "neo4j",
    batch_size: int = _BATCH_SIZE,
) -> dict:
    """Full graph load: articles → sectors → sources → entities → tone signals."""
    ner = NERProcessor()

    with Database(db_path) as db:
        db.init_schema()
        raw_articles = db.conn.execute(
            "SELECT id, url, title, domain, published_at, sector, title_word_count "
            "FROM articles"
        ).fetchall()
        raw_tones = db.conn.execute(
            "SELECT sector, signal_date, tone_value, tone_class, volume_count "
            "FROM tone_signals"
        ).fetchall()

    articles = [
        {
            "id": r[0],
            "url": r[1],
            "title": r[2] or "",
            "domain": r[3] or "unknown",
            "published_at": str(r[4]) if r[4] else None,
            "sector": r[5],
            "title_word_count": r[6] or 0,
        }
        for r in raw_articles
    ]

    tone_rows = [
        {
            "sector": r[0],
            "signal_date": str(r[1]),
            "tone_value": r[2],
            "tone_class": r[3],
            "volume_count": r[4],
        }
        for r in raw_tones
    ]

    total_articles = len(articles)
    total_entities = 0

    with Neo4jClient(neo4j_uri, neo4j_user, neo4j_password, neo4j_database) as client:
        client.setup_constraints()

        # 1 — Article, Sector, Source nodes + relationships
        logger.info("Loading %d articles into Neo4j ...", total_articles)
        for i, batch in enumerate(_batched(articles, batch_size)):
            client.run(_MERGE_ARTICLES, rows=batch)
            logger.info("  articles: %d / %d", min((i + 1) * batch_size, total_articles), total_articles)

        # 2 — NER → Entity nodes + MENTIONS edges
        logger.info("Running NER on article titles ...")
        titles = [a["title"] for a in articles]
        for i, batch in enumerate(_batched(articles, batch_size)):
            title_batch = [a["title"] for a in batch]
            entity_lists = ner.extract_batch(title_batch)

            ent_rows = [
                {"article_id": row["id"], "entities": ents}
                for row, ents in zip(batch, entity_lists)
                if ents
            ]
            if ent_rows:
                client.run(_MERGE_ENTITIES, rows=ent_rows)
                total_entities += sum(len(r["entities"]) for r in ent_rows)

            logger.info(
                "  NER: %d / %d articles processed",
                min((i + 1) * batch_size, total_articles),
                total_articles,
            )

        # 3 — Tone signals
        if tone_rows:
            client.run(_MERGE_TONE_SIGNALS, rows=tone_rows)
            logger.info("Loaded %d tone signals", len(tone_rows))

    summary = {
        "articles": total_articles,
        "tone_signals": len(tone_rows),
        "entity_mentions": total_entities,
    }
    logger.info("Graph load complete: %s", summary)
    return summary
