"""Neo4j connection wrapper for JARVIS-Fi graph layer."""
import logging
from typing import Any

from neo4j import GraphDatabase

logger = logging.getLogger(__name__)

_CONSTRAINTS = [
    "CREATE CONSTRAINT article_id IF NOT EXISTS FOR (a:Article) REQUIRE a.id IS UNIQUE",
    "CREATE CONSTRAINT sector_name IF NOT EXISTS FOR (s:Sector) REQUIRE s.name IS UNIQUE",
    "CREATE CONSTRAINT source_domain IF NOT EXISTS FOR (s:Source) REQUIRE s.domain IS UNIQUE",
    "CREATE CONSTRAINT entity_key IF NOT EXISTS FOR (e:Entity) REQUIRE (e.name, e.type) IS UNIQUE",
    "CREATE CONSTRAINT tone_key IF NOT EXISTS FOR (t:ToneSignal) REQUIRE (t.sector, t.date) IS UNIQUE",
]


class Neo4jClient:
    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j") -> None:
        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        self._database = database

    def close(self) -> None:
        self._driver.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def run(self, cypher: str, **params: Any):
        with self._driver.session(database=self._database) as session:
            return session.run(cypher, **params).consume()

    def setup_constraints(self) -> None:
        for stmt in _CONSTRAINTS:
            self.run(stmt)
        logger.info("Neo4j constraints verified")
