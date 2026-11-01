"""Neo4j GraphClient — higher-level wrapper used by the intelligence layer."""
import logging
from typing import Any

from neo4j import GraphDatabase

logger = logging.getLogger(__name__)


class GraphClient:
    """Neo4j driver wrapper exposing run_query, upsert_node, upsert_relationship."""

    def __init__(
        self, uri: str, user: str, password: str, database: str = "neo4j"
    ) -> None:
        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        self._database = database

    def close(self) -> None:
        self._driver.close()

    def __enter__(self) -> "GraphClient":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def run_query(self, cypher: str, **params: Any) -> list[dict]:
        """Run a Cypher query and return all records as plain dicts.

        Args:
            cypher: Cypher query string.
            **params: Named query parameters.

        Returns:
            List of record dicts.
        """
        with self._driver.session(database=self._database) as session:
            result = session.run(cypher, **params)
            return [dict(r) for r in result]

    def upsert_node(
        self,
        label: str,
        match_props: dict[str, Any],
        set_props: dict[str, Any] | None = None,
    ) -> None:
        """MERGE a node; set additional properties only on CREATE.

        Args:
            label: Node label (e.g. 'Article').
            match_props: Properties used in MERGE key (must be indexed/unique).
            set_props: Extra properties to set only when node is newly created.
        """
        set_props = set_props or {}
        match_keys = ", ".join(f"{k}: $m_{k}" for k in match_props)
        on_create = ""
        if set_props:
            on_create = "ON CREATE SET " + ", ".join(f"n.{k} = ${k}" for k in set_props)
        cypher = f"MERGE (n:{label} {{{match_keys}}}) {on_create}"
        params: dict[str, Any] = {f"m_{k}": v for k, v in match_props.items()}
        params.update(set_props)
        with self._driver.session(database=self._database) as session:
            session.run(cypher, **params).consume()

    def upsert_relationship(
        self,
        from_label: str,
        from_match: dict[str, Any],
        to_label: str,
        to_match: dict[str, Any],
        rel_type: str,
        rel_props: dict[str, Any] | None = None,
    ) -> None:
        """MERGE a relationship between two matched nodes.

        Args:
            from_label: Label of the source node.
            from_match: Properties to MATCH the source node.
            to_label: Label of the target node.
            to_match: Properties to MATCH the target node.
            rel_type: Relationship type string (e.g. 'MENTIONS').
            rel_props: Optional properties to SET on the relationship.
        """
        rel_props = rel_props or {}
        f_conds = " AND ".join(f"a.{k} = $af_{k}" for k in from_match)
        t_conds = " AND ".join(f"b.{k} = $bt_{k}" for k in to_match)
        params: dict[str, Any] = {f"af_{k}": v for k, v in from_match.items()}
        params.update({f"bt_{k}": v for k, v in to_match.items()})
        params.update(rel_props)
        set_clause = ""
        if rel_props:
            set_clause = " SET " + ", ".join(f"r.{k} = ${k}" for k in rel_props)
        cypher = (
            f"MATCH (a:{from_label}) WHERE {f_conds} "
            f"MATCH (b:{to_label}) WHERE {t_conds} "
            f"MERGE (a)-[r:{rel_type}]->(b){set_clause}"
        )
        with self._driver.session(database=self._database) as session:
            session.run(cypher, **params).consume()
