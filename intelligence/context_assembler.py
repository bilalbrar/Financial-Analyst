"""Context assembler: pulls graph paths, vector matches, and tone signals into
a single AssembledContext for the LLM synthesis step."""
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from embeddings.store import EmbeddingStore, SearchResult
from graph.client import GraphClient
from graph.queries import (
    get_entity_articles,
    get_sector_articles,
    get_sector_graph_summary,
)
from storage.database import Database

logger = logging.getLogger(__name__)

# Token-budget constants (each limits items fetched from that source)
MAX_GRAPH_PATHS = 8
MAX_VECTOR_RESULTS = 6
MAX_TONE_DATAPOINTS = 30
MAX_GRAPH_EVENTS = 10

# Hard character limit for format_for_prompt output (~3500 tokens)
_CHAR_LIMIT = 11_500


@dataclass
class GraphContext:
    causal_paths: list[dict] = field(default_factory=list)
    entity_events: list[dict] = field(default_factory=list)
    sector_summary: dict[str, dict] = field(default_factory=dict)
    cypher_queries_run: list[str] = field(default_factory=list)


@dataclass
class VectorContext:
    results: list[SearchResult] = field(default_factory=list)
    query_used: str = ""


@dataclass
class ToneContext:
    series: dict[str, list[dict]] = field(default_factory=dict)
    trends: dict[str, str] = field(default_factory=dict)
    anomalies: list[dict] = field(default_factory=list)


@dataclass
class AssembledContext:
    question: str
    entities: list[str]
    sectors: list[str]
    graph: GraphContext
    vectors: VectorContext
    tone: ToneContext
    assembled_at: datetime
    total_evidence_pieces: int


class ContextAssembler:
    """Aggregates evidence from Neo4j, ChromaDB, and SQLite for an LLM query."""

    def __init__(
        self,
        db: Database,
        graph: GraphClient,
        embeddings: EmbeddingStore,
    ) -> None:
        self._db = db
        self._graph = graph
        self._embeddings = embeddings

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def assemble(
        self,
        question: str,
        entities: list[str],
        sectors: list[str],
    ) -> AssembledContext:
        """Gather all context for the question; each source fails independently.

        Args:
            question: The user's question string.
            entities: Extracted entity names (companies, countries, etc.).
            sectors: Relevant sector names.

        Returns:
            Fully populated AssembledContext (some sub-contexts may be empty
            if their source was unreachable).
        """
        graph_ctx = GraphContext()
        vector_ctx = VectorContext()
        tone_ctx = ToneContext()

        try:
            graph_ctx = self._get_graph_context(entities, sectors)
        except Exception as exc:  # noqa: BLE001
            logger.error("_get_graph_context failed: %s", exc)

        try:
            vector_ctx = self._get_vector_context(question, sectors)
        except Exception as exc:  # noqa: BLE001
            logger.error("_get_vector_context failed: %s", exc)

        try:
            tone_ctx = self._get_tone_context(sectors)
        except Exception as exc:  # noqa: BLE001
            logger.error("_get_tone_context failed: %s", exc)

        total = (
            len(graph_ctx.causal_paths)
            + len(vector_ctx.results)
            + len(tone_ctx.anomalies)
        )

        return AssembledContext(
            question=question,
            entities=entities,
            sectors=sectors,
            graph=graph_ctx,
            vectors=vector_ctx,
            tone=tone_ctx,
            assembled_at=datetime.now(timezone.utc),
            total_evidence_pieces=total,
        )

    # ------------------------------------------------------------------
    # Private: graph
    # ------------------------------------------------------------------

    def _get_graph_context(
        self, entities: list[str], sectors: list[str]
    ) -> GraphContext:
        ctx = GraphContext()

        # Query 1 — articles mentioning queried entities
        if entities:
            q1 = (
                "MATCH (a:Article)-[:MENTIONS]->(e:Entity) "
                "WHERE any(name IN $entities WHERE toLower(e.name) CONTAINS toLower(name)) "
                "MATCH (a)-[:BELONGS_TO]->(s:Sector) "
                "RETURN DISTINCT a.id AS id, a.title AS event_title, "
                "a.published_at AS event_date, a.url AS url, a.domain AS domain, "
                "s.name AS sector, 5 AS relevance, null AS causal_claim, "
                "['Article: ' + a.title, 'Sector: ' + s.name] AS path_labels, "
                "2 AS path_length "
                "ORDER BY a.published_at DESC LIMIT $limit"
            )
            ctx.cypher_queries_run.append(q1)
            entity_rows = self._graph.run_query(
                q1, entities=entities, limit=MAX_GRAPH_PATHS
            )
            ctx.causal_paths = entity_rows

        # Query 2 — recent articles in target sectors (if causal_paths < 3)
        if len(ctx.causal_paths) < 3 and sectors:
            cutoff = (
                datetime.now(timezone.utc) - timedelta(days=90)
            ).strftime("%Y-%m-%d")
            q2 = (
                "MATCH (a:Article)-[:BELONGS_TO]->(s:Sector) "
                "WHERE s.name IN $sectors AND a.published_at >= $cutoff_date "
                "RETURN a.id AS id, a.title AS event_title, "
                "a.published_at AS event_date, a.url AS url, a.domain AS domain, "
                "s.name AS sector, 5 AS market_relevance, null AS causal_claim "
                "ORDER BY a.published_at DESC LIMIT $limit"
            )
            ctx.cypher_queries_run.append(q2)
            sector_rows = self._graph.run_query(
                q2,
                sectors=sectors,
                cutoff_date=cutoff,
                limit=MAX_GRAPH_EVENTS,
            )
            ctx.entity_events = sector_rows

            # Merge into causal_paths if still empty
            if not ctx.causal_paths:
                ctx.causal_paths = sector_rows[:MAX_GRAPH_PATHS]

        # Query 3 — sector summaries
        for sector in sectors:
            ctx.sector_summary[sector] = get_sector_graph_summary(self._graph, sector)

        return ctx

    # ------------------------------------------------------------------
    # Private: vectors
    # ------------------------------------------------------------------

    def _get_vector_context(
        self, question: str, sectors: list[str]
    ) -> VectorContext:
        ctx = VectorContext(query_used=question)

        # Search A — unfiltered semantic match
        results_a = self._embeddings.search(question, n_results=6)

        # Search B — sector-filtered (only when a single sector is given)
        results_b: list[SearchResult] = []
        if len(sectors) == 1:
            results_b = self._embeddings.search(
                question,
                n_results=4,
                filters={"sector": sectors[0]},
            )

        # Merge and deduplicate by article_id
        seen: set[str] = set()
        merged: list[SearchResult] = []
        for r in results_a + results_b:
            if r.article_id not in seen:
                seen.add(r.article_id)
                merged.append(r)

        merged.sort(key=lambda r: r.relevance_score, reverse=True)
        ctx.results = merged[:MAX_VECTOR_RESULTS]
        return ctx

    # ------------------------------------------------------------------
    # Private: tone
    # ------------------------------------------------------------------

    def _get_tone_context(self, sectors: list[str]) -> ToneContext:
        ctx = ToneContext()

        for sector in sectors:
            df = self._db.get_tone_series(sector, days=MAX_TONE_DATAPOINTS)
            if df.empty:
                ctx.series[sector] = []
                ctx.trends[sector] = "flat"
                continue

            series_list = [
                {
                    "date": str(row["signal_date"]),
                    "tone": float(row["tone_value"]) if row["tone_value"] is not None else 0.0,
                    "volume": int(row["volume_count"]) if row["volume_count"] is not None else 0,
                }
                for _, row in df.iterrows()
            ]
            ctx.series[sector] = series_list
            ctx.trends[sector] = _compute_trend(series_list)

        # Anomalies from signals table, filtered to requested sectors
        all_anomalies = self._db.get_anomalies(days=60)
        if sectors:
            ctx.anomalies = [
                a for a in all_anomalies
                if a.get("sector") in sectors or a.get("sector") is None
            ]
        else:
            ctx.anomalies = all_anomalies

        return ctx


# ------------------------------------------------------------------
# Prompt formatter
# ------------------------------------------------------------------

def format_for_prompt(context: AssembledContext) -> str:
    """Serialise AssembledContext into a human-readable text block for the LLM.

    Enforces a hard limit of ~11 500 characters (~3 500 tokens). Truncates
    the graph section first, then the vector section, if over budget.

    Args:
        context: Assembled context object.

    Returns:
        Formatted string with labelled sections.
    """
    graph_lines = _format_graph_section(context.graph)
    vector_lines = _format_vector_section(context.vectors)
    tone_lines = _format_tone_section(context.tone)
    summary_lines = _format_summary_section(context.graph)

    sections = [graph_lines, vector_lines, tone_lines, summary_lines]
    full = "\n\n".join(sections)

    if len(full) <= _CHAR_LIMIT:
        return full

    # Iteratively trim graph paths, then vector results
    paths = list(context.graph.causal_paths)
    results = list(context.vectors.results)

    while len(full) > _CHAR_LIMIT and (paths or results):
        if len(paths) > len(results) and paths:
            paths.pop()
        elif results:
            results.pop()
        elif paths:
            paths.pop()

        trimmed_graph = context.graph.__class__(
            causal_paths=paths,
            entity_events=context.graph.entity_events,
            sector_summary=context.graph.sector_summary,
            cypher_queries_run=context.graph.cypher_queries_run,
        )
        trimmed_vectors = context.vectors.__class__(
            results=results, query_used=context.vectors.query_used
        )
        sections = [
            _format_graph_section(trimmed_graph),
            _format_vector_section(trimmed_vectors),
            tone_lines,
            summary_lines,
        ]
        full = "\n\n".join(sections)

    # Hard truncate as final safety net
    return full[:_CHAR_LIMIT]


def _format_graph_section(graph: GraphContext) -> str:
    lines = ["=== KNOWLEDGE GRAPH EVIDENCE ==="]
    lines.append(f"Causal chains found: {len(graph.causal_paths)}")
    for i, path in enumerate(graph.causal_paths[:MAX_GRAPH_PATHS], 1):
        path_str = " → ".join(path.get("path_labels") or [])
        relevance = path.get("relevance") or path.get("market_relevance") or 5
        claim = path.get("causal_claim") or "N/A"
        lines.append(
            f"\n[{i}] {path.get('event_title', 'Unknown')} ({path.get('event_date', '')})\n"
            f"    Path: {path_str}\n"
            f"    Claim: {claim}\n"
            f"    Relevance: {relevance}/10"
        )
    return "\n".join(lines)


def _format_vector_section(vectors: VectorContext) -> str:
    lines = ["=== RELATED HISTORICAL ARTICLES ==="]
    for i, r in enumerate(vectors.results, 1):
        lines.append(
            f"\n[{i}] {r.title}\n"
            f"    Source: {r.domain} | {r.published_at} | Sector: {r.sector}\n"
            f"    Similarity: {r.relevance_score:.2f}"
        )
    return "\n".join(lines)


def _format_tone_section(tone: ToneContext) -> str:
    lines = ["=== SECTOR TONE SIGNALS ==="]
    for sector, series_list in tone.series.items():
        if not series_list:
            lines.append(f"\n{sector}: no tone data available")
            continue
        trend = tone.trends.get(sector, "flat")
        latest = series_list[-1]
        old_row = series_list[-8] if len(series_list) >= 8 else series_list[0]
        latest_val = latest["tone"]
        old_val = old_row["tone"]
        delta = latest_val - old_val
        lines.append(f"\n{sector} (last {len(series_list)} days):")
        lines.append(f"  Current tone: {latest_val:.2f} | Trend: {trend}")
        lines.append(f"  Tone 7d ago: {old_val:.2f} | Delta: {delta:+.2f}")

    if tone.anomalies:
        lines.append("\nRecent alerts:")
        for alert in tone.anomalies[:5]:
            date = str(alert.get("detected_at") or "")[:10]
            content = alert.get("description") or ""
            lines.append(f"  - [{date}] {content}")
    return "\n".join(lines)


def _format_summary_section(graph: GraphContext) -> str:
    if not graph.sector_summary:
        return ""
    lines = ["=== SECTOR GRAPH SUMMARY ==="]
    for sector, summary in graph.sector_summary.items():
        if not isinstance(summary, dict):
            continue
        event_count = summary.get("event_count", 0)
        top_entities = ", ".join(summary.get("top_entities") or []) or "none"
        top_sources = ", ".join(summary.get("top_sources") or []) or "none"
        lines.append(f"{sector}: {event_count} events | Top entities: {top_entities}")
        lines.append(f"  Top sources: {top_sources}")
    return "\n".join(lines)


def _compute_trend(series: list[dict]) -> str:
    """Compare mean of last 7 days against prior 7 days to label the trend."""
    if len(series) < 2:
        return "flat"
    recent = [r["tone"] for r in series[-7:]]
    prior = [r["tone"] for r in series[-14:-7]] if len(series) >= 14 else [r["tone"] for r in series[: len(series) // 2]]
    if not prior:
        return "flat"
    recent_mean = sum(recent) / len(recent)
    prior_mean = sum(prior) / len(prior)
    delta = recent_mean - prior_mean
    if delta > 0.5:
        return "rising"
    if delta < -0.5:
        return "falling"
    return "flat"
