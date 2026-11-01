"""Tests for intelligence/context_assembler.py."""
import pandas as pd
import pytest

from intelligence.context_assembler import (
    AssembledContext,
    ContextAssembler,
    GraphContext,
    ToneContext,
    VectorContext,
    _CHAR_LIMIT,
    format_for_prompt,
)


def _make_search_results(n: int, sector: str = "energy"):
    from embeddings.store import SearchResult
    return [
        SearchResult(
            article_id=f"art-{i}",
            title=f"Article {i} about {sector}",
            url=f"https://news.com/{i}",
            domain="news.com",
            sector=sector,
            published_at=f"2024-01-{i + 1:02d}",
            relevance_score=0.9 - i * 0.05,
        )
        for i in range(n)
    ]


@pytest.fixture()
def mock_graph(mocker):
    g = mocker.MagicMock()
    g.run_query.return_value = [
        {
            "id": "a1",
            "event_title": "OPEC cuts oil production",
            "event_date": "2024-01-10",
            "url": "https://reuters.com/1",
            "domain": "reuters.com",
            "sector": "energy",
            "relevance": 8,
            "causal_claim": "Supply reduction drives price increase",
            "path_labels": ["Article: OPEC cuts oil production", "Sector: energy"],
            "path_length": 2,
        }
    ]
    return g


@pytest.fixture()
def mock_embeddings(mocker):
    e = mocker.MagicMock()
    e.search.return_value = _make_search_results(3)
    return e


@pytest.fixture()
def mock_db(mocker):
    db = mocker.MagicMock()
    db.get_tone_series.return_value = pd.DataFrame(
        [
            {"signal_date": f"2024-01-{i + 1:02d}", "tone_value": -1.5 + i * 0.1,
             "tone_class": "negative", "volume_count": 10 + i}
            for i in range(20)
        ]
    )
    db.get_anomalies.return_value = [
        {
            "signal_type": "tone_alert",
            "sector": "energy",
            "description": "energy tone is very_negative (-6.3)",
            "severity": "high",
            "detected_at": "2024-01-15 08:00:00",
        }
    ]
    return db


def test_assemble_returns_all_fields(mock_db, mock_graph, mock_embeddings):
    assembler = ContextAssembler(mock_db, mock_graph, mock_embeddings)
    ctx = assembler.assemble(
        question="What is the outlook for energy?",
        entities=["OPEC"],
        sectors=["energy"],
    )
    assert isinstance(ctx, AssembledContext)
    assert ctx.question == "What is the outlook for energy?"
    assert ctx.entities == ["OPEC"]
    assert ctx.sectors == ["energy"]
    assert isinstance(ctx.graph, GraphContext)
    assert isinstance(ctx.vectors, VectorContext)
    assert isinstance(ctx.tone, ToneContext)
    assert ctx.assembled_at is not None


def test_graph_failure_does_not_prevent_vectors(mocker, mock_db, mock_embeddings):
    bad_graph = mocker.MagicMock()
    bad_graph.run_query.side_effect = ConnectionError("Neo4j unreachable")

    assembler = ContextAssembler(mock_db, bad_graph, mock_embeddings)
    ctx = assembler.assemble(
        question="What is the energy outlook?",
        entities=["OPEC"],
        sectors=["energy"],
    )
    # Vector results still populated
    assert len(ctx.vectors.results) > 0
    # Graph is empty but didn't raise
    assert ctx.graph.causal_paths == []


def test_format_for_prompt_under_char_limit(mock_db, mock_graph, mock_embeddings):
    assembler = ContextAssembler(mock_db, mock_graph, mock_embeddings)
    ctx = assembler.assemble("energy outlook?", ["OPEC"], ["energy"])
    text = format_for_prompt(ctx)
    assert len(text) <= _CHAR_LIMIT


def test_format_for_prompt_contains_section_headers(mock_db, mock_graph, mock_embeddings):
    assembler = ContextAssembler(mock_db, mock_graph, mock_embeddings)
    ctx = assembler.assemble("energy outlook?", ["OPEC"], ["energy"])
    text = format_for_prompt(ctx)
    assert "KNOWLEDGE GRAPH EVIDENCE" in text
    assert "RELATED HISTORICAL ARTICLES" in text
    assert "SECTOR TONE SIGNALS" in text


def test_format_for_prompt_truncates_large_graph():
    """100 graph events must still produce output under the char limit."""
    from datetime import datetime, timezone
    from embeddings.store import SearchResult

    # Build a context with 100 graph causal paths
    big_paths = [
        {
            "event_title": f"Event {i}: Long title about geopolitics and oil markets impacting supply",
            "event_date": f"2024-01-{(i % 28) + 1:02d}",
            "url": f"https://reuters.com/{i}",
            "domain": "reuters.com",
            "sector": "energy",
            "relevance": 7,
            "causal_claim": f"Claim number {i}: This event caused significant price movement",
            "path_labels": [f"Article: Event {i}", "Sector: energy"],
            "path_length": 2,
        }
        for i in range(100)
    ]

    ctx = AssembledContext(
        question="test",
        entities=[],
        sectors=["energy"],
        graph=GraphContext(
            causal_paths=big_paths,
            entity_events=[],
            sector_summary={"energy": {"event_count": 100, "top_entities": [], "top_sources": []}},
            cypher_queries_run=[],
        ),
        vectors=VectorContext(results=[], query_used="test"),
        tone=ToneContext(series={}, trends={}, anomalies=[]),
        assembled_at=datetime.now(timezone.utc),
        total_evidence_pieces=100,
    )

    text = format_for_prompt(ctx)
    assert len(text) <= _CHAR_LIMIT


def test_tone_context_populated_with_series(mock_db, mock_graph, mock_embeddings):
    assembler = ContextAssembler(mock_db, mock_graph, mock_embeddings)
    ctx = assembler.assemble("energy?", [], ["energy"])
    assert "energy" in ctx.tone.series
    assert len(ctx.tone.series["energy"]) > 0
    assert ctx.tone.trends["energy"] in ("rising", "falling", "flat")
