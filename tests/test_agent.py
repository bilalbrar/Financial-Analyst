"""Tests for intelligence/agent.py — InvestmentAgent and LangGraph nodes."""
import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone

from intelligence.agent import (
    AgentResponse,
    Citation,
    InvestmentAgent,
    route_after_intent,
)


def _make_response(**kwargs) -> AgentResponse:
    defaults = dict(
        content="Oil markets are under pressure.",
        answer_short="Oil markets are under pressure.",
        citations=[],
        sectors_covered=["energy"],
        graph_nodes_queried=3,
        articles_retrieved=2,
        sources_used=["graph", "vectors"],
        confidence="medium",
        generated_at=datetime.now(timezone.utc),
    )
    defaults.update(kwargs)
    return AgentResponse(**defaults)


@pytest.fixture()
def mock_db(mocker):
    db = mocker.MagicMock()
    db.insert_insight.return_value = 1
    return db


@pytest.fixture()
def mock_graph(mocker):
    g = mocker.MagicMock()
    g.run_query.return_value = []
    return g


@pytest.fixture()
def mock_embeddings(mocker):
    e = mocker.MagicMock()
    from embeddings.store import SearchResult
    e.search.return_value = [
        SearchResult(
            article_id="a1",
            title="OPEC cuts production",
            url="https://reuters.com/1",
            domain="reuters.com",
            sector="energy",
            published_at="2024-01-10",
            relevance_score=0.85,
        )
    ]
    e.collection.count.return_value = 10
    return e


def _patch_llms(mocker, intent_json: str, response_text: str):
    """Patch both LLM factories to return mock LLMs."""
    from langchain_core.messages import AIMessage

    fast_llm = MagicMock()
    fast_llm.invoke.return_value = AIMessage(content=intent_json)

    large_llm = MagicMock()
    large_llm.invoke.return_value = AIMessage(content=response_text)

    mocker.patch("intelligence.agent._get_fast_llm", return_value=fast_llm)
    mocker.patch("intelligence.agent._get_large_llm", return_value=large_llm)
    return fast_llm, large_llm


def test_chat_returns_agent_response(mocker, mock_db, mock_graph, mock_embeddings):
    import pandas as pd
    mock_db.get_tone_series.return_value = pd.DataFrame()
    mock_db.get_anomalies.return_value = []

    _patch_llms(
        mocker,
        intent_json='{"entities": ["OPEC"], "sectors": ["energy"], "time_horizon": "short_term", '
                    '"question_type": "outlook", "requires_graph": true, "requires_history": true}',
        response_text="Energy markets face uncertainty due to OPEC decisions.",
    )

    agent = InvestmentAgent(mock_db, mock_graph, mock_embeddings, llm_provider="ollama")
    result = agent.chat("What is the energy outlook?")

    assert isinstance(result, AgentResponse)
    assert result.content  # non-empty


def test_answer_short_populated_and_under_200_chars(mocker, mock_db, mock_graph, mock_embeddings):
    import pandas as pd
    mock_db.get_tone_series.return_value = pd.DataFrame()
    mock_db.get_anomalies.return_value = []

    _patch_llms(
        mocker,
        '{"entities": [], "sectors": ["energy"], "time_horizon": "medium_term", '
        '"question_type": "general", "requires_graph": false, "requires_history": false}',
        "Energy markets remain volatile as supply constraints persist globally.",
    )

    agent = InvestmentAgent(mock_db, mock_graph, mock_embeddings)
    result = agent.chat("Tell me about energy.")
    assert result.answer_short
    assert len(result.answer_short) <= 200


def test_citations_built_from_vector_context(mocker, mock_db, mock_graph, mock_embeddings):
    import pandas as pd
    mock_db.get_tone_series.return_value = pd.DataFrame()
    mock_db.get_anomalies.return_value = []

    _patch_llms(
        mocker,
        '{"entities": ["OPEC"], "sectors": ["energy"], "time_horizon": "short_term", '
        '"question_type": "risk", "requires_graph": true, "requires_history": true}',
        "OPEC production cuts have driven oil prices higher according to recent reports.",
    )

    agent = InvestmentAgent(mock_db, mock_graph, mock_embeddings)
    result = agent.chat("What is the OPEC impact on energy?")
    # mock_embeddings.search returns 1 SearchResult with url set
    assert len(result.citations) >= 1
    assert all(c.url for c in result.citations)


def test_route_after_intent_returns_generate_for_greeting():
    state = {
        "question": "Hello!",
        "entities": [],
        "sectors": [],
        "error": None,
        "node_trace": [],
        "messages": [],
        "context": None,
        "context_text": "",
        "response": None,
    }
    assert route_after_intent(state) == "generate_response"


def test_route_after_intent_returns_assemble_for_real_question():
    state = {
        "question": "What is the investment outlook for semiconductor stocks given US-China tensions?",
        "entities": [],
        "sectors": [],
        "error": None,
        "node_trace": [],
        "messages": [],
        "context": None,
        "context_text": "",
        "response": None,
    }
    assert route_after_intent(state) == "assemble_context"


def test_route_after_intent_skips_to_generate_on_error():
    state = {
        "question": "What about semiconductors?",
        "entities": [],
        "sectors": [],
        "error": "Something broke",
        "node_trace": [],
        "messages": [],
        "context": None,
        "context_text": "",
        "response": None,
    }
    assert route_after_intent(state) == "generate_response"


def test_graph_exception_still_produces_response(mocker, mock_db, mock_graph, mock_embeddings):
    """A Neo4j failure in assemble_context must not prevent a response."""
    import pandas as pd
    mock_db.get_tone_series.return_value = pd.DataFrame()
    mock_db.get_anomalies.return_value = []
    mock_graph.run_query.side_effect = ConnectionError("Neo4j down")

    _patch_llms(
        mocker,
        '{"entities": ["Saudi Arabia"], "sectors": ["energy"], "time_horizon": "medium_term", '
        '"question_type": "outlook", "requires_graph": true, "requires_history": false}',
        "Despite the graph being unavailable, here is an analysis based on available data.",
    )

    agent = InvestmentAgent(mock_db, mock_graph, mock_embeddings)
    result = agent.chat("Saudi Arabia oil impact?")
    assert isinstance(result, AgentResponse)
    assert result.content


def test_node_trace_contains_expected_nodes(mocker, mock_db, mock_graph, mock_embeddings):
    import pandas as pd
    mock_db.get_tone_series.return_value = pd.DataFrame()
    mock_db.get_anomalies.return_value = []

    _patch_llms(
        mocker,
        '{"entities": ["OPEC"], "sectors": ["energy"], "time_horizon": "short_term", '
        '"question_type": "outlook", "requires_graph": true, "requires_history": true}',
        "Energy analysis complete.",
    )

    agent = InvestmentAgent(mock_db, mock_graph, mock_embeddings)
    agent.chat("Energy outlook?")
    trace = agent.get_node_trace()
    assert "extract_intent" in trace
    assert "generate_response" in trace
    assert "format_response" in trace
