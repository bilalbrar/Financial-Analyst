"""Tests for intelligence/formatter.py — ResponseFormatter."""
import re
from datetime import datetime, timezone

import pytest

from intelligence.agent import AgentResponse, Citation
from intelligence.formatter import (
    ResponseFormatter,
    _calculate_confidence,
    _extract_short_answer,
)


def _make_response(**kwargs) -> AgentResponse:
    defaults = dict(
        content="Energy markets are experiencing elevated volatility driven by supply constraints.",
        answer_short="",
        citations=[
            Citation(
                title="OPEC cuts production by 1m bpd",
                url="https://reuters.com/opec",
                domain="reuters.com",
                published_at="2024-01-10",
                relevance_score=0.9,
            ),
            Citation(
                title="Saudi Arabia signals continued cuts",
                url="https://ft.com/saudi",
                domain="ft.com",
                published_at="2024-01-12",
                relevance_score=None,
            ),
        ],
        sectors_covered=["energy"],
        graph_nodes_queried=6,
        articles_retrieved=4,
        sources_used=["graph", "vectors", "tone"],
        confidence="low",
        generated_at=datetime.now(timezone.utc),
    )
    defaults.update(kwargs)
    return AgentResponse(**defaults)


@pytest.fixture()
def formatter():
    return ResponseFormatter()


# ------------------------------------------------------------------
# Output structure
# ------------------------------------------------------------------


def test_formatted_output_contains_confidence_badge(formatter):
    r = _make_response()
    out = formatter.format(r)
    assert "Confidence:" in out


def test_formatted_output_contains_evidence_summary(formatter):
    r = _make_response()
    out = formatter.format(r)
    assert "Evidence used" in out
    assert "Knowledge graph:" in out
    assert "Historical articles:" in out


def test_formatted_output_contains_citations(formatter):
    r = _make_response()
    out = formatter.format(r)
    assert "Citations" in out
    assert "reuters.com" in out
    assert "ft.com" in out


def test_formatted_output_contains_footer(formatter):
    r = _make_response()
    out = formatter.format(r)
    assert "Generated" in out
    assert "Sources:" in out


def test_citations_numbered_starting_from_one(formatter):
    r = _make_response()
    out = formatter.format(r)
    assert "1." in out
    assert "2." in out


def test_citations_have_real_urls_not_invented(formatter):
    r = _make_response()
    out = formatter.format(r)
    # Both real URLs from citations must appear
    assert "https://reuters.com/opec" in out
    assert "https://ft.com/saudi" in out


def test_no_unclosed_markdown_brackets(formatter):
    r = _make_response()
    out = formatter.format(r)
    # Every [ must be closed by ]
    open_count = out.count("[")
    close_count = out.count("]")
    assert open_count == close_count


# ------------------------------------------------------------------
# Confidence calculation
# ------------------------------------------------------------------


def test_confidence_high_when_nodes_and_articles_sufficient():
    r = _make_response(graph_nodes_queried=5, articles_retrieved=4)
    assert _calculate_confidence(r) == "high"


def test_confidence_medium_when_either_threshold_met():
    r = _make_response(graph_nodes_queried=2, articles_retrieved=0)
    assert _calculate_confidence(r) == "medium"

    r2 = _make_response(graph_nodes_queried=0, articles_retrieved=2)
    assert _calculate_confidence(r2) == "medium"


def test_confidence_low_when_both_zero():
    r = _make_response(graph_nodes_queried=0, articles_retrieved=0)
    assert _calculate_confidence(r) == "low"


def test_formatter_sets_confidence_on_response(formatter):
    r = _make_response(graph_nodes_queried=8, articles_retrieved=5, confidence="low")
    formatter.format(r)
    assert r.confidence == "high"


# ------------------------------------------------------------------
# Short answer extraction
# ------------------------------------------------------------------


def test_extract_short_answer_skips_headings():
    content = "# Market Outlook\n\nEnergy prices are rising due to supply cuts."
    result = _extract_short_answer(content)
    assert not result.startswith("#")
    assert "Energy prices" in result


def test_extract_short_answer_skips_list_markers():
    content = "- First point about energy\n- Second point\n\nThe outlook is cautious."
    result = _extract_short_answer(content)
    assert not result.startswith("-")


def test_extract_short_answer_under_200_chars():
    content = "Energy markets are under pressure from multiple supply-side shocks."
    result = _extract_short_answer(content)
    assert len(result) <= 200


def test_extract_short_answer_fallback_for_all_headings():
    content = "# Heading One\n# Heading Two\n# Heading Three"
    result = _extract_short_answer(content)
    assert len(result) > 0  # should not crash or return empty


def test_formatter_sets_answer_short(formatter):
    r = _make_response(answer_short="")
    formatter.format(r)
    assert r.answer_short
    assert len(r.answer_short) <= 200
