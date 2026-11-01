"""Markdown response formatter for AgentResponse objects."""
import logging
import re
from datetime import timezone

logger = logging.getLogger(__name__)


class ResponseFormatter:
    """Converts an AgentResponse into consistently structured markdown."""

    def format(self, response) -> str:
        """Build the final markdown string and update response metadata.

        Computes confidence from evidence volume, extracts short answer,
        and assembles the full formatted document.

        Args:
            response: AgentResponse instance (mutated in place for
                      answer_short and confidence fields).

        Returns:
            Formatted markdown string.
        """
        response.confidence = _calculate_confidence(response)
        response.answer_short = self._extract_short_answer(response.content)

        parts = [
            self._build_header(response),
            "---",
            response.content,
            "---",
            self._build_evidence_summary(response),
            self._build_citations(response.citations),
            "---",
            self._build_data_sources(response),
        ]
        return "\n\n".join(p for p in parts if p)

    def _build_header(self, response) -> str:
        badge = self._build_confidence_badge(response.confidence)
        sectors = ", ".join(response.sectors_covered) if response.sectors_covered else "general"
        graph_n = len(response.citations)
        tone_sectors = ", ".join(response.sectors_covered) if response.sectors_covered else "none"
        return (
            f"{badge} · {sectors} ·\n"
            f"> Evidence: {response.graph_nodes_queried} graph paths, "
            f"{response.articles_retrieved} articles, "
            f"tone: {tone_sectors}"
        )

    def _build_confidence_badge(self, confidence: str) -> str:
        label = confidence.upper()
        return f"> **Confidence: {label}**"

    def _build_evidence_summary(self, response) -> str:
        most_recent = ""
        if response.citations:
            dates = [
                c.published_at
                for c in response.citations
                if c.published_at
            ]
            if dates:
                most_recent = f" (most recent: {sorted(dates)[-1][:10]})"

        tone_summary = (
            ", ".join(f"{s}" for s in response.sectors_covered)
            if response.sectors_covered
            else "none"
        )

        lines = [
            "### Evidence used",
            f"- Knowledge graph: {response.graph_nodes_queried} nodes queried"
            + (f" across {', '.join(response.sectors_covered)}" if response.sectors_covered else ""),
            f"- Historical articles: {response.articles_retrieved} retrieved{most_recent}",
            f"- Tone signals: {tone_summary}",
        ]
        return "\n".join(lines)

    def _build_citations(self, citations) -> str:
        if not citations:
            return ""
        lines = ["### Citations"]
        for i, c in enumerate(citations, 1):
            date = c.published_at[:10] if c.published_at else ""
            lines.append(f"{i}. [{c.title}]({c.url}) — {c.domain}, {date}")
        return "\n".join(lines)

    def _build_data_sources(self, response) -> str:
        sources_str = ", ".join(response.sources_used) if response.sources_used else "general knowledge"
        ts = response.generated_at.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        return f"*Generated {ts} · Sources: {sources_str}*"

    def _extract_short_answer(self, content: str) -> str:
        return _extract_short_answer(content)


# ------------------------------------------------------------------
# Module-level helpers (used by formatter and tests directly)
# ------------------------------------------------------------------


def _calculate_confidence(response) -> str:
    """Derive confidence label from evidence volume only (never from LLM output).

    Args:
        response: AgentResponse with graph_nodes_queried and articles_retrieved.

    Returns:
        "high" | "medium" | "low"
    """
    nodes = response.graph_nodes_queried
    articles = response.articles_retrieved
    if nodes >= 5 and articles >= 4:
        return "high"
    if nodes >= 2 or articles >= 2:
        return "medium"
    return "low"


def _extract_short_answer(content: str) -> str:
    """Return the first sentence suitable as a one-line summary.

    Skips markdown headings and list markers. Falls back to the first
    150 characters if no qualifying sentence is found.

    Args:
        content: Full LLM-generated response text.

    Returns:
        Short answer string (20–200 characters).
    """
    skip_patterns = re.compile(r"^(\s*#{1,6}\s|[-*]\s|\d+\.\s)")
    sentence_end = re.compile(r"(?<=[.!?])\s+")

    for paragraph in content.split("\n\n"):
        paragraph = paragraph.strip()
        if not paragraph or skip_patterns.match(paragraph):
            continue
        # Split paragraph into sentences
        sentences = sentence_end.split(paragraph)
        for sentence in sentences:
            sentence = sentence.strip()
            if skip_patterns.match(sentence):
                continue
            if 20 <= len(sentence) <= 200:
                return sentence

    # Fallback
    plain = content.strip()[:150]
    return plain + "..." if len(content.strip()) > 150 else plain
