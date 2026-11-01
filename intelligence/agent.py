"""LangGraph investment research agent for JARVIS-Fi."""
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from config import settings
from embeddings.store import EmbeddingStore
from graph.client import GraphClient
from intelligence.context_assembler import AssembledContext, ContextAssembler, format_for_prompt
from intelligence.prompts import (
    AGENT_CONTEXT_TEMPLATE,
    AGENT_SYSTEM,
    INTENT_EXTRACTION_SYSTEM,
    INTENT_EXTRACTION_USER,
    NO_EVIDENCE_DISCLAIMER,
)
from storage.database import Database

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Data classes
# ------------------------------------------------------------------


@dataclass
class Citation:
    title: str
    url: str
    domain: str
    published_at: str
    relevance_score: float | None = None


@dataclass
class AgentResponse:
    content: str
    answer_short: str
    citations: list[Citation]
    sectors_covered: list[str]
    graph_nodes_queried: int
    articles_retrieved: int
    sources_used: list[str]
    confidence: str
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ------------------------------------------------------------------
# LangGraph state — only serialisable fields
# ------------------------------------------------------------------


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    question: str
    entities: list[str]
    sectors: list[str]
    context: AssembledContext | None
    context_text: str
    response: AgentResponse | None
    error: str | None
    node_trace: list[str]


# ------------------------------------------------------------------
# LLM factory
# ------------------------------------------------------------------


def _get_fast_llm(provider: str = "ollama"):
    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=settings.GROQ_FAST_MODEL,
            api_key=settings.GROQ_API_KEY,
            temperature=0,
        )
    from langchain_ollama import ChatOllama
    return ChatOllama(
        model=settings.OLLAMA_FAST_MODEL,
        base_url=settings.OLLAMA_BASE_URL,
        temperature=0,
    )


def _get_large_llm(provider: str = "ollama"):
    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=settings.GROQ_LARGE_MODEL,
            api_key=settings.GROQ_API_KEY,
            temperature=0.2,
        )
    from langchain_ollama import ChatOllama
    return ChatOllama(
        model=settings.OLLAMA_LARGE_MODEL,
        base_url=settings.OLLAMA_BASE_URL,
        temperature=0.2,
    )


def _log_llm(model: str, prompt: str, response: str, elapsed: float) -> None:
    logger.debug(
        "LLM | model=%s prompt_tokens~%d response_tokens~%d latency=%.2fs",
        model, len(prompt) // 4, len(response) // 4, elapsed,
    )


# ------------------------------------------------------------------
# Conditional routing (module-level — no external dependencies)
# ------------------------------------------------------------------


def route_after_intent(state: AgentState) -> str:
    if state.get("error"):
        return "generate_response"
    question_lower = state["question"].lower()
    greetings = ["hello", "hi", "hey", "thanks", "thank you"]
    if any(g in question_lower for g in greetings) and len(state["question"]) < 30:
        return "generate_response"
    return "assemble_context"


# ------------------------------------------------------------------
# Node builders — return closures that capture external dependencies
# ------------------------------------------------------------------

_DEFAULT_INTENT: dict[str, Any] = {
    "entities": [],
    "sectors": [],
    "time_horizon": "medium_term",
    "question_type": "general",
    "requires_graph": True,
    "requires_history": True,
}


def _make_extract_intent(provider: str):
    def extract_intent(state: AgentState) -> dict:
        trace = list(state.get("node_trace") or []) + ["extract_intent"]
        try:
            llm = _get_fast_llm(provider)
            model_name = (
                settings.GROQ_FAST_MODEL if provider == "groq" else settings.OLLAMA_FAST_MODEL
            )
            prompt = INTENT_EXTRACTION_USER.format(question=state["question"])
            msgs = [
                SystemMessage(content=INTENT_EXTRACTION_SYSTEM),
                HumanMessage(content=prompt),
            ]
            t0 = time.monotonic()
            resp = llm.invoke(msgs)
            elapsed = time.monotonic() - t0
            raw = resp.content if hasattr(resp, "content") else str(resp)
            _log_llm(f"{provider}/{model_name}", prompt, raw, elapsed)

            cleaned = raw.strip().strip("```").strip()
            if cleaned.startswith("json"):
                cleaned = cleaned[4:].strip()
            intent = json.loads(cleaned)
        except Exception as exc:  # noqa: BLE001
            logger.warning("extract_intent failed (%s), using defaults", exc)
            intent = _DEFAULT_INTENT

        return {
            "entities": intent.get("entities") or [],
            "sectors": intent.get("sectors") or [],
            "node_trace": trace,
        }

    return extract_intent


def _make_assemble_context(assembler: ContextAssembler):
    def assemble_context(state: AgentState) -> dict:
        trace = list(state.get("node_trace") or []) + ["assemble_context"]
        try:
            ctx = assembler.assemble(
                question=state["question"],
                entities=state.get("entities") or [],
                sectors=state.get("sectors") or [],
            )
            if not (state.get("entities") or state.get("sectors")):
                ctx_text = (
                    "No specific entities or sectors identified. "
                    "Providing general market context."
                )
            else:
                ctx_text = format_for_prompt(ctx)
            return {"context": ctx, "context_text": ctx_text, "node_trace": trace}
        except Exception as exc:  # noqa: BLE001
            logger.error("assemble_context error: %s", exc)
            return {
                "context": None,
                "context_text": "",
                "error": str(exc),
                "node_trace": trace,
            }

    return assemble_context


def _split_context_sections(ctx_text: str) -> tuple[str, str, str]:
    lines_by: dict[str, list[str]] = {"graph": [], "vector": [], "tone": []}
    current: str | None = None
    for line in ctx_text.splitlines():
        if "KNOWLEDGE GRAPH" in line:
            current = "graph"
        elif "HISTORICAL ARTICLES" in line:
            current = "vector"
        elif "TONE SIGNALS" in line or "SECTOR GRAPH SUMMARY" in line:
            current = "tone"
        if current:
            lines_by[current].append(line)
    return (
        "\n".join(lines_by["graph"]),
        "\n".join(lines_by["vector"]),
        "\n".join(lines_by["tone"]),
    )


def _extract_citations(context: AssembledContext | None) -> list[Citation]:
    if context is None:
        return []
    seen: set[str] = set()
    citations: list[Citation] = []
    for r in context.vectors.results:
        if r.url and r.url not in seen:
            seen.add(r.url)
            citations.append(Citation(r.title, r.url, r.domain, r.published_at, r.relevance_score))
    for ev in context.graph.entity_events:
        url = ev.get("url") or ""
        if url and url not in seen:
            seen.add(url)
            citations.append(
                Citation(
                    ev.get("event_title") or "",
                    url,
                    ev.get("domain") or "",
                    str(ev.get("event_date") or ""),
                    None,
                )
            )
    return citations


def _make_generate_response(provider: str):
    def generate_response(state: AgentState) -> dict:
        trace = list(state.get("node_trace") or []) + ["generate_response"]
        context: AssembledContext | None = state.get("context")
        ctx_text: str = state.get("context_text") or ""

        sources_used: list[str] = []
        if context:
            if context.graph.causal_paths or context.graph.entity_events:
                sources_used.append("graph")
            if context.vectors.results:
                sources_used.append("vectors")
            if any(context.tone.series.values()):
                sources_used.append("tone")

        disclaimer = "" if sources_used else NO_EVIDENCE_DISCLAIMER
        graph_s, vector_s, tone_s = _split_context_sections(ctx_text)
        user_msg = AGENT_CONTEXT_TEMPLATE.format(
            question=state["question"],
            graph_context=graph_s or "No graph evidence available.",
            vector_context=vector_s or "No related articles found.",
            tone_context=tone_s or "No tone signals available.",
        )

        model_name = settings.GROQ_LARGE_MODEL if provider == "groq" else settings.OLLAMA_LARGE_MODEL
        try:
            llm = _get_large_llm(provider)
            msgs = [
                SystemMessage(content=AGENT_SYSTEM + disclaimer),
                HumanMessage(content=user_msg),
            ]
            t0 = time.monotonic()
            resp = llm.invoke(msgs)
            elapsed = time.monotonic() - t0
            content = resp.content if hasattr(resp, "content") else str(resp)
            _log_llm(f"{provider}/{model_name}", user_msg, content, elapsed)
        except Exception as exc:  # noqa: BLE001
            logger.error("generate_response LLM call failed: %s", exc)
            content = (
                f"Unable to generate a response due to an error: {exc}\n\n"
                "Please check your LLM provider configuration."
            )

        citations = _extract_citations(context)
        graph_nodes = (
            len(context.graph.causal_paths) + len(context.graph.entity_events)
            if context else 0
        )
        articles = len(context.vectors.results) if context else 0
        sectors_covered = list(context.sectors) if context else state.get("sectors") or []

        response = AgentResponse(
            content=content,
            answer_short="",
            citations=citations,
            sectors_covered=sectors_covered,
            graph_nodes_queried=graph_nodes,
            articles_retrieved=articles,
            sources_used=sources_used,
            confidence="low",
        )
        return {
            "response": response,
            "messages": [AIMessage(content=content)],
            "node_trace": trace,
        }

    return generate_response


def _format_response_node(state: AgentState) -> dict:
    from intelligence.formatter import ResponseFormatter

    trace = list(state.get("node_trace") or []) + ["format_response"]
    response: AgentResponse | None = state.get("response")

    if response is None:
        placeholder = AgentResponse(
            content="No response was generated.",
            answer_short="No response generated.",
            citations=[],
            sectors_covered=[],
            graph_nodes_queried=0,
            articles_retrieved=0,
            sources_used=[],
            confidence="low",
        )
        return {"response": placeholder, "node_trace": trace}

    try:
        formatted = ResponseFormatter().format(response)
        response.content = formatted
    except Exception as exc:  # noqa: BLE001
        logger.error("format_response failed: %s", exc)

    return {"response": response, "node_trace": trace}


# ------------------------------------------------------------------
# Public class
# ------------------------------------------------------------------


class InvestmentAgent:
    """High-level entry point for the JARVIS-Fi research pipeline."""

    def __init__(
        self,
        db: Database,
        graph_client: GraphClient,
        embeddings: EmbeddingStore,
        llm_provider: str = "ollama",
    ) -> None:
        self._db = db
        self._provider = llm_provider
        assembler = ContextAssembler(db, graph_client, embeddings)

        # Build graph with closures that capture assembler and provider
        graph = StateGraph(AgentState)
        graph.add_node("extract_intent", _make_extract_intent(llm_provider))
        graph.add_node("assemble_context", _make_assemble_context(assembler))
        graph.add_node("generate_response", _make_generate_response(llm_provider))
        graph.add_node("format_response", _format_response_node)

        graph.set_entry_point("extract_intent")
        graph.add_conditional_edges(
            "extract_intent",
            route_after_intent,
            {
                "assemble_context": "assemble_context",
                "generate_response": "generate_response",
            },
        )
        graph.add_edge("assemble_context", "generate_response")
        graph.add_edge("generate_response", "format_response")
        graph.add_edge("format_response", END)

        self._app = graph.compile()
        self._last_trace: list[str] = []

    def chat(
        self,
        message: str,
        history: list[dict] | None = None,
    ) -> AgentResponse:
        """Run the agent pipeline for a user message.

        Args:
            message: The user's question.
            history: Optional prior turns as
                     [{"role": "user"|"assistant", "content": str}, ...].

        Returns:
            Populated AgentResponse.
        """
        messages: list[BaseMessage] = []
        for turn in history or []:
            if turn.get("role") == "user":
                messages.append(HumanMessage(content=turn["content"]))
            elif turn.get("role") == "assistant":
                messages.append(AIMessage(content=turn["content"]))
        messages.append(HumanMessage(content=message))

        initial_state: dict[str, Any] = {
            "messages": messages,
            "question": message,
            "entities": [],
            "sectors": [],
            "context": None,
            "context_text": "",
            "response": None,
            "error": None,
            "node_trace": [],
        }

        result = self._app.invoke(initial_state)
        self._last_trace = result.get("node_trace") or []

        response: AgentResponse = result.get("response") or AgentResponse(
            content="No response generated.",
            answer_short="",
            citations=[],
            sectors_covered=[],
            graph_nodes_queried=0,
            articles_retrieved=0,
            sources_used=[],
            confidence="low",
        )

        try:
            self._db.insert_insight({
                "type": "agent_response",
                "sector": ",".join(response.sectors_covered) or None,
                "title": message[:200],
                "content": response.content[:2000],
                "context_json": json.dumps({
                    "confidence": response.confidence,
                    "graph_nodes": response.graph_nodes_queried,
                    "articles": response.articles_retrieved,
                    "sources": response.sources_used,
                }),
            })
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to log agent response: %s", exc)

        return response

    def get_node_trace(self) -> list[str]:
        """Return the node execution trace from the last chat() call."""
        return list(self._last_trace)
