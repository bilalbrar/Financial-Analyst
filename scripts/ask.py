"""CLI interface for the JARVIS-Fi investment research agent.

Usage:
    python scripts/ask.py "What is the outlook for semiconductor stocks?"
    python scripts/ask.py "Should I invest in energy given OPEC decisions?"
    python scripts/ask.py --sector energy "What caused the 2022 energy crisis?"
    python scripts/ask.py --debug "Tell me about US-China trade tensions"
    python scripts/ask.py --provider groq "What is the macro outlook?"
    python scripts/ask.py --no-color "Oil market summary"
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings
from embeddings.store import EmbeddingStore
from graph.client import GraphClient
from intelligence.agent import InvestmentAgent
from storage.database import Database

# ANSI colour codes
_RESET = "\033[0m"
_BOLD = "\033[1m"
_CYAN = "\033[36m"
_DIM = "\033[2m"
_YELLOW = "\033[33m"
_GREEN = "\033[32m"

_BAR = "━" * 54


def _c(text: str, code: str, use_color: bool) -> str:
    return f"{code}{text}{_RESET}" if use_color else text


def main() -> None:
    parser = argparse.ArgumentParser(description="JARVIS-Fi Research Assistant")
    parser.add_argument("question", nargs="?", default=None, help="Your research question")
    parser.add_argument("--sector", type=str, default=None, metavar="STR",
                        help="Pre-filter context to this sector")
    parser.add_argument("--provider", type=str, default=None,
                        choices=["ollama", "groq"],
                        help="Override LLM provider (default: from .env)")
    parser.add_argument("--debug", action="store_true",
                        help="Print node trace, context, and raw LLM output")
    parser.add_argument("--no-color", action="store_true", dest="no_color",
                        help="Plain text output without ANSI codes")
    args = parser.parse_args()

    if not args.question:
        parser.print_help()
        sys.exit(0)

    use_color = not args.no_color
    provider = args.provider or settings.LLM_PROVIDER

    # Build question, optionally prepending sector context
    question = args.question
    if args.sector:
        question = f"[Sector: {args.sector}] {question}"

    # Initialise dependencies
    try:
        db = Database(settings.DB_PATH)
        db.connect()
        db.init_schema()
    except Exception as exc:
        print(f"Database error: {exc}", file=sys.stderr)
        sys.exit(1)

    try:
        graph_client = GraphClient(
            settings.NEO4J_URI, settings.NEO4J_USER, settings.NEO4J_PASSWORD,
            settings.NEO4J_DATABASE,
        )
    except Exception as exc:
        print(f"Warning: Neo4j unavailable ({exc}). Proceeding without graph context.",
              file=sys.stderr)
        graph_client = None  # type: ignore[assignment]

    try:
        embeddings = EmbeddingStore(
            persist_path=settings.CHROMA_PERSIST_PATH,
            collection_name=settings.CHROMA_COLLECTION_NAME,
        )
    except Exception as exc:
        print(f"Warning: ChromaDB unavailable ({exc}). Proceeding without vector context.",
              file=sys.stderr)
        embeddings = None  # type: ignore[assignment]

    if graph_client is None or embeddings is None:
        # Minimal stub so agent can still run (sources will return empty)
        from unittest.mock import MagicMock
        if graph_client is None:
            graph_client = MagicMock()
            graph_client.run_query.return_value = []
        if embeddings is None:
            embeddings = MagicMock()
            embeddings.search.return_value = []
            embeddings.collection.count.return_value = 0

    agent = InvestmentAgent(db, graph_client, embeddings, llm_provider=provider)

    # Print header
    print(_c(_BAR, _CYAN, use_color))
    print(_c("JARVIS-Fi Research Assistant", _BOLD, use_color))
    print(_c(_BAR, _CYAN, use_color))
    print()

    t0 = time.monotonic()
    try:
        response = agent.chat(question)
    except Exception as exc:
        print(f"Agent error: {exc}", file=sys.stderr)
        db.close()
        sys.exit(1)
    elapsed = time.monotonic() - t0

    if args.debug:
        print(_c("=== NODE TRACE ===", _DIM, use_color))
        print(" → ".join(agent.get_node_trace()))
        print()

    # Main response
    print(response.content)
    print()

    # Footer
    print(_c(_BAR, _CYAN, use_color))
    confidence_color = _GREEN if response.confidence == "high" else _YELLOW
    print(
        _c(f"Graph nodes: {response.graph_nodes_queried}", _DIM, use_color)
        + " | "
        + _c(f"Articles: {response.articles_retrieved}", _DIM, use_color)
        + " | "
        + _c(f"Confidence: {response.confidence.upper()}", confidence_color, use_color)
    )
    print(_c(f"Time: {elapsed:.1f}s", _DIM, use_color))
    print(_c(_BAR, _CYAN, use_color))

    db.close()
    if hasattr(graph_client, "close"):
        try:
            graph_client.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
