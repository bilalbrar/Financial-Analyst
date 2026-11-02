"""Groq-backed 'Financial Guru' synthesis for the dashboard.

Deliberately light: it grounds a single Groq chat call in the already-computed
market+news context (correlations, volatility stats, graph entities, headlines).
It does NOT import the LangGraph agent / ChromaDB stack, so the dashboard stays
fast and has no heavy load-time dependencies.
"""
import logging

from config import settings
from config.market_map import ALL_SECTORS, SECTOR_INSTRUMENTS

logger = logging.getLogger(__name__)

GURU_SYSTEM = (
    "You are JARVIS-Fi, a sharp, plain-spoken markets analyst. Answer ONLY from the "
    "CONTEXT provided. The CONTEXT has an ALL-SECTOR SNAPSHOT (summary stats for every "
    "sector) plus DETAILED data for one focused sector. Answer about whichever sector(s) "
    "the question is actually about — if you only have summary stats for it, answer from "
    "those and note that its tab has deeper detail; do NOT claim there is no data when a "
    "summary line exists. This is decision-support about volatility, risk and news "
    "attribution — NEVER give explicit buy/sell calls or price targets. Cite the actual "
    "numbers and entity names from the context. Be concise (a short paragraph plus a few "
    "bullets). Finish with one line 'Risk read: ...'. Note that news flow predicts "
    "volatility better than direction."
)

# Words that hint at a sector when the user types a free-text question.
_SECTOR_ALIASES: dict[str, list[str]] = {
    "energy": ["energy", "oil", "crude", "wti", "brent", "gasoline", "opec", "petrol"],
    "semiconductors": ["semiconductor", "semi", "chip", "nvidia", "intel", "tsmc", "amd", "soxx"],
    "banking": ["bank", "financial", "xlf", "lending", "credit"],
    "commodities": ["commodit", "wheat", "copper", "grain", "dbc", "metal", "agricultur"],
    "geopolitical": ["geopolit", "war", "sanction", "gold", "gld", "conflict", "military"],
    "macro": ["macro", "inflation", "cpi", "gdp", "recession", "unemployment", "spy", "s&p", "economy"],
}


def detect_sector(question: str, default: str) -> str:
    """Pick the sector a free-text question is about; fall back to the dropdown sector."""
    ql = (question or "").lower()
    for s in ALL_SECTORS:           # explicit sector name wins
        if s in ql:
            return s
    for s in ALL_SECTORS:           # then keyword aliases, in sector order
        if any(k in ql for k in _SECTOR_ALIASES.get(s, [])):
            return s
    return default


def llm_available() -> bool:
    """True if a Groq key is configured."""
    return settings.LLM_PROVIDER == "groq" and bool(settings.GROQ_API_KEY)


def _llm():
    from langchain_groq import ChatGroq
    return ChatGroq(
        model=settings.GROQ_LARGE_MODEL,
        api_key=settings.GROQ_API_KEY,
        temperature=0.3,
        max_tokens=650,
    )


def _fmt_r(r) -> str:
    """Format a correlation that may be None (sectors with prices but no news overlap)."""
    return f"{r:+.2f}" if r is not None else "n/a"


def build_global_snapshot(stats_by_sector: dict) -> str:
    """One summary line per sector so the guru can field cross-sector questions."""
    L = ["ALL-SECTOR SNAPSHOT (summary; detailed context for the focused sector follows):"]
    for s, st in stats_by_sector.items():
        inst = SECTOR_INSTRUMENTS[s]["label"]
        if not st.get("ok"):
            L.append(f"- {s} ({inst}): no cached data")
            continue
        amp = st.get("amplification")
        amp_txt = f"{amp:.1f}x" if amp else "n/a"
        L.append(
            f"- {s} ({inst}): link r={_fmt_r(st['r_same'])} ({st['link_confidence']}), "
            f"{amp_txt} vol on news-spike days, regime {st['vol_regime']}, "
            f"last {st['last_close']:.2f} ({st['last_ret']:+.2f}%/1d), "
            f"news-spike today: {'yes' if st['spike_today'] else 'no'}"
        )
    return "\n".join(L)


def build_context(sector, stats, signals, entities, headlines, moves) -> str:
    """Assemble a compact, grounded context block for the LLM."""
    inst = SECTOR_INSTRUMENTS[sector]["label"]
    L = [f"SECTOR: {sector}   INSTRUMENT: {inst}", "", "MARKET & LINK STATS:"]
    if stats.get("ok"):
        amp = stats.get("amplification")
        L += [
            f"- as of {stats['last_date']}: close {stats['last_close']:.2f}, "
            f"1d {stats['last_ret']:+.2f}%"
            + (f", 5d {stats['ret_5d']:+.2f}%" if stats.get("ret_5d") is not None else ""),
            f"- news->price correlation r={_fmt_r(stats['r_same'])} same-day, "
            f"r={_fmt_r(stats['r_next'])} next-day over {stats['n']} trading days "
            f"(link is '{stats['link_confidence']}')",
            f"- on news-spike days the instrument moves {stats['spike_absret']:.1f}% "
            f"vs {stats['normal_absret']:.1f}% normal"
            + (f" ({amp:.1f}x amplification)" if amp else ""),
            f"- volatility regime: {stats['vol_regime']} "
            f"(10d {stats['recent_vol']:.1f}% vs {stats['hist_vol']:.1f}% baseline)",
            f"- today news-spike: {'YES' if stats['spike_today'] else 'no'} "
            f"({stats['last_entity_vol']} driver-entity articles latest day)",
        ]
    else:
        L.append("- insufficient cached data for this sector")

    if signals:
        L += ["", "CURRENT SIGNALS:"]
        L += [f"- [{s['severity']}] {s['headline']}: {s['detail']}" for s in signals]

    if entities:
        L += ["", "TOP NEWS DRIVERS (knowledge graph):"]
        L += [f"- {n} ({t}): {m} mentions" for n, t, m in entities[:6]]

    if moves:
        L += ["", "BIGGEST RECENT MOVES:"]
        L += [f"- {m['date']}: {m['ret_pct']:+.1f}% with {m['entity_vol']} driver articles"
              for m in moves[:5]]

    if headlines:
        L += ["", "RECENT HEADLINES:"]
        L += [f"- {str(h['published_at'])[:10]} {h['domain']}: {h['title'][:110]}"
              for h in headlines[:6]]

    return "\n".join(L)


def ask(question: str, context: str) -> str:
    """Run one grounded Groq completion. Raises on failure (caller handles fallback)."""
    from langchain_core.messages import HumanMessage, SystemMessage
    llm = _llm()
    user = f"CONTEXT:\n{context}\n\nQUESTION: {question}\n\nAnswer:"
    resp = llm.invoke([SystemMessage(content=GURU_SYSTEM), HumanMessage(content=user)])
    return resp.content if hasattr(resp, "content") else str(resp)
