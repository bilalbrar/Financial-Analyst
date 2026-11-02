"""News <-> price correlation engine for JARVIS-Fi.

Two halves:
  * build_*  — refresh-time helpers that join Neo4j entity mentions + SQLite
               article volume onto the cached price series (writes sector_daily).
  * analyse_* — dashboard-time helpers that operate purely on the cached
               sector_daily DataFrame, so page loads never touch Neo4j.
"""
import logging

import numpy as np
import pandas as pd

from config.market_map import SECTOR_DRIVERS, SECTOR_INSTRUMENTS
from graph.client import GraphClient
from storage.database import Database

logger = logging.getLogger(__name__)

SPIKE_PCTILE = 0.80          # a "news-spike" day is at/above this entity-volume pctile
STRONG_R, MODERATE_R = 0.30, 0.15


# ======================================================================
# Build half (refresh time — needs Neo4j)
# ======================================================================

def existing_drivers(graph: GraphClient, names: list[str]) -> list[str]:
    """Return the subset of candidate driver names that exist in the graph."""
    rows = graph.run_query(
        "MATCH (e:Entity) WHERE e.name IN $names RETURN DISTINCT e.name AS n",
        names=names,
    )
    return [r["n"] for r in rows]


def _daily_entity_counts(graph: GraphClient, drivers: list[str]) -> dict[str, int]:
    if not drivers:
        return {}
    rows = graph.run_query(
        """
        MATCH (a:Article)-[:MENTIONS]->(e:Entity)
        WHERE e.name IN $drivers
        RETURN substring(a.published_at, 0, 10) AS day, count(DISTINCT a) AS c
        ORDER BY day
        """,
        drivers=drivers,
    )
    return {r["day"]: r["c"] for r in rows if r["day"]}


def _daily_article_counts(db: Database, sector: str, start: str) -> dict[str, int]:
    rows = db.conn.execute(
        "SELECT substr(published_at,1,10) AS day, COUNT(*) AS c "
        "FROM articles WHERE sector = ? AND published_at >= ? GROUP BY day",
        (sector, start),
    ).fetchall()
    return {r["day"]: r["c"] for r in rows}


def build_sector_daily(
    db: Database, graph: GraphClient, sector: str, start: str
) -> list[dict]:
    """Join cached prices with news activity for a sector into sector_daily rows.

    Args:
        db: Connected Database.
        graph: Connected GraphClient (Neo4j).
        sector: Sector name.
        start: ISO date lower bound.

    Returns:
        List of row dicts ready for Database.replace_sector_daily.
    """
    inst = SECTOR_INSTRUMENTS[sector]
    prices = db.get_prices(inst["key"], since=start)
    if prices.empty:
        logger.warning("No cached prices for %s (%s)", sector, inst["key"])
        return []

    prices = prices.copy()
    prices["ret_pct"] = prices["close"].pct_change() * 100
    prices["abs_ret_pct"] = prices["ret_pct"].abs()

    drivers = existing_drivers(graph, SECTOR_DRIVERS.get(sector, []))
    logger.info("%s drivers in graph: %s", sector, drivers)
    entity_counts = _daily_entity_counts(graph, drivers)
    article_counts = _daily_article_counts(db, sector, start)

    rows: list[dict] = []
    for _, p in prices.iterrows():
        day = str(p["price_date"])[:10]
        rows.append({
            "sector": sector,
            "trade_date": day,
            "instrument": inst["key"],
            "close": float(p["close"]),
            "ret_pct": None if pd.isna(p["ret_pct"]) else float(p["ret_pct"]),
            "abs_ret_pct": None if pd.isna(p["abs_ret_pct"]) else float(p["abs_ret_pct"]),
            "article_vol": int(article_counts.get(day, 0)),
            "entity_vol": int(entity_counts.get(day, 0)),
        })
    return rows


# ======================================================================
# Analyse half (dashboard time — pure pandas on sector_daily)
# ======================================================================

def load_frame(db: Database, sector: str) -> pd.DataFrame:
    """Load the cached sector_daily frame, typed and date-sorted."""
    df = db.get_sector_daily(sector)
    if df.empty:
        return df
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.sort_values("trade_date").reset_index(drop=True)
    return df


def _safe_corr(a: pd.Series, b: pd.Series) -> float | None:
    x = pd.concat([a, b], axis=1).dropna()
    if len(x) < 10 or x.iloc[:, 0].std() == 0 or x.iloc[:, 1].std() == 0:
        return None
    return float(x.iloc[:, 0].corr(x.iloc[:, 1]))


def compute_stats(df: pd.DataFrame) -> dict:
    """Correlations, event study and current-state metrics for one sector."""
    if df.empty or len(df) < 12:
        return {"ok": False, "n": 0 if df.empty else len(df)}

    d = df.copy()
    d["absret_next"] = d["abs_ret_pct"].shift(-1)

    r_same = _safe_corr(d["entity_vol"], d["abs_ret_pct"])
    r_next = _safe_corr(d["entity_vol"], d["absret_next"])
    r_vol = _safe_corr(d["article_vol"], d["abs_ret_pct"])

    thr = float(np.nanpercentile(d["entity_vol"], SPIKE_PCTILE * 100))
    spike = d[d["entity_vol"] >= thr]
    normal = d[d["entity_vol"] < thr]
    spike_absret = float(spike["abs_ret_pct"].mean()) if len(spike) else float("nan")
    normal_absret = float(normal["abs_ret_pct"].mean()) if len(normal) else float("nan")
    amplification = (
        spike_absret / normal_absret
        if normal_absret and not np.isnan(normal_absret) and normal_absret > 0
        else None
    )

    last = d.iloc[-1]
    recent_vol = float(d["ret_pct"].tail(10).std())
    hist_vol = float(d["ret_pct"].std())
    if hist_vol and recent_vol > hist_vol * 1.25:
        regime = "elevated"
    elif hist_vol and recent_vol < hist_vol * 0.75:
        regime = "calm"
    else:
        regime = "normal"

    ret_5d = (
        float((last["close"] / d.iloc[-6]["close"] - 1) * 100) if len(d) >= 6 else None
    )

    if r_same is None:
        confidence = "unknown"
    elif r_same >= STRONG_R:
        confidence = "strong"
    elif r_same >= MODERATE_R:
        confidence = "moderate"
    else:
        confidence = "weak"

    return {
        "ok": True,
        "n": len(d),
        "r_same": r_same,
        "r_next": r_next,
        "r_vol": r_vol,
        "spike_threshold": thr,
        "spike_absret": spike_absret,
        "normal_absret": normal_absret,
        "amplification": amplification,
        "spike_days": int(len(spike)),
        "link_confidence": confidence,
        # current state
        "last_date": str(last["trade_date"].date()),
        "last_close": float(last["close"]),
        "last_ret": None if pd.isna(last["ret_pct"]) else float(last["ret_pct"]),
        "ret_5d": ret_5d,
        "last_entity_vol": int(last["entity_vol"]),
        "spike_today": bool(last["entity_vol"] >= thr),
        "vol_regime": regime,
        "recent_vol": recent_vol,
        "hist_vol": hist_vol,
    }


def detect_signals(sector: str, df: pd.DataFrame, stats: dict) -> list[dict]:
    """Translate stats into plain-English, decision-support signals (no buy/sell)."""
    if not stats.get("ok"):
        return [{
            "sector": sector, "kind": "data", "severity": "low",
            "headline": f"{sector}: insufficient aligned data",
            "detail": f"Only {stats.get('n', 0)} aligned trading days cached.",
        }]

    inst = SECTOR_INSTRUMENTS[sector]["label"]
    out: list[dict] = []

    # 1. News-spike risk (forward-looking volatility warning)
    if stats["spike_today"]:
        amp = stats.get("amplification")
        amp_txt = f"~{amp:.1f}x" if amp else "elevated"
        out.append({
            "sector": sector, "kind": "news_spike",
            "severity": "high" if (amp or 0) >= 2 else "medium",
            "headline": f"Elevated news flow on {inst}",
            "detail": (
                f"{stats['last_entity_vol']} driver-entity articles on "
                f"{stats['last_date']} (≥{int(SPIKE_PCTILE*100)}th pct). "
                f"On similar days {inst} averaged {stats['spike_absret']:.1f}% daily "
                f"moves vs {stats['normal_absret']:.1f}% normal ({amp_txt} volatility)."
            ),
            "value": stats["last_entity_vol"],
        })

    # 2. Volatility regime
    if stats["vol_regime"] == "elevated":
        out.append({
            "sector": sector, "kind": "vol_regime", "severity": "medium",
            "headline": f"{inst} in an elevated-volatility regime",
            "detail": (
                f"10-day realised vol {stats['recent_vol']:.1f}% vs "
                f"{stats['hist_vol']:.1f}% baseline — wider swings than usual."
            ),
            "value": stats["recent_vol"],
        })

    # 3. Divergence: news hot but price quiet (or move unexplained by news)
    quiet = stats.get("ret_5d") is not None and abs(stats["ret_5d"]) < 1.5
    if stats["spike_today"] and quiet:
        out.append({
            "sector": sector, "kind": "divergence", "severity": "medium",
            "headline": f"News-price divergence on {inst}",
            "detail": (
                f"News flow is elevated but {inst} is roughly flat over 5 days "
                f"({stats['ret_5d']:+.1f}%). Watch for a delayed catch-up move."
            ),
            "value": stats["ret_5d"],
        })

    # 4. Link confidence (meta — how much to trust this sector's news link now)
    out.append({
        "sector": sector, "kind": "link_confidence",
        "severity": "info",
        "headline": f"News→price link: {stats['link_confidence']}",
        "detail": (
            (f"r={stats['r_same']:+.2f}" if stats.get("r_same") is not None else "r=n/a")
            + f" between driver-entity news and |{inst} daily move| over {stats['n']} days"
            + (f"; next-day r={stats['r_next']:+.2f}." if stats.get("r_next") is not None else ".")
        ),
        "value": stats["r_same"],
    })
    return out


def top_moves(df: pd.DataFrame, n: int = 8) -> list[dict]:
    """Return the n largest single-day price moves with that day's news counts."""
    if df.empty:
        return []
    d = df.dropna(subset=["abs_ret_pct"])
    top = d.reindex(d["abs_ret_pct"].sort_values(ascending=False).index).head(n)
    return [
        {
            "date": str(r["trade_date"].date()),
            "ret_pct": float(r["ret_pct"]),
            "entity_vol": int(r["entity_vol"]),
            "article_vol": int(r["article_vol"]),
        }
        for _, r in top.iterrows()
    ]


def attribution(db: Database, graph: GraphClient | None, sector: str, date: str) -> dict:
    """Explain a given day: which driver entities spiked + sample headlines.

    Articles always come from SQLite; entity breakdown uses Neo4j when available.
    """
    arts = db.conn.execute(
        "SELECT title, domain, url, published_at FROM articles "
        "WHERE sector = ? AND substr(published_at,1,10) = ? "
        "ORDER BY title_word_count DESC LIMIT 12",
        (sector, date),
    ).fetchall()
    articles = [dict(a) for a in arts]

    entities: list[dict] = []
    if graph is not None:
        try:
            drivers = SECTOR_DRIVERS.get(sector, [])
            rows = graph.run_query(
                """
                MATCH (a:Article)-[:MENTIONS]->(e:Entity)
                WHERE e.name IN $drivers AND substring(a.published_at,0,10) = $date
                RETURN e.name AS name, e.type AS type, count(DISTINCT a) AS mentions
                ORDER BY mentions DESC LIMIT 10
                """,
                drivers=drivers, date=date,
            )
            entities = [dict(r) for r in rows]
        except Exception as exc:  # noqa: BLE001
            logger.warning("attribution entity lookup failed: %s", exc)

    return {"sector": sector, "date": date, "entities": entities, "articles": articles}
