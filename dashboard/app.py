"""JARVIS-Fi — local 'Financial Guru' dashboard (Plotly Dash).

Reads only from the local SQLite cache (sector_daily / prices / articles) so page
loads are fast and never call Alpha Vantage. Run `scripts/refresh_market.py` to
update the underlying data. Neo4j is used lazily, only for entity attribution.

    python dashboard/app.py    ->    http://127.0.0.1:8050
"""
import sys
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dash_table, dcc, html
from plotly.subplots import make_subplots

from config import settings
from config.market_map import ALL_SECTORS, SECTOR_DRIVERS, SECTOR_INSTRUMENTS
from graph.client import GraphClient
from intelligence import guru
from signals.market_correlation import (
    SPIKE_PCTILE,
    attribution,
    compute_stats,
    detect_signals,
    load_frame,
    top_moves,
)
from storage.database import Database

# ---- theme ----
BG, PANEL, FG, MUTED = "#0e1117", "#1a1f2b", "#e6e6e6", "#8b95a7"
ACCENT = "#4da3ff"
SEV = {"high": "#e74c3c", "medium": "#e67e22", "low": "#7f8c8d", "info": "#3498db"}
UP, DOWN = "#2ecc71", "#e74c3c"

CARD = {"backgroundColor": PANEL, "borderRadius": "10px", "padding": "16px",
        "margin": "8px", "border": f"1px solid #2a3142"}


@contextmanager
def db_conn():
    db = Database(settings.DB_PATH).connect()
    try:
        yield db
    finally:
        db.close()


# ======================================================================
# Data helpers
# ======================================================================

def all_stats() -> dict:
    out = {}
    with db_conn() as db:
        for sector in ALL_SECTORS:
            df = load_frame(db, sector)
            out[sector] = {"df": df, "stats": compute_stats(df)}
    return out


def all_signals(stats_by_sector: dict) -> list[dict]:
    sigs = []
    for sector, blob in stats_by_sector.items():
        if not blob["df"].empty:
            sigs += detect_signals(sector, blob["df"], blob["stats"])
    order = {"high": 0, "medium": 1, "low": 2, "info": 3}
    return sorted(sigs, key=lambda s: order.get(s["severity"], 9))


@lru_cache(maxsize=16)
def top_entities(sector: str) -> tuple:
    try:
        g = GraphClient(settings.NEO4J_URI, settings.NEO4J_USER,
                        settings.NEO4J_PASSWORD, settings.NEO4J_DATABASE)
        rows = g.run_query(
            "MATCH (a:Article)-[:MENTIONS]->(e:Entity) WHERE e.name IN $drivers "
            "RETURN e.name AS name, e.type AS type, count(DISTINCT a) AS mentions "
            "ORDER BY mentions DESC LIMIT 12",
            drivers=SECTOR_DRIVERS.get(sector, []),
        )
        g.close()
        return tuple((r["name"], r["type"], r["mentions"]) for r in rows)
    except Exception:  # noqa: BLE001
        return tuple()


def last_refreshed() -> str:
    with db_conn() as db:
        row = db.conn.execute("SELECT MAX(computed_at) FROM sector_daily").fetchone()
    return (row[0] or "never")[:19] if row else "never"


# ======================================================================
# Figures
# ======================================================================

def fig_price(df, label) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28],
                        vertical_spacing=0.05)
    fig.add_trace(go.Scatter(x=df["trade_date"], y=df["close"], mode="lines",
                             name=label, line=dict(color=ACCENT, width=1.8)), row=1, col=1)

    if df["entity_vol"].max() > 0:
        thr = df["entity_vol"].quantile(SPIKE_PCTILE)
        sp = df[df["entity_vol"] >= thr]
        colors = [UP if r >= 0 else DOWN for r in sp["ret_pct"].fillna(0)]
        fig.add_trace(go.Scatter(
            x=sp["trade_date"], y=sp["close"], mode="markers", name="news spike",
            marker=dict(size=(8 + sp["entity_vol"] / max(sp["entity_vol"].max(), 1) * 18),
                        color=colors, line=dict(width=1, color="#fff"), opacity=0.85),
            customdata=sp[["entity_vol", "ret_pct"]],
            hovertemplate="%{x|%Y-%m-%d}<br>%{customdata[0]} driver articles"
                          "<br>move %{customdata[1]:+.2f}%<extra></extra>",
        ), row=1, col=1)

    fig.add_trace(go.Bar(x=df["trade_date"], y=df["entity_vol"], name="driver articles",
                         marker_color="#5b6b8c"), row=2, col=1)
    fig.update_layout(template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=BG,
                      height=520, margin=dict(l=40, r=20, t=30, b=30),
                      legend=dict(orientation="h", y=1.08), showlegend=True)
    fig.update_yaxes(title_text="price", row=1, col=1)
    fig.update_yaxes(title_text="news/day", row=2, col=1)
    return fig


def fig_bars(pairs, title, fmt="{:.2f}", good_high=True) -> go.Figure:
    pairs = [p for p in pairs if p[1] is not None]
    pairs.sort(key=lambda p: p[1], reverse=True)
    labels = [p[0] for p in pairs]
    vals = [p[1] for p in pairs]
    colors = [ACCENT if (v >= 0) == good_high else "#e67e22" for v in vals]
    fig = go.Figure(go.Bar(x=vals, y=labels, orientation="h", marker_color=colors,
                           text=[fmt.format(v) for v in vals], textposition="auto"))
    fig.update_layout(template="plotly_dark", paper_bgcolor=PANEL, plot_bgcolor=PANEL,
                      height=300, margin=dict(l=10, r=10, t=40, b=20), title=title)
    return fig


# ======================================================================
# Tab renderers
# ======================================================================

def _metric(label, value, color=FG):
    return html.Div([
        html.Div(label, style={"color": MUTED, "fontSize": "12px"}),
        html.Div(value, style={"color": color, "fontSize": "22px", "fontWeight": 600}),
    ], style={**CARD, "flex": "1", "minWidth": "150px"})


def render_overview():
    data = all_stats()
    amp = [(s, data[s]["stats"].get("amplification")) for s in ALL_SECTORS]
    link = [(s, data[s]["stats"].get("r_same")) for s in ALL_SECTORS]

    rows = []
    for s in ALL_SECTORS:
        st = data[s]["stats"]
        if not st.get("ok"):
            rows.append({"sector": s, "instrument": SECTOR_INSTRUMENTS[s]["key"],
                         "last": "—", "1d %": "—", "5d %": "—", "link": "no data",
                         "spike today": "—", "amplification": "—"})
            continue
        rows.append({
            "sector": s, "instrument": st["last_close"] and SECTOR_INSTRUMENTS[s]["key"],
            "last": f"{st['last_close']:.2f}",
            "1d %": f"{st['last_ret']:+.2f}" if st["last_ret"] is not None else "—",
            "5d %": f"{st['ret_5d']:+.2f}" if st["ret_5d"] is not None else "—",
            "link": st["link_confidence"],
            "spike today": "YES" if st["spike_today"] else "no",
            "amplification": f"{st['amplification']:.1f}x" if st["amplification"] else "—",
        })

    sigs = all_signals(data)
    feed = [html.Div([
        html.Span(s["severity"].upper(), style={
            "backgroundColor": SEV.get(s["severity"], MUTED), "color": "#fff",
            "padding": "2px 8px", "borderRadius": "6px", "fontSize": "11px",
            "marginRight": "8px"}),
        html.Span(s["headline"], style={"fontWeight": 600}),
        html.Div(s["detail"], style={"color": MUTED, "fontSize": "13px", "marginTop": "3px"}),
    ], style={**CARD, "margin": "6px 8px"}) for s in sigs]

    return html.Div([
        html.Div([
            _metric("Sectors tracked", str(len(ALL_SECTORS))),
            _metric("Active signals", str(len([s for s in sigs if s["severity"] != "info"])),
                    color=ACCENT),
            _metric("News spikes today",
                    str(sum(1 for s in ALL_SECTORS if data[s]["stats"].get("spike_today"))),
                    color="#e67e22"),
        ], style={"display": "flex", "flexWrap": "wrap"}),
        html.Div([
            html.Div(dcc.Graph(figure=fig_bars(
                amp, "News→volatility amplification (spike vs normal day)", "{:.1f}x")),
                style={**CARD, "flex": "1", "minWidth": "380px"}),
            html.Div(dcc.Graph(figure=fig_bars(
                link, "News→price link strength  (r)", "{:+.2f}")),
                style={**CARD, "flex": "1", "minWidth": "380px"}),
        ], style={"display": "flex", "flexWrap": "wrap"}),
        html.H3("Cross-sector snapshot", style={"margin": "16px 8px 4px"}),
        html.Div(dash_table.DataTable(
            data=rows,
            columns=[{"name": c, "id": c} for c in
                     ["sector", "instrument", "last", "1d %", "5d %", "link",
                      "spike today", "amplification"]],
            style_as_list_view=True,
            style_header={"backgroundColor": PANEL, "color": FG, "fontWeight": "bold"},
            style_cell={"backgroundColor": BG, "color": FG, "border": "1px solid #2a3142",
                        "padding": "8px", "textAlign": "center"},
        ), style={**CARD}),
        html.H3("Decision-support signals", style={"margin": "16px 8px 4px"}),
        html.Div(feed if feed else html.Div("Run scripts/refresh_market.py to populate.",
                                            style={"color": MUTED, "padding": "12px"})),
    ])


def render_deepdive():
    return html.Div([
        dcc.Dropdown(id="dd-sector", options=[{"label": s, "value": s} for s in ALL_SECTORS],
                     value="energy", clearable=False,
                     style={"width": "320px", "color": "#000"}),
        html.Div(id="dd-content"),
    ])


def render_whymove():
    return html.Div([
        dcc.Dropdown(id="wm-sector", options=[{"label": s, "value": s} for s in ALL_SECTORS],
                     value="energy", clearable=False,
                     style={"width": "320px", "color": "#000"}),
        html.P("Biggest daily moves — click a row to see the news behind it.",
               style={"color": MUTED}),
        html.Div(id="wm-table"),
        html.Div(id="wm-attr"),
    ])


def render_guru():
    engine = (f"🧠 Groq · {settings.GROQ_LARGE_MODEL}" if guru.llm_available()
              else "⚙️ computed brief (no LLM key configured)")
    return html.Div([
        html.Div([
            dcc.Dropdown(id="guru-sector",
                         options=[{"label": s, "value": s} for s in ALL_SECTORS],
                         value="energy", clearable=False,
                         style={"width": "320px", "color": "#000"}),
            html.Span(engine, style={"color": MUTED, "marginLeft": "12px", "fontSize": "13px"}),
        ], style={"display": "flex", "alignItems": "center"}),
        dcc.Textarea(id="guru-q",
                     value="What's the current risk picture for this sector and what news is driving it?",
                     style={"width": "100%", "height": "70px", "marginTop": "10px",
                            "color": "#000", "borderRadius": "6px", "padding": "8px"}),
        html.Button("Ask the Guru", id="guru-btn", n_clicks=0,
                    style={"margin": "10px 0", "padding": "8px 18px",
                           "backgroundColor": ACCENT, "border": "none",
                           "borderRadius": "8px", "color": "#fff", "cursor": "pointer"}),
        dcc.Loading(html.Div(id="guru-out", style={**CARD})),
    ])


# ======================================================================
# App
# ======================================================================

app = Dash(__name__, suppress_callback_exceptions=True, title="JARVIS-Fi Guru")
server = app.server

app.layout = html.Div([
    html.Div([
        html.H1("⚡ JARVIS-Fi — Financial Guru", style={"margin": "0", "color": FG}),
        html.Div([
            "News × market intelligence over your knowledge graph + Alpha Vantage   ·   ",
            html.Span(id="refreshed", style={"color": MUTED}),
        ], style={"color": MUTED, "fontSize": "13px", "marginTop": "4px"}),
    ], style={"padding": "16px 20px", "borderBottom": f"2px solid {ACCENT}"}),
    dcc.Tabs(id="tabs", value="overview", children=[
        dcc.Tab(label="Market Overview", value="overview"),
        dcc.Tab(label="Sector Deep-Dive", value="deepdive"),
        dcc.Tab(label="Why Did It Move?", value="whymove"),
        dcc.Tab(label="Ask the Guru", value="guru"),
    ], colors={"border": "#2a3142", "primary": ACCENT, "background": PANEL}),
    html.Div(id="tab-content", style={"padding": "8px"}),
], style={"backgroundColor": BG, "minHeight": "100vh", "fontFamily": "Segoe UI, sans-serif"})


@app.callback(Output("tab-content", "children"), Output("refreshed", "children"),
              Input("tabs", "value"))
def _render_tab(tab):
    refreshed = f"data refreshed: {last_refreshed()}"
    if tab == "overview":
        return render_overview(), refreshed
    if tab == "deepdive":
        return render_deepdive(), refreshed
    if tab == "whymove":
        return render_whymove(), refreshed
    if tab == "guru":
        return render_guru(), refreshed
    return html.Div(), refreshed


@app.callback(Output("dd-content", "children"), Input("dd-sector", "value"))
def _deepdive(sector):
    with db_conn() as db:
        df = load_frame(db, sector)
        stats = compute_stats(df)
    label = SECTOR_INSTRUMENTS[sector]["label"]
    if df.empty or not stats.get("ok"):
        return html.Div(f"No cached data for {sector}. Run scripts/refresh_market.py.",
                        style={"color": MUTED, "padding": "20px"})

    cards = html.Div([
        _metric("Last close", f"{stats['last_close']:.2f}"),
        _metric("1-day", f"{stats['last_ret']:+.2f}%" if stats["last_ret"] is not None else "—",
                UP if (stats["last_ret"] or 0) >= 0 else DOWN),
        _metric("Same-day link r", f"{stats['r_same']:+.2f}" if stats["r_same"] is not None else "—"),
        _metric("Next-day link r", f"{stats['r_next']:+.2f}" if stats["r_next"] is not None else "—"),
        _metric("Vol amplification",
                f"{stats['amplification']:.1f}x" if stats["amplification"] else "—", "#e67e22"),
        _metric("Vol regime", stats["vol_regime"]),
    ], style={"display": "flex", "flexWrap": "wrap"})

    ents = top_entities(sector)
    ent_items = [html.Li(f"{n} ({t}) — {m} mentions", style={"color": FG})
                 for n, t, m in ents] or [html.Li("graph unavailable", style={"color": MUTED})]

    with db_conn() as db:
        heads = db.conn.execute(
            "SELECT title, domain, url, published_at FROM articles WHERE sector=? "
            "ORDER BY published_at DESC LIMIT 8", (sector,)).fetchall()
    head_items = [html.Li(html.A(f"{h['title']} — {h['domain']}", href=h["url"], target="_blank",
                                 style={"color": ACCENT, "textDecoration": "none"}))
                  for h in heads]

    return html.Div([
        cards,
        html.Div(dcc.Graph(figure=fig_price(df, label)), style={**CARD}),
        html.Div([
            html.Div([html.H4("Top driving entities (knowledge graph)",
                              style={"color": FG}), html.Ul(ent_items)],
                     style={**CARD, "flex": "1", "minWidth": "320px"}),
            html.Div([html.H4("Latest headlines", style={"color": FG}), html.Ul(head_items)],
                     style={**CARD, "flex": "1", "minWidth": "320px"}),
        ], style={"display": "flex", "flexWrap": "wrap"}),
    ])


@app.callback(Output("wm-table", "children"), Input("wm-sector", "value"))
def _whymove_table(sector):
    with db_conn() as db:
        df = load_frame(db, sector)
    moves = top_moves(df, 10)
    if not moves:
        return html.Div("No data.", style={"color": MUTED})
    rows = [{"date": m["date"], "move %": f"{m['ret_pct']:+.2f}",
             "driver articles": m["entity_vol"], "all articles": m["article_vol"]}
            for m in moves]
    return dash_table.DataTable(
        id="wm-dt", data=rows,
        columns=[{"name": c, "id": c} for c in ["date", "move %", "driver articles", "all articles"]],
        row_selectable="single", selected_rows=[],
        style_as_list_view=True,
        style_header={"backgroundColor": PANEL, "color": FG, "fontWeight": "bold"},
        style_cell={"backgroundColor": BG, "color": FG, "border": "1px solid #2a3142",
                    "padding": "8px", "textAlign": "center"},
        style_data_conditional=[
            {"if": {"filter_query": "{move %} contains -", "column_id": "move %"},
             "color": DOWN},
        ],
    )


@app.callback(Output("wm-attr", "children"),
              Input("wm-dt", "selected_rows"), State("wm-dt", "data"),
              State("wm-sector", "value"))
def _whymove_attr(selected, data, sector):
    if not selected or not data:
        return html.Div("Select a move above.", style={"color": MUTED, "padding": "12px"})
    date = data[selected[0]]["date"]
    g = None
    try:
        g = GraphClient(settings.NEO4J_URI, settings.NEO4J_USER,
                        settings.NEO4J_PASSWORD, settings.NEO4J_DATABASE)
        with db_conn() as db:
            attr = attribution(db, g, sector, date)
    finally:
        if g:
            g.close()

    ent = ", ".join(f"{e['name']} ({e['mentions']})" for e in attr["entities"]) or "—"
    arts = [html.Li(html.A(f"{a['title']} — {a['domain']}", href=a["url"], target="_blank",
                           style={"color": ACCENT, "textDecoration": "none"}))
            for a in attr["articles"]] or [html.Li("no articles stored for this day",
                                                    style={"color": MUTED})]
    return html.Div([
        html.H4(f"What drove {sector} on {date}", style={"color": FG}),
        html.Div([html.B("Driver entities in the news: "), ent], style={"marginBottom": "8px"}),
        html.Ul(arts),
    ], style={**CARD})


@app.callback(Output("guru-out", "children"), Output("guru-sector", "value"),
              Input("guru-btn", "n_clicks"),
              State("guru-sector", "value"), State("guru-q", "value"))
def _guru(n, sector, question):
    if not n:
        return "Pick a sector, ask a question, and click Ask the Guru.", sector
    q = (question or "Give me the current risk picture and what's driving it.").strip()
    focus = guru.detect_sector(q, sector)   # honour the sector named in the question
    with db_conn() as db:
        frames = {s: load_frame(db, s) for s in ALL_SECTORS}
        stats_all = {s: compute_stats(frames[s]) for s in ALL_SECTORS}
        df, stats = frames[focus], stats_all[focus]
        if not stats.get("ok"):
            return (f"No cached data for {focus}. Run scripts/refresh_market.py first.", focus)
        sigs = detect_signals(focus, df, stats)
        moves = top_moves(df, 5)
        heads = [dict(h) for h in db.conn.execute(
            "SELECT title, domain, url, published_at FROM articles WHERE sector=? "
            "ORDER BY published_at DESC LIMIT 8", (focus,)).fetchall()]
    ents = list(top_entities(focus))
    brief = _build_brief(focus, stats, sigs, ents)

    if not guru.llm_available():
        return (_guru_layout("LLM not configured — set GROQ_API_KEY in .env.", brief,
                             badge="⚙️ computed brief"), focus)

    context = (guru.build_global_snapshot(stats_all) + "\n\n"
               + guru.build_context(focus, stats, sigs, ents, heads, moves))
    try:
        answer = guru.ask(q, context)
        badge = f"🧠 Groq · {settings.GROQ_LARGE_MODEL} · focus: {focus}"
        return _guru_layout(answer, brief, badge=badge), focus
    except Exception as exc:  # noqa: BLE001
        return (_guru_layout(f"LLM call failed ({exc}). Showing the computed brief.",
                             brief, badge="⚙️ computed brief (LLM error)"), focus)


def _guru_layout(answer: str, brief: str, badge: str):
    return html.Div([
        html.Span(badge, style={"backgroundColor": PANEL, "color": ACCENT,
                                "padding": "2px 10px", "borderRadius": "6px",
                                "fontSize": "12px"}),
        dcc.Markdown(answer, style={"marginTop": "10px", "lineHeight": "1.6"}),
        html.Details([
            html.Summary("facts the guru used (computed, ungenerated)",
                         style={"color": MUTED, "cursor": "pointer"}),
            html.Pre(brief, style={"whiteSpace": "pre-wrap", "color": MUTED,
                                   "fontSize": "13px", "marginTop": "8px"}),
        ], style={"marginTop": "14px"}),
    ])


def _build_brief(sector, stats, sigs, ents) -> str:
    inst = SECTOR_INSTRUMENTS[sector]["label"]
    ent_txt = ", ".join(n for n, _, _ in ents[:5]) or "n/a"
    amp = stats.get("amplification")
    lines = [
        f"📊 {sector.upper()} — {inst}",
        f"As of {stats['last_date']}, last close {stats['last_close']:.2f} "
        f"({stats['last_ret']:+.2f}% on the day, {stats['ret_5d']:+.2f}% over 5 days)."
        if stats["ret_5d"] is not None else
        f"As of {stats['last_date']}, last close {stats['last_close']:.2f}.",
        "",
        f"News→price link: {stats['link_confidence'].upper()} "
        f"(r={stats['r_same']:+.2f} same-day, r={stats['r_next']:+.2f} next-day, "
        f"over {stats['n']} trading days).",
        f"On high-news days this instrument moves "
        + (f"~{amp:.1f}x more than normal "
           f"({stats['spike_absret']:.1f}% vs {stats['normal_absret']:.1f}% avg |move|)."
           if amp else "about the same as normal."),
        f"Volatility regime is currently {stats['vol_regime'].upper()} "
        f"(10-day {stats['recent_vol']:.1f}% vs {stats['hist_vol']:.1f}% baseline).",
        "",
        f"Key news drivers tracked: {ent_txt}.",
        "",
        "What this means:",
    ]
    actionable = [s for s in sigs if s["severity"] != "info"]
    if not actionable:
        lines.append("• No elevated signals right now — news flow and volatility are unremarkable.")
    for s in actionable:
        lines.append(f"• [{s['severity'].upper()}] {s['headline']}: {s['detail']}")
    lines += [
        "",
        "⚠️ Decision-support only. These are volatility/risk and attribution signals derived "
        "from news flow — not buy/sell advice. News data tracks volatility better than direction.",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    app.run(debug=False, port=8050)
