"""Maps each news sector to a tradable price instrument and to the knowledge-graph
entities whose news flow drives that instrument.

All instruments use Alpha Vantage *free-tier* REST endpoints:
  - commodity endpoints (WTI) need no symbol
  - everything else is an ETF pulled via TIME_SERIES_DAILY

The driver-entity lists are candidates; at runtime they are intersected with the
entities that actually exist in the Neo4j graph, so unknown names cost nothing.
"""

# sector -> instrument definition
#   key:        cache key used in the `prices` / `sector_daily` tables
#   av_function: Alpha Vantage REST function name
#   symbol:     equity/ETF symbol (None for commodity endpoints)
#   label:      human-readable name for the dashboard
#   kind:       'commodity' or 'etf' (affects how the JSON is parsed)
SECTOR_INSTRUMENTS: dict[str, dict] = {
    "energy": {
        "key": "WTI", "av_function": "WTI", "symbol": None,
        "label": "WTI Crude Oil", "kind": "commodity",
    },
    "semiconductors": {
        "key": "SOXX", "av_function": "TIME_SERIES_DAILY", "symbol": "SOXX",
        "label": "iShares Semiconductor ETF (SOXX)", "kind": "etf",
    },
    "banking": {
        "key": "XLF", "av_function": "TIME_SERIES_DAILY", "symbol": "XLF",
        "label": "Financial Select Sector ETF (XLF)", "kind": "etf",
    },
    "commodities": {
        "key": "DBC", "av_function": "TIME_SERIES_DAILY", "symbol": "DBC",
        "label": "Invesco DB Commodity ETF (DBC)", "kind": "etf",
    },
    "geopolitical": {
        "key": "GLD", "av_function": "TIME_SERIES_DAILY", "symbol": "GLD",
        "label": "SPDR Gold Shares (GLD)", "kind": "etf",
    },
    "macro": {
        "key": "SPY", "av_function": "TIME_SERIES_DAILY", "symbol": "SPY",
        "label": "S&P 500 ETF (SPY)", "kind": "etf",
    },
}

# sector -> knowledge-graph entity names whose mentions move the instrument
SECTOR_DRIVERS: dict[str, list[str]] = {
    "energy": [
        "Iran", "Hormuz", "OPEC", "Russia", "Venezuela", "Saudi Arabia",
        "Saudi", "Ukraine", "Strait of Hormuz", "OPEC+",
    ],
    "semiconductors": [
        "NVIDIA", "Nvidia", "Intel", "TSMC", "AMD", "Taiwan", "ASML",
        "Samsung", "Micron", "Qualcomm", "Broadcom",
    ],
    "banking": [
        "Fed", "Federal Reserve", "Jerome Powell", "Powell", "ECB",
        "JPMorgan", "SVB", "Goldman Sachs", "Goldman", "Morgan Stanley",
    ],
    "commodities": [
        "China", "OPEC", "Trump", "gold", "copper", "wheat", "Russia", "India",
    ],
    "geopolitical": [
        "Iran", "Russia", "Ukraine", "China", "Taiwan", "Israel", "Hamas",
        "North Korea", "NATO", "Trump", "Hormuz", "Venezuela",
    ],
    "macro": [
        "Fed", "Powell", "Jerome Powell", "IMF", "World Bank", "Trump",
        "China", "Federal Reserve", "ECB",
    ],
}

# Optional extra free commodity series fetched for the energy deep-dive.
BONUS_COMMODITIES: dict[str, str] = {
    "BRENT": "BRENT",
    "NATURAL_GAS": "NATURAL_GAS",
}

ALL_SECTORS: list[str] = list(SECTOR_INSTRUMENTS.keys())
