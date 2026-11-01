"""Financial theme keyword lists for GDELT queries."""

FINANCIAL_THEMES: dict[str, list[str]] = {
    "energy": [
        "oil price", "OPEC production", "crude oil", "natural gas", "energy crisis",
        "oil sanctions", "LNG exports", "oil supply", "oil demand", "oil production cut",
    ],
    "semiconductors": [
        "semiconductor", "chip shortage", "TSMC foundry", "NVIDIA", "Intel",
        "chip ban", "export controls semiconductors", "chip supply chain",
        "Taiwan chip", "foundry",
    ],
    "banking": [
        "interest rate hike", "Federal Reserve", "central bank rate",
        "bank collapse", "banking crisis", "SVB collapse", "inflation rate",
        "rate cut", "monetary policy", "quantitative easing",
    ],
    "commodities": [
        "wheat price", "grain supply", "food inflation", "copper price",
        "gold price", "commodity shortage", "supply chain disruption",
        "port blockage", "freight rates", "trade war tariff",
    ],
    "geopolitical": [
        "sanctions", "trade embargo", "military conflict", "border dispute",
        "military coup", "political instability", "armed conflict", "invasion",
        "diplomatic crisis", "nuclear threat",
    ],
    "macro": [
        "recession", "GDP growth", "unemployment rate", "CPI inflation",
        "sovereign debt", "IMF bailout", "World Bank", "currency devaluation",
        "hyperinflation", "debt ceiling",
    ],
}

TRUSTED_SOURCES: list[str] = [
    "reuters.com",
    "bloomberg.com",
    "ft.com",
    "wsj.com",
    "cnbc.com",
    "bbc.co.uk",
    "theguardian.com",
    "economist.com",
    "apnews.com",
    "aljazeera.com",
    "nikkei.com",
    "scmp.com",
    "lemonde.fr",
    "spiegel.de",
    "xinhuanet.com",
    "marketwatch.com",
    "investing.com",
    "seekingalpha.com",
    "zerohedge.com",
    "forexlive.com",
]
