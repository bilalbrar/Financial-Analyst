CREATE TABLE IF NOT EXISTS articles (
    id               TEXT PRIMARY KEY,
    url              TEXT NOT NULL UNIQUE,
    title            TEXT,
    domain           TEXT,
    language         TEXT DEFAULT 'en',
    source_country   TEXT,
    published_at     DATETIME,
    sector           TEXT,
    keywords_matched TEXT,
    fetched_at       DATETIME DEFAULT CURRENT_TIMESTAMP,
    title_word_count INTEGER
);

CREATE INDEX IF NOT EXISTS idx_articles_published ON articles(published_at);
CREATE INDEX IF NOT EXISTS idx_articles_sector    ON articles(sector);
CREATE INDEX IF NOT EXISTS idx_articles_domain    ON articles(domain);

CREATE TABLE IF NOT EXISTS tone_signals (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    sector       TEXT NOT NULL,
    keywords     TEXT NOT NULL,
    signal_date  DATE NOT NULL,
    tone_value   REAL,
    tone_class   TEXT,
    volume_count INTEGER,
    created_at   DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(sector, signal_date)
);

CREATE INDEX IF NOT EXISTS idx_tone_sector_date ON tone_signals(sector, signal_date);

CREATE TABLE IF NOT EXISTS signals (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_type  TEXT NOT NULL,
    sector       TEXT,
    entity_name  TEXT,
    entity_type  TEXT,
    severity     TEXT NOT NULL,
    value        REAL,
    baseline     REAL,
    description  TEXT,
    window_days  INTEGER,
    detected_at  DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_signals_detected ON signals(detected_at);
CREATE INDEX IF NOT EXISTS idx_signals_sector   ON signals(sector);
CREATE INDEX IF NOT EXISTS idx_signals_type     ON signals(signal_type);

CREATE TABLE IF NOT EXISTS insights (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    type         TEXT NOT NULL,
    sector       TEXT,
    title        TEXT,
    content      TEXT,
    context_json TEXT,
    created_at   DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_insights_type    ON insights(type);
CREATE INDEX IF NOT EXISTS idx_insights_created ON insights(created_at);

CREATE TABLE IF NOT EXISTS ingestion_runs (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    run_type         TEXT,
    sector           TEXT,
    start_date       TEXT,
    end_date         TEXT,
    articles_fetched INTEGER DEFAULT 0,
    articles_new     INTEGER DEFAULT 0,
    articles_dupes   INTEGER DEFAULT 0,
    status           TEXT DEFAULT 'running',
    error_message    TEXT,
    started_at       DATETIME DEFAULT CURRENT_TIMESTAMP,
    completed_at     DATETIME
);

-- Cached Alpha Vantage daily price series (one row per instrument per day).
CREATE TABLE IF NOT EXISTS prices (
    instrument   TEXT NOT NULL,   -- e.g. 'WTI', 'SOXX', 'XLF'
    price_date   DATE NOT NULL,
    close        REAL,
    fetched_at   DATETIME DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (instrument, price_date)
);

CREATE INDEX IF NOT EXISTS idx_prices_instr_date ON prices(instrument, price_date);

-- Pre-computed daily join of news activity and price for each sector.
-- Powers the dashboard charts, correlations and signals without re-querying
-- Neo4j on every page load.
CREATE TABLE IF NOT EXISTS sector_daily (
    sector        TEXT NOT NULL,
    trade_date    DATE NOT NULL,
    instrument    TEXT,
    close         REAL,
    ret_pct       REAL,            -- daily % change in close
    abs_ret_pct   REAL,
    article_vol   INTEGER DEFAULT 0,   -- all sector articles that day
    entity_vol    INTEGER DEFAULT 0,   -- articles mentioning the sector's driver entities
    computed_at   DATETIME DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (sector, trade_date)
);

CREATE INDEX IF NOT EXISTS idx_sector_daily ON sector_daily(sector, trade_date);
