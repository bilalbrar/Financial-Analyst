"""APScheduler job runner for live GDELT ingestion."""
import json
import logging
import logging.handlers
from datetime import datetime, timedelta, timezone
from pathlib import Path

from apscheduler.schedulers.blocking import BlockingScheduler

from config import settings
from config.themes import FINANCIAL_THEMES
from graph.neo4j_client import Neo4jClient
from graph.ner import NERProcessor
from graph.writer import push_articles, push_tone_signal
from ingestion.article_processor import process_articles
from ingestion.gdelt_client import GDELTClient
from ingestion.sentiment import tone_series_to_signal
from storage.database import Database

logger = logging.getLogger(__name__)

_DATA_DIR = Path(settings.DATA_DIR)
_DATA_DIR.mkdir(parents=True, exist_ok=True)

# NER model loaded once at startup — expensive to reload per job
_ner: NERProcessor | None = None


def _get_ner() -> NERProcessor:
    global _ner
    if _ner is None:
        _ner = NERProcessor()
    return _ner


def _neo4j() -> Neo4jClient:
    return Neo4jClient(
        settings.NEO4J_URI,
        settings.NEO4J_USER,
        settings.NEO4J_PASSWORD,
        settings.NEO4J_DATABASE,
    )


def _setup_logging() -> None:
    fmt = "%(asctime)s %(levelname)-8s %(name)s - %(message)s"
    level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)

    root = logging.getLogger()
    root.setLevel(level)

    if not root.handlers:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter(fmt))
        root.addHandler(stream_handler)

    log_file = _DATA_DIR / "scheduler.log"
    file_handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=10 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(logging.Formatter(fmt))
    root.addHandler(file_handler)


def _hourly_articles_job() -> None:
    """Fetch last 7 days of articles for every sector, insert into SQLite and Neo4j."""
    logger.info("--- hourly_articles_job START ---")
    client = GDELTClient(
        request_delay=settings.GDELT_REQUEST_DELAY_SECONDS,

        language=settings.GDELT_LANGUAGE,
    )
    now = datetime.now(timezone.utc)
    start = (now - timedelta(days=7)).strftime("%Y-%m-%d")
    end = now.strftime("%Y-%m-%d")

    ner = _get_ner()

    with Database(settings.DB_PATH) as db, _neo4j() as neo4j:
        for sector, keywords in FINANCIAL_THEMES.items():
            new_articles: list[dict] = []

            for keyword in keywords:
                try:
                    raw = client.fetch_articles([keyword], start, end)
                    processed = process_articles(raw, sector, [keyword])
                    for _, row in processed.iterrows():
                        article = row.to_dict()
                        if db.insert_article(article):
                            new_articles.append(article)
                except Exception as exc:  # noqa: BLE001
                    logger.error("hourly_articles_job [%s/%s]: %s", sector, keyword, exc)

            if new_articles:
                try:
                    push_articles(new_articles, neo4j, ner)
                except Exception as exc:  # noqa: BLE001
                    logger.error("Neo4j push failed [%s]: %s", sector, exc)

            logger.info("hourly_articles_job [%s] new=%d", sector, len(new_articles))

    logger.info("--- hourly_articles_job END ---")


def _daily_tone_job() -> None:
    """Fetch tone timeline for past 7 days, upsert into SQLite and Neo4j."""
    logger.info("--- daily_tone_job START ---")
    client = GDELTClient(
        request_delay=settings.GDELT_REQUEST_DELAY_SECONDS,

        language=settings.GDELT_LANGUAGE,
    )
    now = datetime.now(timezone.utc)
    start = (now - timedelta(days=7)).strftime("%Y-%m-%d")
    end = now.strftime("%Y-%m-%d")

    with Database(settings.DB_PATH) as db, _neo4j() as neo4j:
        for sector, keywords in FINANCIAL_THEMES.items():
            try:
                tone_df = client.fetch_tone_timeline(keywords[:3], start, end)
                signal = tone_series_to_signal(tone_df)

                if signal["mean"] is None:
                    continue

                tone_row = {
                    "sector": sector,
                    "keywords": json.dumps(keywords[:3]),
                    "signal_date": end,
                    "tone_value": signal["mean"],
                    "tone_class": signal["latest_classification"],
                    "volume_count": len(tone_df) if not tone_df.empty else 0,
                }
                db.upsert_tone_signal(tone_row)

                try:
                    push_tone_signal(tone_row, neo4j)
                except Exception as exc:  # noqa: BLE001
                    logger.error("Neo4j tone push failed [%s]: %s", sector, exc)

                logger.info(
                    "daily_tone_job [%s] mean=%.2f class=%s trend=%s",
                    sector, signal["mean"], signal["latest_classification"], signal["trend"],
                )
            except Exception as exc:  # noqa: BLE001
                logger.error("daily_tone_job [%s]: %s", sector, exc)

    logger.info("--- daily_tone_job END ---")


def _signal_job() -> None:
    """Detect signals from Neo4j + SQLite and write to the signals table."""
    logger.info("--- signal_job START ---")
    try:
        from signals.generator import generate_all
        with Database(settings.DB_PATH) as db, _neo4j() as neo4j:
            detected = generate_all(db, neo4j)
            for sig in detected:
                db.insert_signal(sig)
        logger.info("signal_job: %d signals detected", len(detected))
    except Exception as exc:  # noqa: BLE001
        logger.error("signal_job failed: %s", exc)
    logger.info("--- signal_job END ---")


def _health_check_job() -> None:
    """Log DB row counts to data/health.log."""
    health_log = _DATA_DIR / "health.log"
    with Database(settings.DB_PATH) as db:
        counts = db.row_counts()
    ts = datetime.now(timezone.utc).isoformat()
    line = f"{ts} | " + " | ".join(f"{t}={c}" for t, c in counts.items())
    with health_log.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")
    logger.info("Health check: %s", line)


def run_scheduler() -> None:
    """Start the blocking APScheduler with all configured jobs."""
    _setup_logging()
    _get_ner()  # warm up NER at startup, not during first job run
    logger.info("Starting JARVIS-Fi scheduler (tz=%s)", settings.SCHEDULER_TIMEZONE)

    scheduler = BlockingScheduler(timezone=settings.SCHEDULER_TIMEZONE)

    scheduler.add_job(
        _hourly_articles_job,
        trigger="interval",
        minutes=60,
        id="hourly_articles",
        name="Fetch articles + push to Neo4j (hourly)",
    )
    scheduler.add_job(
        _daily_tone_job,
        trigger="cron",
        hour=0,
        minute=5,
        id="daily_tone",
        name="Fetch tone signals + push to Neo4j (daily)",
    )
    scheduler.add_job(
        _signal_job,
        trigger="interval",
        hours=6,
        id="signal_detection",
        name="Detect signals (every 6h)",
    )
    scheduler.add_job(
        _health_check_job,
        trigger="interval",
        minutes=15,
        id="health_check",
        name="Health check (15-min)",
    )

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped.")
