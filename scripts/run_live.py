"""Start the JARVIS-Fi live ingestion scheduler.

Usage:
    python scripts/run_live.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from storage.database import Database
from config import settings
from ingestion.scheduler import run_scheduler


def main() -> None:
    """Initialise DB schema then start the blocking scheduler."""
    with Database(settings.DB_PATH) as db:
        db.init_schema()
    run_scheduler()


if __name__ == "__main__":
    main()
