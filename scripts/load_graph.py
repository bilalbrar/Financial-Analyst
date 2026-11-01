"""Load JARVIS-Fi SQLite data into Neo4j.

Usage:
    python scripts/load_graph.py
    python scripts/load_graph.py --neo4j-uri bolt://localhost:7687
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings
from graph.loader import load

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s %(levelname)-8s %(name)s - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("load_graph")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Load JARVIS-Fi data into Neo4j")
    p.add_argument("--neo4j-uri", default=settings.NEO4J_URI)
    p.add_argument("--neo4j-user", default=settings.NEO4J_USER)
    p.add_argument("--neo4j-password", default=settings.NEO4J_PASSWORD)
    p.add_argument("--neo4j-database", default=settings.NEO4J_DATABASE)
    p.add_argument("--batch-size", type=int, default=500)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    logger.info("Connecting to Neo4j at %s", args.neo4j_uri)

    summary = load(
        db_path=settings.DB_PATH,
        neo4j_uri=args.neo4j_uri,
        neo4j_user=args.neo4j_user,
        neo4j_password=args.neo4j_password,
        neo4j_database=args.neo4j_database,
        batch_size=args.batch_size,
    )

    print(f"\n{'='*50}")
    print("GRAPH LOAD COMPLETE")
    print(f"{'='*50}")
    for k, v in summary.items():
        print(f"  {k:<20} {v:>8,}")


if __name__ == "__main__":
    main()
