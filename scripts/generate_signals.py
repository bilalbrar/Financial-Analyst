"""Run signal detection manually and print results.

Usage:
    python scripts/generate_signals.py
    python scripts/generate_signals.py --hours 48   # show signals from last 48h
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tabulate import tabulate

from config import settings
from graph.neo4j_client import Neo4jClient
from signals.generator import generate_all
from storage.database import Database

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s %(levelname)-8s %(name)s - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("generate_signals")


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect and store JARVIS-Fi signals")
    parser.add_argument("--hours", type=int, default=24,
                        help="Show signals detected in the last N hours after writing (default: 24)")
    args = parser.parse_args()

    with Database(settings.DB_PATH) as db, Neo4jClient(
        settings.NEO4J_URI, settings.NEO4J_USER,
        settings.NEO4J_PASSWORD, settings.NEO4J_DATABASE,
    ) as neo4j:
        db.init_schema()
        detected = generate_all(db, neo4j)

        for sig in detected:
            db.insert_signal(sig)

        print(f"\n{'='*60}")
        print(f"SIGNALS DETECTED: {len(detected)}")
        print(f"{'='*60}\n")

        if not detected:
            print("No signals above threshold.")
            return

        rows = [
            [
                s["signal_type"],
                s.get("sector") or "—",
                s.get("entity_name") or "—",
                s["severity"].upper(),
                f"{s['value']:.1f}" if s.get("value") is not None else "—",
                f"{s['baseline']:.1f}" if s.get("baseline") is not None else "—",
                s.get("description", "")[:70],
            ]
            for s in sorted(detected, key=lambda x: x["severity"], reverse=True)
        ]
        print(tabulate(
            rows,
            headers=["type", "sector", "entity", "severity", "value", "baseline", "description"],
            tablefmt="simple",
        ))

        print(f"\n--- Summary by type ---")
        for sig_type in ("tone_alert", "tone_delta", "entity_spike", "cross_sector", "volume_spike"):
            count = sum(1 for s in detected if s["signal_type"] == sig_type)
            if count:
                print(f"  {sig_type:<20} {count}")


if __name__ == "__main__":
    main()
