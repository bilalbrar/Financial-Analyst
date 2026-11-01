"""One-time / incremental ChromaDB backfill from SQLite articles.

Usage:
    python scripts/sync_embeddings.py
    python scripts/sync_embeddings.py --batch-size 250
    python scripts/sync_embeddings.py --sector energy
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings
from embeddings.store import EmbeddingStore
from storage.database import Database


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync SQLite articles into ChromaDB")
    parser.add_argument("--batch-size", type=int, default=500, metavar="N")
    parser.add_argument("--sector", type=str, default=None, metavar="STR")
    args = parser.parse_args()

    t0 = time.time()  # always defined before any branching

    store = EmbeddingStore(
        persist_path=settings.CHROMA_PERSIST_PATH,
        collection_name=settings.CHROMA_COLLECTION_NAME,
    )

    with Database(settings.DB_PATH) as db:
        db.init_schema()

        if args.sector:
            rows = db.conn.execute(
                "SELECT id, url, title, domain, published_at, sector FROM articles "
                "WHERE sector = ?",
                (args.sector,),
            ).fetchall()
            articles = [dict(r) for r in rows]
        else:
            result = store.sync_from_db(db, batch_size=args.batch_size)
            _print_result(result, store, t0)
            return

    # Sector-filtered path (manual batching)
    total = len(articles)

    try:
        from tqdm import tqdm
        pbar = tqdm(total=total, desc="Syncing embeddings", unit="articles")
    except ImportError:
        pbar = None

    newly_added = 0
    failed = 0
    already_ids = set(store.collection.get(ids=[a["id"] for a in articles], include=[])["ids"])

    for i in range(0, total, args.batch_size):
        batch = [a for a in articles[i : i + args.batch_size] if a["id"] not in already_ids]
        try:
            n = store.add_articles(batch)
            newly_added += n
        except Exception as exc:
            print(f"  Batch {i} failed: {exc}", file=sys.stderr)
            failed += len(batch)
        if pbar:
            pbar.update(min(args.batch_size, total - i))

    if pbar:
        pbar.close()

    elapsed = time.time() - t0
    already = len(already_ids)
    print(f"\nAlready in ChromaDB: {already}")
    print(f"Newly embedded:      {newly_added}")
    print(f"Failed:              {failed}")
    print(f"Collection total:    {store.collection.count()}")
    print(f"Time elapsed:        {elapsed:.1f}s")


def _print_result(result, store, t0: float) -> None:
    elapsed = time.time() - t0
    print(f"\nAlready in ChromaDB: {result.already_embedded}")
    print(f"Newly embedded:      {result.newly_added}")
    print(f"Failed:              {result.failed}")
    print(f"Collection total:    {store.collection.count()}")
    print(f"Time elapsed:        {elapsed:.1f}s")


if __name__ == "__main__":
    main()
