"""ChromaDB semantic search store with MMR reranking."""
import logging
from dataclasses import dataclass, field

import logging

import chromadb
import numpy as np
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer

# ChromaDB 0.5.x has a broken posthog integration (wrong call signature).
# The failures are logged at ERROR level, so we must go above CRITICAL to silence them.
_CHROMA_TELEM_LOGGERS = [
    "chromadb.telemetry.product.posthog",
    "chromadb.telemetry.product",
    "chromadb.telemetry",
]
for _name in _CHROMA_TELEM_LOGGERS:
    logging.getLogger(_name).setLevel(logging.CRITICAL + 1)

from storage.database import Database

logger = logging.getLogger(__name__)

_EMBED_BATCH = 500


@dataclass
class SearchResult:
    article_id: str
    title: str
    url: str
    domain: str
    sector: str
    published_at: str
    relevance_score: float
    embedding: list[float] = field(default_factory=list, repr=False)


@dataclass
class SyncResult:
    total_in_db: int
    already_embedded: int
    newly_added: int
    failed: int


class EmbeddingStore:
    """ChromaDB wrapper with sentence-transformer embeddings and MMR reranking."""

    def __init__(
        self,
        persist_path: str,
        collection_name: str,
        model_name: str = "all-MiniLM-L6-v2",
    ) -> None:
        self._model = SentenceTransformer(model_name)
        self.client = chromadb.PersistentClient(
            path=persist_path,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "EmbeddingStore ready: collection=%s items=%d",
            collection_name,
            self.collection.count(),
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _encode(self, texts: list[str]) -> np.ndarray:
        return self._model.encode(texts, batch_size=64, show_progress_bar=False)

    @staticmethod
    def _mmr_rerank(
        candidate_embeddings: list[list[float]],
        query_embedding: list[float],
        n_results: int,
        lambda_: float = 0.5,
    ) -> list[int]:
        """Maximal Marginal Relevance reranking using numpy dot products.

        Args:
            candidate_embeddings: Embeddings of candidate documents.
            query_embedding: Embedding of the query.
            n_results: Number of documents to select.
            lambda_: Trade-off between relevance (1.0) and diversity (0.0).

        Returns:
            Indices into candidate_embeddings in selection order.
        """
        if not candidate_embeddings:
            return []

        embs = np.array(candidate_embeddings, dtype=np.float32)
        query = np.array(query_embedding, dtype=np.float32)

        # L2-normalise for cosine similarity via dot product
        emb_norms = np.linalg.norm(embs, axis=1, keepdims=True)
        embs_norm = embs / np.where(emb_norms > 0, emb_norms, 1.0)
        q_norm = query / (np.linalg.norm(query) + 1e-9)

        relevance = embs_norm @ q_norm  # (n,)

        selected: list[int] = []
        remaining = list(range(len(candidate_embeddings)))

        for _ in range(min(n_results, len(candidate_embeddings))):
            if not selected:
                best = remaining[int(np.argmax([relevance[i] for i in remaining]))]
            else:
                sel_embs = embs_norm[selected]  # (k, dim)
                best_score = float("-inf")
                best = remaining[0]
                for i in remaining:
                    sim_to_sel = float(np.max(embs_norm[i] @ sel_embs.T))
                    score = lambda_ * float(relevance[i]) - (1 - lambda_) * sim_to_sel
                    if score > best_score:
                        best_score = score
                        best = i
            selected.append(best)
            remaining.remove(best)

        return selected

    def _build_where(self, filters: dict | None) -> dict | None:
        if not filters:
            return None
        conditions: list[dict] = []
        if "sector" in filters:
            conditions.append({"sector": {"$eq": filters["sector"]}})
        if "domain" in filters:
            conditions.append({"domain": {"$in": filters["domain"]}})
        if "published_after" in filters:
            conditions.append({"published_at": {"$gte": filters["published_after"]}})
        if "published_before" in filters:
            conditions.append({"published_at": {"$lte": filters["published_before"]}})
        if not conditions:
            return None
        return conditions[0] if len(conditions) == 1 else {"$and": conditions}

    def _rows_to_results(
        self,
        ids: list[str],
        metadatas: list[dict],
        distances: list[float],
        embeddings: list[list[float]] | None = None,
    ) -> list[SearchResult]:
        results = []
        for idx, (aid, meta, dist) in enumerate(zip(ids, metadatas, distances)):
            emb = embeddings[idx] if embeddings else []
            results.append(
                SearchResult(
                    article_id=aid,
                    title=meta.get("title", ""),
                    url=meta.get("url", ""),
                    domain=meta.get("domain", ""),
                    sector=meta.get("sector", ""),
                    published_at=meta.get("published_at", ""),
                    relevance_score=max(0.0, 1.0 - dist),
                    embedding=emb,
                )
            )
        return results

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_articles(self, articles: list[dict]) -> int:
        """Embed and store articles; skips any already in the collection.

        Args:
            articles: List of dicts with keys: id, title, sector, domain,
                      published_at, url.

        Returns:
            Count of newly added items (not total).
        """
        if not articles:
            return 0

        all_ids = [a["id"] for a in articles]

        # Find which IDs are already stored
        existing = self.collection.get(ids=all_ids, include=[])
        existing_ids = set(existing["ids"])
        new_articles = [a for a in articles if a["id"] not in existing_ids]

        if not new_articles:
            return 0

        total_added = 0
        for i in range(0, len(new_articles), _EMBED_BATCH):
            batch = new_articles[i : i + _EMBED_BATCH]
            titles = [a.get("title") or "" for a in batch]
            try:
                embs = self._encode(titles)
                self.collection.add(
                    ids=[a["id"] for a in batch],
                    embeddings=embs.tolist(),
                    metadatas=[
                        {
                            "title": a.get("title") or "",
                            "sector": a.get("sector") or "",
                            "domain": a.get("domain") or "",
                            "published_at": str(a.get("published_at") or ""),
                            "url": a.get("url") or "",
                        }
                        for a in batch
                    ],
                    documents=titles,
                )
                total_added += len(batch)
            except Exception as exc:  # noqa: BLE001
                logger.error("add_articles batch %d failed: %s", i, exc)

        logger.info("add_articles: %d new, %d already present", total_added, len(existing_ids))
        return total_added

    def search(
        self,
        query: str,
        n_results: int = 8,
        filters: dict | None = None,
    ) -> list[SearchResult]:
        """Semantic search with MMR reranking.

        Args:
            query: Free-text query string.
            n_results: Number of results to return after reranking.
            filters: Optional filter dict (sector, domain, published_after,
                     published_before).

        Returns:
            Sorted list of SearchResult (no embedding field populated).
        """
        if self.collection.count() == 0:
            return []

        query_emb = self._encode([query])[0]
        where = self._build_where(filters)

        fetch_n = min(n_results * 2, self.collection.count())
        try:
            raw = self.collection.query(
                query_embeddings=[query_emb.tolist()],
                n_results=fetch_n,
                where=where,
                include=["embeddings", "metadatas", "distances"],
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("search query failed: %s", exc)
            return []

        ids = raw["ids"][0]
        metadatas = raw["metadatas"][0]
        distances = raw["distances"][0]
        embeddings = raw["embeddings"][0] if raw.get("embeddings") else None

        if not ids:
            return []

        candidate_embs = embeddings if embeddings else []
        selected_idx = self._mmr_rerank(
            candidate_embs, query_emb.tolist(), n_results
        )

        results = self._rows_to_results(ids, metadatas, distances, embeddings)
        reranked = [results[i] for i in selected_idx]

        # Strip embeddings before returning
        for r in reranked:
            r.embedding = []

        return sorted(reranked, key=lambda r: r.relevance_score, reverse=True)

    def get_similar_to_article(
        self, article_id: str, n_results: int = 5
    ) -> list[SearchResult]:
        """Find articles semantically similar to a stored article.

        Args:
            article_id: ID of the source article (must already be in the collection).
            n_results: Number of neighbours to return.

        Returns:
            Sorted list of SearchResult excluding the source article.
        """
        try:
            stored = self.collection.get(
                ids=[article_id], include=["embeddings"]
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("get_similar_to_article fetch: %s", exc)
            return []

        if not stored["ids"] or not stored.get("embeddings"):
            return []

        source_emb = stored["embeddings"][0]
        fetch_n = min(n_results + 1, self.collection.count())

        try:
            raw = self.collection.query(
                query_embeddings=[source_emb],
                n_results=fetch_n,
                include=["metadatas", "distances"],
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("get_similar_to_article query: %s", exc)
            return []

        ids = raw["ids"][0]
        metadatas = raw["metadatas"][0]
        distances = raw["distances"][0]

        results = self._rows_to_results(ids, metadatas, distances)
        # Exclude source article
        results = [r for r in results if r.article_id != article_id]
        return sorted(results[:n_results], key=lambda r: r.relevance_score, reverse=True)

    def sync_from_db(self, db: Database, batch_size: int = _EMBED_BATCH) -> SyncResult:
        """Pull all articles from SQLite not yet embedded and add them.

        Args:
            db: Connected Database instance.
            batch_size: Articles to process per batch.

        Returns:
            SyncResult with counts.
        """
        rows = db.conn.execute(
            "SELECT id, url, title, domain, published_at, sector FROM articles"
        ).fetchall()
        all_articles = [dict(r) for r in rows]
        total_in_db = len(all_articles)

        if total_in_db == 0:
            return SyncResult(total_in_db=0, already_embedded=0, newly_added=0, failed=0)

        all_ids = [a["id"] for a in all_articles]
        existing = self.collection.get(ids=all_ids, include=[])
        existing_ids = set(existing["ids"])
        already_embedded = len(existing_ids)
        new_articles = [a for a in all_articles if a["id"] not in existing_ids]

        newly_added = 0
        failed = 0

        for i in range(0, len(new_articles), batch_size):
            batch = new_articles[i : i + batch_size]
            try:
                added = self.add_articles(batch)
                newly_added += added
            except Exception as exc:  # noqa: BLE001
                logger.error("sync_from_db batch %d: %s", i, exc)
                failed += len(batch)

        return SyncResult(
            total_in_db=total_in_db,
            already_embedded=already_embedded,
            newly_added=newly_added,
            failed=failed,
        )

    def get_stats(self) -> dict:
        """Return collection statistics.

        Returns:
            Dict with total_items, by_sector, by_domain, oldest_article,
            newest_article.
        """
        total = self.collection.count()
        if total == 0:
            return {
                "total_items": 0,
                "by_sector": {},
                "by_domain": {},
                "oldest_article": None,
                "newest_article": None,
            }

        all_items = self.collection.get(include=["metadatas"])
        metadatas = all_items.get("metadatas") or []

        by_sector: dict[str, int] = {}
        by_domain: dict[str, int] = {}
        dates: list[str] = []

        for meta in metadatas:
            s = meta.get("sector") or "unknown"
            by_sector[s] = by_sector.get(s, 0) + 1
            d = meta.get("domain") or "unknown"
            by_domain[d] = by_domain.get(d, 0) + 1
            dt = meta.get("published_at") or ""
            if dt:
                dates.append(dt)

        top_domains = dict(
            sorted(by_domain.items(), key=lambda kv: kv[1], reverse=True)[:10]
        )
        dates_sorted = sorted(dates)

        return {
            "total_items": total,
            "by_sector": by_sector,
            "by_domain": top_domains,
            "oldest_article": dates_sorted[0] if dates_sorted else None,
            "newest_article": dates_sorted[-1] if dates_sorted else None,
        }
