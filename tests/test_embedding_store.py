"""Tests for embeddings/store.py — EmbeddingStore."""
import pytest

from embeddings.store import EmbeddingStore, SyncResult


def _make_articles(n: int, sector: str = "energy") -> list[dict]:
    return [
        {
            "id": f"art-{i}",
            "title": f"Article about {sector} markets number {i}",
            "sector": sector,
            "domain": f"news{i % 3}.com",
            "published_at": f"2024-01-{i + 1:02d}",
            "url": f"https://news{i % 3}.com/article/{i}",
        }
        for i in range(n)
    ]


@pytest.fixture()
def store(tmp_path):
    return EmbeddingStore(
        persist_path=str(tmp_path / "chroma"),
        collection_name="test_collection",
    )


def test_add_articles_returns_correct_count(store):
    articles = _make_articles(5)
    added = store.add_articles(articles)
    assert added == 5


def test_add_articles_idempotent(store):
    articles = _make_articles(5)
    store.add_articles(articles)
    added_again = store.add_articles(articles)
    assert added_again == 0
    assert store.collection.count() == 5


def test_search_returns_results_with_valid_scores(store):
    store.add_articles(_make_articles(10))
    results = store.search("energy oil price market", n_results=3)
    assert len(results) >= 1
    for r in results:
        assert 0.0 <= r.relevance_score <= 1.0
        assert r.embedding == []  # stripped before returning


def test_search_with_sector_filter(store):
    energy = _make_articles(5, sector="energy")
    banking = _make_articles(5, sector="banking")
    store.add_articles(energy + banking)

    results = store.search("financial market update", n_results=5,
                           filters={"sector": "energy"})
    for r in results:
        assert r.sector == "energy"


def test_search_returns_empty_when_collection_empty(store):
    results = store.search("anything")
    assert results == []


def test_get_similar_to_article_excludes_source(store):
    store.add_articles(_make_articles(8))
    similar = store.get_similar_to_article("art-0", n_results=3)
    ids = [r.article_id for r in similar]
    assert "art-0" not in ids
    assert len(similar) <= 3


def test_get_similar_to_article_missing_id(store):
    store.add_articles(_make_articles(3))
    result = store.get_similar_to_article("nonexistent-id")
    assert result == []


def test_sync_from_db(tmp_path, mocker):
    store = EmbeddingStore(
        persist_path=str(tmp_path / "chroma"),
        collection_name="sync_test",
    )
    # Pre-populate 5 of 10 articles
    pre_articles = _make_articles(5)
    store.add_articles(pre_articles)

    # Mock Database with 10 articles total
    all_articles = _make_articles(10)

    mock_db = mocker.MagicMock()
    mock_conn = mocker.MagicMock()
    mock_db.conn = mock_conn

    import sqlite3
    rows = [
        mocker.MagicMock(
            spec=sqlite3.Row,
            **{"__getitem__": lambda self, k: {
                "id": a["id"], "url": a["url"], "title": a["title"],
                "domain": a["domain"], "published_at": a["published_at"],
                "sector": a["sector"],
            }[k]}
        )
        for a in all_articles
    ]
    # Make dict(row) work
    for row, art in zip(rows, all_articles):
        row.keys.return_value = ["id", "url", "title", "domain", "published_at", "sector"]
        row.__iter__ = lambda self: iter([])

    mock_conn.execute.return_value.fetchall.return_value = [
        type("Row", (), {"__iter__": lambda s: iter(a.items()),
                         "keys": lambda s: list(a.keys()),
                         **{k: v for k, v in a.items()}})()
        for a in all_articles
    ]

    # Patch db.conn.execute to return article rows
    mock_db.conn.execute.return_value.fetchall.return_value = all_articles

    # Override add_articles to work with dict items
    original_add = store.add_articles

    def patched_add(articles):
        return original_add(articles)

    store_add_spy = mocker.patch.object(store, "add_articles", side_effect=patched_add)

    # Use a real Database-like object instead
    class FakeDB:
        def __init__(self, articles):
            self._articles = articles

        @property
        def conn(self):
            class FakeConn:
                def __init__(self, articles):
                    self._articles = articles

                def execute(self, _sql):
                    class Result:
                        def __init__(self, rows):
                            self._rows = rows
                        def fetchall(self):
                            return self._rows
                    return Result(self._articles)

            return FakeConn(self._articles)

    fake_db = FakeDB(all_articles)
    result: SyncResult = store.sync_from_db(fake_db)  # type: ignore[arg-type]

    assert result.total_in_db == 10
    assert result.already_embedded == 5
    assert result.newly_added == 5
    assert result.failed == 0


def test_mmr_rerank_prefers_diverse_results():
    """MMR should pick the less redundant candidate over a near-duplicate."""
    import numpy as np

    # query embedding
    query = [1.0, 0.0, 0.0, 0.0]

    # candidate 0: very relevant to query
    cand0 = [0.99, 0.0, 0.0, 0.0]
    # candidate 1: near-duplicate of cand0
    cand1 = [0.98, 0.01, 0.0, 0.0]
    # candidate 2: diverse (lower relevance but different direction)
    cand2 = [0.5, 0.86, 0.0, 0.0]

    selected = EmbeddingStore._mmr_rerank(
        [cand0, cand1, cand2], query, n_results=2, lambda_=0.5
    )

    # First pick should be cand0 (most relevant)
    assert selected[0] == 0
    # Second pick should be cand2 (diverse) not cand1 (near-duplicate)
    assert selected[1] == 2
