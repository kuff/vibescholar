"""Shared test fixtures for VibeScholar tests."""
from __future__ import annotations

from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent
SAMPLE_PAPERS_DIR = TESTS_DIR / "test_papers"


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Temporary data directory that persists for the whole module."""
    return tmp_path_factory.mktemp("vibescholar_test")


@pytest.fixture(scope="module")
def indexed_backend(data_dir: Path):
    """Full backend with 10 test PDFs indexed. For read-only tests."""
    from vibescholar import config

    config.configure(data_dir=data_dir)
    config.ensure_data_dirs()

    from vibescholar.db import IndexDatabase
    from vibescholar.embeddings import Embedder
    from vibescholar.vectors import FaissStore
    from vibescholar.indexer import PdfIndexer
    from vibescholar.search import SearchService

    embedder = Embedder()
    db = IndexDatabase(config.DB_PATH)
    store = FaissStore(config.FAISS_INDEX_PATH, embedder.dimension)
    indexer = PdfIndexer(db, embedder, store)

    directory_id = db.upsert_directory(str(SAMPLE_PAPERS_DIR))
    stats = indexer.index_directory(directory_id, SAMPLE_PAPERS_DIR)

    search_service = SearchService(db, embedder, store)

    class Backend:
        pass

    b = Backend()
    b.db = db
    b.embedder = embedder
    b.store = store
    b.indexer = indexer
    b.search_service = search_service
    b.stats = stats
    b.directory_id = directory_id
    return b


@pytest.fixture
def empty_db(tmp_path: Path):
    """Fresh IndexDatabase with no data. For testing mutations."""
    from vibescholar import config

    config.configure(data_dir=tmp_path)
    config.ensure_data_dirs()

    from vibescholar.db import IndexDatabase

    return IndexDatabase(config.DB_PATH)
