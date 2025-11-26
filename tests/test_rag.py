from types import SimpleNamespace
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src import rag
from src.db import Base, Document, DocumentChunk, EMBEDDING_DIM


def create_sqlite_session_factory():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    return sessionmaker(bind=engine, autoflush=False, autocommit=False)


def test_split_text_produces_overlapping_chunks():
    text = "0123456789" * 50
    chunks = rag._split_text(text, chunk_size=100, overlap=20)

    assert len(chunks) > 1
    assert all(c for c in chunks)

    first, second = chunks[0], chunks[1]
    overlap = first[-20:]
    assert overlap in second


def test_ingest_text_creates_document_and_chunks(monkeypatch):
    SessionFactory = create_sqlite_session_factory()

    def fake_get_embedding(text: str):
        return [0.0] * EMBEDDING_DIM

    monkeypatch.setattr("src.rag._client", object())
    monkeypatch.setattr("src.rag._get_embedding", fake_get_embedding)

    sample_text = "Тестовый текст для RAG. " * 50

    with SessionFactory() as db:
        doc = rag.ingest_text(db, title="Test Doc", source="test.txt", text=sample_text)

        documents = db.query(Document).all()
        chunks = db.query(DocumentChunk).order_by(DocumentChunk.chunk_index.asc()).all()

    assert len(documents) == 1
    assert documents[0].id == doc.id
    assert documents[0].title == "Test Doc"

    assert len(chunks) > 0
    assert all(c.document_id == doc.id for c in chunks)
    assert all(isinstance(c.embedding, list) or c.embedding is not None for c in chunks)


def test_retrieve_relevant_chunks_uses_embedding_and_limit(monkeypatch):
    called = {"query": None, "limit": None, "embedding_for": None}

    def fake_get_embedding(text: str):
        called["embedding_for"] = text
        return [0.1, 0.2, 0.3]

    # Гарантируем, что функция не вернёт [] из-за отсутствия клиента,
    # и реально вызовет _get_embedding
    monkeypatch.setattr("src.rag._client", object())
    monkeypatch.setattr("src.rag._get_embedding", fake_get_embedding)

    fake_chunks = [
        SimpleNamespace(text="chunk-1"),
        SimpleNamespace(text="chunk-2"),
        SimpleNamespace(text="chunk-3"),
    ]

    class FakeQuery:
        def __init__(self, items):
            self._items = items

        def order_by(self, *args, **kwargs):  # pragma: no cover - просто игнорируем выражение сортировки
            return self

        def limit(self, n):
            called["limit"] = n
            self._limit = n
            return self

        def all(self):
            return self._items[: self._limit]

    class FakeSession:
        def __init__(self, items):
            self._items = items

        def query(self, model):  # noqa: ARG002 - model не используется в тесте
            return FakeQuery(self._items)

    db = FakeSession(fake_chunks)

    result = rag.retrieve_relevant_chunks(db, "какой-то вопрос", limit=2)

    assert called["embedding_for"] == "какой-то вопрос"
    assert called["limit"] == 2
    assert len(result) == 2
    assert result[0].text == "chunk-1"
    assert result[1].text == "chunk-2"


def test_retrieve_relevant_chunks_returns_empty_if_no_client(monkeypatch):
    monkeypatch.setattr("src.rag._client", None)

    result = rag.retrieve_relevant_chunks(db=SimpleNamespace(), query="q", limit=3)

    assert result == []


def test_sample_file_ingest_and_retrieve(monkeypatch):
    """Интеграционный тест: загружаем data/sample.txt и проверяем RAG-поиск.

    Используем фейковый _get_embedding, чтобы не ходить в реальный OpenAI,
    но прогоняем полный цикл ingest_text -> retrieve_relevant_chunks.
    """

    SessionFactory = create_sqlite_session_factory()

    def fake_get_embedding(text: str):
        return [0.1] * EMBEDDING_DIM

    monkeypatch.setattr("src.rag._client", object())
    monkeypatch.setattr("src.rag._get_embedding", fake_get_embedding)

    sample_path = Path(__file__).resolve().parent.parent / "data" / "sample.txt"
    sample_text = sample_path.read_text(encoding="utf-8")

    with SessionFactory() as db:
        doc = rag.ingest_text(db, title="Тестовый документ", source=str(sample_path), text=sample_text)
        doc_id = doc.id
        stored_chunks = (
            db.query(DocumentChunk)
            .order_by(DocumentChunk.chunk_index.asc())
            .all()
        )

    class FakeQuery:
        def __init__(self, items):
            self._items = items

        def order_by(self, *args, **kwargs):  # pragma: no cover - просто игнорируем выражение сортировки
            return self

        def limit(self, n):
            self._limit = n
            return self

        def all(self):
            return self._items[: self._limit]

    class FakeSession:
        def __init__(self, items):
            self._items = items

        def query(self, model):  # noqa: ARG002 - model не используется в тесте
            return FakeQuery(self._items)

    fake_db = FakeSession(stored_chunks)

    chunks = rag.retrieve_relevant_chunks(fake_db, "тестовый документ для RAG", limit=3)

    assert doc_id is not None

    assert chunks

    first_non_empty_line = next(l.strip() for l in sample_text.splitlines() if l.strip())
    snippet = first_non_empty_line[:30]

    assert any(snippet in chunk.text for chunk in chunks)
