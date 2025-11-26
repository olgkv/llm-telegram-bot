import os
import logging
from typing import List

from sqlalchemy.orm import Session
from openai import OpenAI
import openai

from src.db import Document, DocumentChunk, EMBEDDING_DIM


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

logger = logging.getLogger(__name__)

_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


def _get_embedding(text: str) -> List[float]:
    if _client is None:
        return []

    try:
        response = _client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text,
        )
    except Exception:
        return []

    return response.data[0].embedding


def _split_text(text: str, chunk_size: int = 800, overlap: int = 200) -> List[str]:
    """Простое разбиение текста на чанки по символам с перекрытием."""
    chunks: List[str] = []
    start = 0
    length = len(text)

    while start < length:
        end = min(start + chunk_size, length)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
        if start <= 0:
            break

    return chunks


def ingest_text(db: Session, title: str, source: str, text: str) -> Document:
    """Сохраняет текстовый документ и его чанки с эмбеддингами в БД."""
    logger.info("Starting ingestion: title=%r, source=%r, length=%d chars", title, source, len(text))

    document = Document(title=title, source=source)
    db.add(document)
    db.commit()
    db.refresh(document)

    chunks = _split_text(text)
    logger.info("Document id=%s split into %d raw chunks", document.id, len(chunks))

    saved_chunks = 0
    for idx, chunk_text in enumerate(chunks):
        embedding = _get_embedding(chunk_text)
        if not embedding:
            continue

        db.add(
            DocumentChunk(
                document_id=document.id,
                chunk_index=idx,
                text=chunk_text,
                embedding=embedding,
            )
        )
        saved_chunks += 1

    db.commit()
    logger.info("Finished ingestion for document id=%s: saved %d chunks with embeddings", document.id, saved_chunks)
    return document


def retrieve_relevant_chunks(db: Session, query: str, limit: int = 3) -> List[DocumentChunk]:
    """Возвращает наиболее релевантные чанки документа для запроса."""
    if _client is None:
        return []

    embedding = _get_embedding(query)
    if not embedding:
        return []

    return (
        db.query(DocumentChunk)
        .order_by(DocumentChunk.embedding.l2_distance(embedding))
        .limit(limit)
        .all()
    )


def load_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


if __name__ == "__main__":
    """Простейшая консольная утилита: python -m src.rag path/to/file.txt 'Title'"""
    import sys

    from src.db import SessionLocal, init_db

    # Базовая настройка логирования для CLI-запуска
    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 2:
        print("Usage: python -m src.rag path/to/file.txt [title]")
        raise SystemExit(1)

    file_path = sys.argv[1]
    title = sys.argv[2] if len(sys.argv) > 2 else os.path.basename(file_path)

    logger.info("Loading text file for ingestion: path=%s, title=%r", file_path, title)
    text = load_text_file(file_path)

    init_db()
    with SessionLocal() as db:
        doc = ingest_text(db, title=title, source=file_path, text=text)
        print(f"Ingested document id={doc.id}, title={doc.title}")
