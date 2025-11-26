import os
from datetime import datetime, timezone

from sqlalchemy import Column, Integer, BigInteger, String, Text, DateTime, ForeignKey, create_engine, text, select
from sqlalchemy.orm import declarative_base, relationship, sessionmaker, Session
from pgvector.sqlalchemy import Vector


DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://llm_bot_user:llm_bot_password@postgres:5432/llm_bot")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

Base = declarative_base()

MAX_MESSAGES_PER_USER = 30
EMBEDDING_DIM = 1536


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    telegram_id = Column(BigInteger, unique=True, index=True, nullable=False)
    username = Column(String(255), nullable=True)
    first_name = Column(String(255), nullable=True)
    last_name = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

    messages = relationship("Message", back_populates="user", cascade="all, delete-orphan")


class Message(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    role = Column(String(32), nullable=False)  # user / assistant / system
    content = Column(Text, nullable=False)
    token_count = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False, index=True)

    user = relationship("User", back_populates="messages")


class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(255), nullable=False)
    source = Column(String(255), nullable=True)  # путь к файлу или другой идентификатор
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False, index=True)

    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    chunk_index = Column(Integer, nullable=False)
    text = Column(Text, nullable=False)
    embedding = Column(Vector(EMBEDDING_DIM), nullable=False)

    document = relationship("Document", back_populates="chunks")


def init_db() -> None:
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()

    Base.metadata.create_all(bind=engine)


def trim_old_messages(db: Session, user_id: int, keep_last: int = MAX_MESSAGES_PER_USER) -> None:
    """Удаляет самые старые сообщения пользователя, оставляя только последние keep_last.

    Используется для управления длиной истории диалога.
    """
    total = db.query(Message).filter(Message.user_id == user_id).count()
    if total <= keep_last:
        return

    subquery = (
        db.query(Message.id)
        .filter(Message.user_id == user_id)
        .order_by(Message.created_at.desc())
        .offset(keep_last)
        .subquery()
    )

    if subquery is not None:
        ids_select = select(subquery.c.id)
        db.query(Message).filter(Message.id.in_(ids_select)).delete(synchronize_session=False)
        db.commit()
