from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.conversation_service import ConversationService
from src.db import Base, User, Message, MAX_MESSAGES_PER_USER


def create_sqlite_session_factory():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    return sessionmaker(bind=engine, autoflush=False, autocommit=False)


def make_fake_user(user_id: int = 123) -> SimpleNamespace:
    return SimpleNamespace(
        id=user_id,
        username="testuser",
        first_name="Test",
        last_name="User",
    )


def test_register_start_creates_user_and_messages():
    SessionFactory = create_sqlite_session_factory()
    service = ConversationService(session_factory=SessionFactory)
    tg_user = make_fake_user()

    greeting_text = "hello"
    service.register_start(tg_user, greeting_text)

    with SessionFactory() as db:
        users = db.query(User).all()
        messages = db.query(Message).order_by(Message.created_at.asc()).all()

    assert len(users) == 1
    assert users[0].telegram_id == tg_user.id

    assert len(messages) == 2
    assert messages[0].role == "user" and messages[0].content == "/start"
    assert messages[1].role == "assistant" and messages[1].content == greeting_text


def test_clear_history_removes_all_messages():
    SessionFactory = create_sqlite_session_factory()
    service = ConversationService(session_factory=SessionFactory)
    tg_user = make_fake_user()

    service.register_start(tg_user, "hello")
    service.clear_history(tg_user)

    with SessionFactory() as db:
        messages = db.query(Message).all()

    assert messages == []


def test_trim_old_messages_keeps_only_last_max():
    SessionFactory = create_sqlite_session_factory()
    service = ConversationService(session_factory=SessionFactory)
    tg_user = make_fake_user()

    total = MAX_MESSAGES_PER_USER + 5
    for i in range(total):
        service.add_user_message(tg_user, f"msg-{i}")

    with SessionFactory() as db:
        messages = db.query(Message).order_by(Message.created_at.asc()).all()

    assert len(messages) == MAX_MESSAGES_PER_USER
    remaining_contents = [m.content for m in messages]
    expected = [f"msg-{i}" for i in range(total - MAX_MESSAGES_PER_USER, total)]
    assert remaining_contents == expected


def test_add_user_message_stores_token_count():
    SessionFactory = create_sqlite_session_factory()
    service = ConversationService(session_factory=SessionFactory)
    tg_user = make_fake_user()

    service.add_user_message(tg_user, "short message")

    with SessionFactory() as db:
        messages = db.query(Message).all()

    assert len(messages) == 1
    assert messages[0].token_count > 0


def test_add_user_message_skips_when_too_many_tokens(monkeypatch):
    SessionFactory = create_sqlite_session_factory()
    service = ConversationService(session_factory=SessionFactory)
    tg_user = make_fake_user()

    monkeypatch.setattr("src.conversation_service.count_tokens", lambda content: 5001)

    service.add_user_message(tg_user, "very long message")

    with SessionFactory() as db:
        messages = db.query(Message).all()

    assert messages == []


def test_add_user_message_respects_daily_limit(monkeypatch):
    SessionFactory = create_sqlite_session_factory()
    service = ConversationService(session_factory=SessionFactory)
    tg_user = make_fake_user()

    monkeypatch.setattr("src.conversation_service.check_daily_limit", lambda db, user_id: False)

    service.add_user_message(tg_user, "any message")

    with SessionFactory() as db:
        messages = db.query(Message).all()

    assert messages == []
