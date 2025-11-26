from types import SimpleNamespace

import pytest

from src.llm_service import LLMService


class DummyMessage:
    def __init__(self, role: str, content: str) -> None:
        self.role = role
        self.content = content


class FakeConversationService:
    def __init__(self) -> None:
        self.history = []  # type: ignore[var-annotated]
        self.user_messages = []
        self.assistant_messages = []

    def register_start(self, tg_user, greeting_text: str) -> None:  # pragma: no cover - не используется здесь
        self.add_user_message(tg_user, "/start")
        self.add_assistant_message(tg_user, greeting_text)

    def clear_history(self, tg_user) -> None:  # pragma: no cover - не используется здесь
        self.history.clear()

    def add_user_message(self, tg_user, content: str) -> None:
        self.user_messages.append((tg_user.id, content))
        self.history.append(DummyMessage("user", content))

    def add_assistant_message(self, tg_user, content: str) -> None:
        self.assistant_messages.append((tg_user.id, content))
        self.history.append(DummyMessage("assistant", content))

    def get_history(self, tg_user):
        return list(self.history)


class LimitedConversationService(FakeConversationService):
    """Фейковый сервис, который имитирует превышенный дневной лимит токенов.

    add_user_message ничего не добавляет в историю, а get_history всегда
    возвращает пустой список.
    """

    def add_user_message(self, tg_user, content: str) -> None:  # pragma: no cover - простая заглушка
        # Сообщение не сохраняется, как если бы дневной лимит был превышен.
        return

    def get_history(self, tg_user):  # noqa: D401 - простая реализация для тестов
        return []


@pytest.fixture()
def fake_user():
    return SimpleNamespace(id=123, username="testuser")


def test_generate_reply_uses_llm_and_updates_history(monkeypatch, fake_user):
    fake_conv = FakeConversationService()

    fake_conv.history = [
        DummyMessage("user", "hi"),
        DummyMessage("assistant", "hello"),
    ]

    captured_messages = {}

    def fake_generate_answer(messages):
        captured_messages["messages"] = messages
        return "LLM-REPLY"

    def fake_retrieve_relevant_chunks(db, query: str, limit: int = 3):  # pragma: no cover
        return [SimpleNamespace(text="chunk-1"), SimpleNamespace(text="chunk-2")]

    from contextlib import contextmanager

    @contextmanager
    def dummy_session():
        yield None

    monkeypatch.setattr("src.llm_service.generate_answer", fake_generate_answer)
    monkeypatch.setattr("src.llm_service.retrieve_relevant_chunks", fake_retrieve_relevant_chunks)
    monkeypatch.setattr("src.llm_service.SessionLocal", lambda: dummy_session())

    service = LLMService(fake_conv)

    reply = service.generate_reply(fake_user, "How are you?")

    assert reply == "LLM-REPLY"

    assert any(m[1] == "How are you?" for m in fake_conv.user_messages)
    assert any(m[1] == "LLM-REPLY" for m in fake_conv.assistant_messages)

    msgs = captured_messages["messages"]
    assert msgs[0]["role"] == "system"
    assert any(m["role"] == "system" and "chunk-1" in m["content"] for m in msgs[1:3])
    roles = [m["role"] for m in msgs]
    assert "user" in roles and "assistant" in roles


def test_generate_reply_respects_daily_token_limit(monkeypatch, fake_user):
    fake_conv = LimitedConversationService()

    def fake_generate_answer(messages):  # pragma: no cover - не должен вызываться
        raise AssertionError("generate_answer should not be called when daily limit is exceeded")

    def fake_retrieve_relevant_chunks(db, query: str, limit: int = 3):  # pragma: no cover - не должен вызываться
        raise AssertionError("retrieve_relevant_chunks should not be called when daily limit is exceeded")

    from contextlib import contextmanager

    @contextmanager
    def dummy_session():  # pragma: no cover - не должен вызываться
        yield None

    monkeypatch.setattr("src.llm_service.generate_answer", fake_generate_answer)
    monkeypatch.setattr("src.llm_service.retrieve_relevant_chunks", fake_retrieve_relevant_chunks)
    monkeypatch.setattr("src.llm_service.SessionLocal", lambda: dummy_session())

    service = LLMService(fake_conv)

    reply = service.generate_reply(fake_user, "Hello")

    assert "дневной лимит токенов" in reply
