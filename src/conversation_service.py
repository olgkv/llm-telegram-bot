from typing import List
from datetime import datetime, timezone

from aiogram import types
from sqlalchemy import func
from sqlalchemy.orm import Session

from src.db import SessionLocal, User, Message, trim_old_messages, MAX_MESSAGES_PER_USER
from src.token_counter import count_tokens, check_daily_limit, MAX_MESSAGE_TOKENS


class ConversationService:
    """Сервис для работы с пользователями и сообщениями (историей диалога)."""

    def __init__(self, session_factory=SessionLocal) -> None:
        self._session_factory = session_factory

    def _get_db(self) -> Session:
        return self._session_factory()

    def _get_or_create_user(self, db: Session, tg_user: types.User) -> User:
        user = db.query(User).filter(User.telegram_id == tg_user.id).first()
        if user is None:
            user = User(
                telegram_id=tg_user.id,
                username=tg_user.username,
                first_name=tg_user.first_name,
                last_name=tg_user.last_name,
            )
            db.add(user)
            db.commit()
            db.refresh(user)
        return user

    def _save_message(self, db: Session, user: User, role: str, content: str) -> None:
        if not content:
            return

        tokens = count_tokens(content)
        if tokens > MAX_MESSAGE_TOKENS:
            return

        if not check_daily_limit(db, user.id):
            return

        message = Message(user_id=user.id, role=role, content=content, token_count=tokens)
        db.add(message)
        db.commit()
        trim_old_messages(db, user.id, keep_last=MAX_MESSAGES_PER_USER)

    def register_start(self, tg_user: types.User, greeting_text: str) -> None:
        """Регистрирует пользователя и сохраняет приветственное сообщение."""
        with self._get_db() as db:
            user = self._get_or_create_user(db, tg_user)
            self._save_message(db, user, "user", "/start")
            self._save_message(db, user, "assistant", greeting_text)

    def clear_history(self, tg_user: types.User) -> None:
        """Очищает историю диалога пользователя."""
        with self._get_db() as db:
            user = self._get_or_create_user(db, tg_user)
            db.query(Message).filter(Message.user_id == user.id).delete()
            db.commit()

    def add_user_message(self, tg_user: types.User, content: str) -> None:
        with self._get_db() as db:
            user = self._get_or_create_user(db, tg_user)
            self._save_message(db, user, "user", content)

    def add_assistant_message(self, tg_user: types.User, content: str) -> None:
        with self._get_db() as db:
            user = self._get_or_create_user(db, tg_user)
            self._save_message(db, user, "assistant", content)

    def get_history(self, tg_user: types.User) -> List[Message]:
        """Возвращает историю сообщений пользователя в хронологическом порядке."""
        with self._get_db() as db:
            user = self._get_or_create_user(db, tg_user)
            messages = (
                db.query(Message)
                .filter(Message.user_id == user.id)
                .order_by(Message.created_at.asc())
                .all()
            )
            return list(messages)

    def get_stats(self, tg_user: types.User) -> dict:
        """Возвращает статистику токенов за сегодня для пользователя."""
        with self._get_db() as db:
            user = self._get_or_create_user(db, tg_user)
            today = datetime.now(timezone.utc).date()

            total_tokens = (
                db.query(func.coalesce(func.sum(Message.token_count), 0))
                .filter(
                    Message.user_id == user.id,
                    func.date(Message.created_at) == today,
                )
                .scalar()
            )

            message_count = (
                db.query(Message)
                .filter(
                    Message.user_id == user.id,
                    func.date(Message.created_at) == today,
                )
                .count()
            )

            return {
                "today_tokens": int(total_tokens or 0),
                "today_messages": message_count,
                "max_daily_tokens": 50000,
            }
