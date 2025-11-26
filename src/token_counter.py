import os
import tiktoken
from datetime import datetime, timezone

from sqlalchemy import func
from sqlalchemy.orm import Session

from src.db import Message



MAX_MESSAGE_TOKENS = int(os.getenv("MAX_MESSAGE_TOKENS", "4000"))


def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    """Подсчёт количества токенов в тексте для указанной модели.

    Используется для грубого контроля длины сообщений, попадающих в историю
    диалога и в контекст LLM.
    """
    if not text:
        return 0

    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    return len(encoding.encode(text))


def check_daily_limit(db: Session, user_id: int, max_tokens: int = 50000) -> bool:
    """Проверяет, не превышен ли дневной лимит токенов для пользователя."""
    today = datetime.now(timezone.utc).date()
    total = (
        db.query(func.coalesce(func.sum(Message.token_count), 0))
        .filter(
            Message.user_id == user_id,
            func.date(Message.created_at) == today,
        )
        .scalar()
    )
    return int(total or 0) < max_tokens
