import os
import logging
import time
from typing import List, Dict

from openai import OpenAI


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

logger = logging.getLogger(__name__)


def _create_client() -> OpenAI | None:
    """Создаёт клиента OpenAI на основе OPENAI_API_KEY.

    Возвращает None, если ключ не задан.
    """
    if not OPENAI_API_KEY:
        return None

    return OpenAI(api_key=OPENAI_API_KEY, timeout=20.0)


client = _create_client()


SYSTEM_PROMPT = (
    "Ты — дружелюбный и внимательный AI-ассистент службы поддержки. "
    "Отвечай кратко, по делу и на русском языке, если пользователь не просит иное. "
    "Используй историю диалога и будь последовательным в ответах. "
    "Если чего-то не знаешь или не хватает контекста, честно говори об этом. "
    "Если в системных сообщениях тебе переданы фрагменты документов, ты можешь безопасно "
    "цитировать их дословно и опираться на них в ответах, но не утверждай, что имеешь доступ "
    "к произвольным файлам вне этих фрагментов."
)


def generate_answer(messages: List[Dict[str, str]]) -> str:
    """Отправляет сообщения в OpenAI и возвращает текст ответа.

    messages: список словарей вида {"role": "system|user|assistant", "content": "..."}
    """
    if client is None:
        return (
            "⚠️ OpenAI API ключ не настроен. "
            "Добавьте OPENAI_API_KEY в переменные окружения."
        )

    max_attempts = 3
    base_delay = 1.0

    last_error: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=messages,
                temperature=0.4,
            )

            usage = getattr(response, "usage", None)
            if usage is not None:
                logger.info(
                    "OpenAI usage: prompt=%s, completion=%s, total=%s",
                    getattr(usage, "prompt_tokens", None),
                    getattr(usage, "completion_tokens", None),
                    getattr(usage, "total_tokens", None),
                )

            choice = response.choices[0]
            return choice.message.content or "Извините, не удалось сформировать ответ."
        except Exception as exc:  # noqa: BLE001 - хотим перехватить любые сетевые/HTTP ошибки
            last_error = exc
            logger.warning("OpenAI call failed on attempt %s/%s: %s", attempt, max_attempts, exc)

            if attempt == max_attempts:
                break

            time.sleep(base_delay * attempt)

    logger.error("OpenAI call failed after %s attempts: %s", max_attempts, last_error)
    return (
        "⚠️ Ошибка при обращении к OpenAI. "
        "Пожалуйста, проверьте API-ключ и настройки, либо попробуйте позже."
    )
