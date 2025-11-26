from typing import List

from aiogram import types

from src.db import SessionLocal, Message, MAX_MESSAGES_PER_USER
from src.openai_client import SYSTEM_PROMPT, generate_answer
from src.rag import retrieve_relevant_chunks
from src.conversation_service import ConversationService


class LLMService:
    """Сервис оркестрации LLM-ответов: контекст диалога + RAG."""

    def __init__(self, conversation_service: ConversationService) -> None:
        self._conversation_service = conversation_service

    def generate_reply(self, tg_user: types.User, user_text: str) -> str:
        """Сохраняет сообщение пользователя, собирает контекст и получает ответ LLM."""
        self._conversation_service.add_user_message(tg_user, user_text)

        history: List[Message] = self._conversation_service.get_history(tg_user)
        if not history:
            return "⚠️ Достигнут дневной лимит токенов. Попробуйте снова завтра."

        rag_context = ""
        with SessionLocal() as db:
            chunks = retrieve_relevant_chunks(db, user_text, limit=3)
            if chunks:
                joined_chunks = "\n\n---\n\n".join(chunk.text for chunk in chunks)
                rag_context = (
                    "Вот релевантные фрагменты из базы знаний. Используй их при ответе, "
                    "ты можешь цитировать эти фрагменты дословно, если это полезно пользователю, "
                    "но не ссылайся напрямую на внутренние идентификаторы или пути к файлам.\n\n"
                    f"{joined_chunks}"
                )

        oa_messages: List[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
        if rag_context:
            oa_messages.append({"role": "system", "content": rag_context})

        for msg in history[-MAX_MESSAGES_PER_USER:]:
            oa_messages.append({"role": msg.role, "content": msg.content})

        reply_text = generate_answer(oa_messages)

        self._conversation_service.add_assistant_message(tg_user, reply_text)

        return reply_text
