import asyncio
import logging
import os
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from dotenv import load_dotenv
from src.db import init_db
from src.conversation_service import ConversationService
from src.llm_service import LLMService


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

if not TELEGRAM_BOT_TOKEN:
    logger.error("TELEGRAM_BOT_TOKEN is not set. Please configure it in the environment or .env file.")
    raise SystemExit(1)

bot = Bot(token=TELEGRAM_BOT_TOKEN)
dp = Dispatcher()

conversation_service = ConversationService()
llm_service = LLMService(conversation_service)


@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    """Обработчик команды /start"""
    greeting_text = (
        "👋 Привет! Я AI-агент поддержки.\n\n"
        "Просто напиши свой вопрос, а я отвечу, используя контекст нашего диалога.\n\n"
        "Команды:\n"
        "/start - показать это сообщение\n"
        "/clear - очистить историю диалога"
    )

    tg_user = message.from_user
    logger.info("/start from user_id=%s", tg_user.id)
    conversation_service.register_start(tg_user, greeting_text)

    await message.answer(greeting_text)


@dp.message(Command("clear"))
async def cmd_clear(message: types.Message):
    tg_user = message.from_user
    logger.info("/clear from user_id=%s", tg_user.id)
    conversation_service.clear_history(tg_user)

    await message.answer("✅ История диалога очищена (в dev-режиме)")


@dp.message(Command("stats", "stat"))
async def cmd_stats(message: types.Message):
    """Показывает статистику использования токенов за сегодня"""
    tg_user = message.from_user
    stats = conversation_service.get_stats(tg_user)

    text = (
        "📊 Статистика за сегодня:\n"
        f"• Сообщений: {stats['today_messages']}\n"
        f"• Токенов использовано: {stats['today_tokens']:,}\n"
        f"• Лимит токенов: {stats['max_daily_tokens']:,}"
    )
    await message.answer(text)


@dp.message()
async def echo_message(message: types.Message):
    """Эхо-обработчик — теперь отвечает через OpenAI GPT-4o-mini"""
    user_text = message.text

    if not user_text:
        await message.answer("Пока я понимаю только текстовые сообщения. Пожалуйста, отправь текст.")
        return

    tg_user = message.from_user
    try:
        reply_text = await asyncio.to_thread(llm_service.generate_reply, tg_user, user_text)
    except Exception as e:
        logger.error("LLM error: %s", e)
        await message.answer("⚠️ Ошибка на сервере. Попробуйте позже.")
        return

    await message.answer(reply_text)


async def main():
    """Запуск бота"""
    init_db()
    logger.info("🚀 Bot starting...")
    try:
        await dp.start_polling(bot)
    finally:
        logger.info("Bot shutdown, closing resources...")
        await bot.session.close()


if __name__ == "__main__":
    asyncio.run(main())