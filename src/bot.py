import asyncio
import os
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from dotenv import load_dotenv

# Загрузка переменных окружения
load_dotenv()
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

# Инициализация бота и диспетчера
bot = Bot(token=TELEGRAM_BOT_TOKEN)
dp = Dispatcher()

@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    """Обработчик команды /start"""
    await message.answer(
        "👋 Привет! Я AI-агент поддержки.\n\n"
        "Просто напиши свой вопрос, а я отвечу, используя контекст нашего диалога.\n\n"
        "Команды:\n"
        "/start - показать это сообщение\n"
        "/clear - очистить историю диалога"
    )

@dp.message(Command("clear"))
async def cmd_clear(message: types.Message):
    """Очищаем историю (заглушка)"""
    # Пока просто сообщение, потом добавим очистку в БД
    await message.answer("✅ История диалога очищена (в dev-режиме)")

@dp.message()
async def echo_message(message: types.Message):
    """Эхо-обработчик — пока просто повторяем сообщение"""
    # TODO: Здесь будет интеграция с OpenAI
    user_text = message.text
    await message.answer(f"🤖 Вы сказали: {user_text}\n\n(Скоро я буду отвечать через GPT-4o-mini)")

async def main():
    """Запуск бота"""
    print("🚀 Bot starting...")
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())