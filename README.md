# LLM Telegram Support Bot

Telegram-агент с памятью диалога и поиском по документам (RAG).

## Запуск за 1 команду

```bash
make up
```

## Команды бота

- `/start` - Начать диалог
- `/clear` - Очистить историю
- `/stats` - Показать статистику токенов за сегодня

## Загрузка документов

```bash
docker exec -it llm_bot_app python -m src.rag path/to/file.txt "Название документа"
```

Где `path/to/file.txt` — путь к текстовому файлу в корне проекта (он монтируется в контейнер как `/app`).

## Технологии

- Python 3.13, aiogram 3.x, SQLAlchemy 2.0
- PostgreSQL + pgvector
- OpenAI GPT-4o-mini + эмбеддинги
- tiktoken (оптимизация токенов)
- Docker Compose
