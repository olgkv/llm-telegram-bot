PYTHON_BIN ?= python3.13
VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

.PHONY: help venv install run test docker-build up down logs ps bash demo load-doc

help:
	@echo "Available targets:"
	@echo "  make venv          - create local virtualenv (.venv)"
	@echo "  make install       - install Python dependencies into .venv"
	@echo "  make run           - run bot locally via Python from .venv"
	@echo "  make test          - run pytest test suite from .venv"
	@echo "  make docker-build  - build Docker image llm-telegram-bot"
	@echo "  make up            - start services via docker compose (detached)"
	@echo "  make down          - stop services via docker compose"
	@echo "  make logs          - follow bot service logs"
	@echo "  make ps            - show docker compose services status"
	@echo "  make bash          - open shell in running bot container"
	@echo "  make demo          - start stack and show quickstart instructions"
	@echo "  make load-doc      - ingest sample RAG document into bot DB"

venv:
	$(PYTHON_BIN) -m venv $(VENV)

install: venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

run: install
	$(PYTHON) -m src.bot

test: install
	$(PYTHON) -m pytest

docker-build:
	docker build -t llm-telegram-bot .

up:
	docker compose up -d

down:
	docker compose down

logs:
	docker compose logs -f bot

ps:
	docker compose ps

bash:
	docker exec -it llm_bot_app /bin/bash

demo: up
	@echo "🚀 Бот запущен! Вот что можно сделать:"
	@echo "1. Напишите боту в Telegram /start"
	@echo "2. Загрузите документ: make load-doc"
	@echo "3. Смотрите логи: make logs"
	@echo "4. Остановите: make down"

load-doc: up
	docker exec -it llm_bot_app python -m src.rag data/sample.txt "Тестовый документ"
	@echo "✅ Документ загружен! Задавайте боту вопросы по его содержанию."

