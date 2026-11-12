# Веб-интерфейс юридического ассистента: FastAPI (web/server.py) + ядро (корневые *.py) + собранный фронт.
# Сборка в два этапа: Node собирает фронт, в итоговом образе только Python и статика.

# --- 1. Фронтенд (React + Vite) ---
FROM node:22-alpine AS frontend
WORKDIR /build
COPY web/frontend/package.json web/frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY web/frontend/ ./
RUN npm run build

# --- 2. Бэкенд и ядро ---
FROM python:3.13-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DATA_DIR=/app/data
WORKDIR /app

COPY requirements.txt /tmp/requirements-core.txt
COPY web/requirements.txt /tmp/requirements-web.txt
RUN pip install --no-cache-dir -r /tmp/requirements-core.txt -r /tmp/requirements-web.txt

# Ядро (плоские модули в корне), готовый индекс эмбеддингов, база знаний и демо-вопросы
COPY *.py index_vectors.json ./
COPY data/knowledge-base.zip data/demo_questions.csv ./data/
# Веб-бэкенд и собранный фронт
COPY web/__init__.py web/server.py ./web/
COPY --from=frontend /build/dist ./web/frontend/dist

RUN useradd --create-home --uid 10001 app && chown -R app /app
USER app

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=120s --retries=15 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/ready', timeout=4)"
CMD ["uvicorn", "web.server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
