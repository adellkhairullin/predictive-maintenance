# Образ приложения: ETL, API, обучение. Сборка из корня проекта (см. docker-compose.yml).
#   runtime — для работы (ETL, API, обучение финальной модели)
#   dev     — runtime + тесты

# ---------- runtime ----------
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Сначала только зависимости: при изменении кода этот слой берётся из кэша
COPY requirements-runtime.txt .
RUN pip install -r requirements-runtime.txt

COPY sql/ sql/
COPY src/ src/
COPY api/ api/
COPY models/ models/
COPY data/raw/ data/raw/

# Запуск не от root
RUN useradd --create-home --uid 1000 app \
    && mkdir -p data/processed data/samples \
    && chown -R app:app /app
USER app

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# ---------- dev: для запуска тестов ----------
FROM runtime AS dev
USER root
# версии совпадают с requirements.txt
RUN pip install pytest==9.2.0 pytest-cov==7.1.0 httpx==0.28.1
COPY pytest.ini .
COPY tests/ tests/
RUN chown -R app:app /app
USER app
CMD ["pytest", "-v"]
