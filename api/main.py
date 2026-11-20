"""FastAPI-сервис: вероятность отказа двигателя в ближайшие 30 циклов по показаниям датчиков.

Запуск из корня проекта:
    .\\.venv\\Scripts\\python.exe -m uvicorn api.main:app --port 8000
Документация и форма для пробных запросов: http://127.0.0.1:8000/docs

Переменные окружения:
    PDM_MODEL_NAME      имя модели в папке models (по умолчанию final_v2)
    PDM_PREDICTION_LOG  db — писать журнал в PostgreSQL (по умолчанию), none — не писать
"""
from __future__ import annotations

import math
import os
import time
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from api.history import HistoryStore, InMemoryHistoryStore, needed_rows
from api.prediction_log import PredictionLogger, make_logger
from api.schemas import HealthResponse, ModelInfo, PredictionResponse, SensorReading
from src.data.features import FEATURE_SENSORS, last_row_features
from src.models.input_ranges import load_ranges
from src.models.predict import FailurePredictor

SHORT_HISTORY_CYCLES = 6   # на циклах 1–5 тренд неустойчив (см. notebooks/04_final_model.ipynb)


def _json_safe(value):
    """NaN и бесконечность нельзя записать в JSON — заменяем их строкой (для текста ошибки 422)."""
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _warm_up(predictor: FailurePredictor) -> None:
    """Пробное предсказание при старте: первый вызов догружает библиотеки (~2 с), и без прогрева
    эту задержку получил бы первый клиент."""
    row = {"cycle": 1, **{s: 1.0 for s in FEATURE_SENSORS}}
    features = last_row_features([row])
    predictor.model.predict_proba(pd.DataFrame([features])[predictor.features])


async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": _json_safe(jsonable_encoder(exc.errors()))})


def create_app(predictor: FailurePredictor | None = None, store: HistoryStore | None = None,
               prediction_logger: PredictionLogger | None = None,
               ranges: dict | None = None) -> FastAPI:
    """Создаёт приложение. Зависимости можно подменить (используется в тестах)."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.predictor = predictor or FailurePredictor(os.getenv("PDM_MODEL_NAME", "final_v2"))
        app.state.store = store if store is not None else InMemoryHistoryStore()
        app.state.logger = prediction_logger or make_logger(os.getenv("PDM_PREDICTION_LOG", "db"))
        app.state.ranges = ranges if ranges is not None else load_ranges()
        _warm_up(app.state.predictor)
        yield
        app.state.logger.close()   # дописать журнал перед остановкой

    app = FastAPI(title="Predictive Maintenance API",
                  description="Предсказание отказа турбовентиляторного двигателя (NASA C-MAPSS FD001) "
                              "в горизонте 30 циклов по показаниям датчиков.",
                  version="1.0.0", lifespan=lifespan)
    app.add_exception_handler(RequestValidationError, validation_error_handler)

    @app.get("/health", response_model=HealthResponse, tags=["service"])
    def health():
        return HealthResponse(status="ok", model_version=app.state.predictor.version,
                              engines_tracked=len(app.state.store), prediction_log=app.state.logger.name)

    @app.get("/model", response_model=ModelInfo, tags=["service"])
    def model_info():
        p = app.state.predictor
        m = p.meta
        return ModelInfo(
            name=p.name, version=p.version, algorithm=m.get("algorithm", ""), n_features=len(p.features),
            threshold=p.threshold, threshold_rule=m.get("threshold_rule", ""),
            horizon_cycles=m.get("horizon_cycles", 30), created_at=m.get("created_at", ""),
            metrics={"cv_pr_auc_mean": m.get("cv", {}).get("pr_auc_mean"),
                     "test_recall": m.get("test", {}).get("recall"), "test_fpr": m.get("test", {}).get("fpr")},
            libraries=m.get("libraries", {}))

    @app.post("/predict", response_model=PredictionResponse, tags=["prediction"],
              responses={409: {"description": "Цикл подан не по порядку (пропуск или повтор)"},
                         422: {"description": "Ошибка валидации входных данных"}})
    def predict(reading: SensorReading):
        t0 = time.perf_counter()
        st, p = app.state.store, app.state.predictor
        row = reading.model_dump()
        with st.lock:   # проверка порядка и добавление в историю — атомарно для одного процесса
            last = st.last_cycle(reading.unit_id)
            expected = 1 if last is None else last + 1
            if reading.cycle != expected:
                kind = "повтор уже обработанного цикла" if last is not None and reading.cycle <= last else "пропуск циклов"
                raise HTTPException(status.HTTP_409_CONFLICT, detail={
                    "error": f"Цикл подан не по порядку: {kind}", "unit_id": reading.unit_id,
                    "received_cycle": reading.cycle, "expected_cycle": expected})
            history = needed_rows(st.history(reading.unit_id) + [row])
            features = last_row_features(history)
            proba = float(p.model.predict_proba(pd.DataFrame([features])[p.features])[:, 1][0])
            st.append(reading.unit_id, row)

        warnings = []
        if reading.cycle < SHORT_HISTORY_CYCLES:
            warnings.append("short_history")
        for s in FEATURE_SENSORS:
            r = app.state.ranges.get(s)
            if r and not (r["min"] <= row[s] <= r["max"]):
                warnings.append(f"out_of_range:{s}")

        response = PredictionResponse(
            unit_id=reading.unit_id, cycle=reading.cycle, probability=proba, alarm=proba >= p.threshold,
            threshold=p.threshold, model_version=p.version, history_cycles=len(history), warnings=warnings,
            latency_ms=round((time.perf_counter() - t0) * 1000, 3))
        app.state.logger.log({   # только постановка в очередь; запись в БД — в отдельном потоке
            "unit_id": response.unit_id, "cycle": response.cycle, "model_version": response.model_version,
            "probability": response.probability, "alarm": response.alarm, "threshold": response.threshold,
            "latency_ms": response.latency_ms, "warnings": response.warnings})
        return response

    @app.delete("/engines/{unit_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["prediction"],
                responses={404: {"description": "Истории по двигателю нет"}})
    def reset_engine(unit_id: int):
        """Сбросить историю двигателя (например, после ремонта): следующий цикл снова начинается с 1."""
        with app.state.store.lock:
            if not app.state.store.reset(unit_id):
                raise HTTPException(status.HTTP_404_NOT_FOUND, detail=f"Нет истории по двигателю {unit_id}")
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    return app


app = create_app()
