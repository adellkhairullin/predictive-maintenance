"""Схемы запросов и ответов API с валидацией (Pydantic)."""
from __future__ import annotations

from typing import Annotated, Optional

from pydantic import BaseModel, ConfigDict, Field

# Показание датчика: конечное число (без NaN и бесконечностей) в физически разумных пределах.
Sensor = Annotated[float, Field(gt=0, lt=100_000, allow_inf_nan=False)]
OptionalSensor = Annotated[Optional[float], Field(default=None, ge=0, lt=100_000, allow_inf_nan=False)]
Setting = Annotated[Optional[float], Field(default=None, ge=-1, le=200, allow_inf_nan=False)]

EXAMPLE = {
    "unit_id": 1, "cycle": 1, "setting_1": -0.0007, "setting_2": -0.0004, "setting_3": 100.0,
    "s_1": 518.67, "s_2": 641.82, "s_3": 1589.7, "s_4": 1400.6, "s_5": 14.62, "s_6": 21.61,
    "s_7": 554.36, "s_8": 2388.06, "s_9": 9046.19, "s_10": 1.3, "s_11": 47.47, "s_12": 521.66,
    "s_13": 2388.02, "s_14": 8138.62, "s_15": 8.4195, "s_16": 0.03, "s_17": 392, "s_18": 2388,
    "s_19": 100.0, "s_20": 39.06, "s_21": 23.419,
}


class SensorReading(BaseModel):
    """Показания одного двигателя за один цикл работы.

    Обязательны 14 датчиков, которые использует модель. Остальные (s_1, s_5, s_6, s_10, s_16, s_18,
    s_19 и режимные параметры) можно передавать — они проверяются, но не используются.
    Неизвестные поля запрещены, чтобы опечатка в названии датчика не прошла незамеченной.
    """
    model_config = ConfigDict(extra="forbid", json_schema_extra={"example": EXAMPLE})

    unit_id: int = Field(gt=0, le=1_000_000, description="Номер двигателя")
    cycle: int = Field(gt=0, le=10_000, description="Номер цикла; циклы подаются по порядку, начиная с 1")
    setting_1: Setting
    setting_2: Setting
    setting_3: Setting
    s_1: OptionalSensor
    s_2: Sensor
    s_3: Sensor
    s_4: Sensor
    s_5: OptionalSensor
    s_6: OptionalSensor
    s_7: Sensor
    s_8: Sensor
    s_9: Sensor
    s_10: OptionalSensor
    s_11: Sensor
    s_12: Sensor
    s_13: Sensor
    s_14: Sensor
    s_15: Sensor
    s_16: OptionalSensor
    s_17: Sensor
    s_18: OptionalSensor
    s_19: OptionalSensor
    s_20: Sensor
    s_21: Sensor


class PredictionResponse(BaseModel):
    unit_id: int
    cycle: int
    probability: float = Field(description="Вероятность отказа в ближайшие 30 циклов")
    alarm: bool = Field(description="Тревога: вероятность >= порога")
    threshold: float
    model_version: str
    history_cycles: int = Field(description="Сколько циклов истории использовано для признаков")
    warnings: list[str] = Field(description="short_history — мало истории; out_of_range:<датчик> — "
                                            "значение вне диапазона обучающих данных")
    latency_ms: float = Field(description="Время обработки запроса сервером, мс")


class HealthResponse(BaseModel):
    status: str
    model_version: str
    engines_tracked: int
    online_store: str = Field(description="memory или redis")
    prediction_log: str


class EngineFeatures(BaseModel):
    unit_id: int
    cycle: int = Field(description="Цикл, для которого рассчитаны признаки")
    features: dict[str, float] = Field(description="71 признак (см. features.feature_registry)")
    feature_set_version: str
    source: str = Field(description="online — рассчитаны API; offline:<набор> — материализованы из PostgreSQL")
    probability: Optional[float] = None
    alarm: Optional[bool] = None
    model_version: Optional[str] = None
    updated_at: Optional[str] = None


class ModelInfo(BaseModel):
    name: str
    version: str
    algorithm: str
    n_features: int
    threshold: float
    threshold_rule: str
    horizon_cycles: int
    created_at: str
    metrics: dict
    libraries: dict
