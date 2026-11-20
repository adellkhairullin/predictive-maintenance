"""Замер времени ответа API (гипотеза H4): 1000 последовательных запросов /predict после прогрева.

Запросы — реальные показания тестовых двигателей FD001, подаваемые цикл за циклом, как в потоке.
Время измеряется на стороне клиента (полный HTTP-запрос) и на стороне сервера (поле latency_ms).

Сначала запустите сервис в отдельном терминале:
    .\\.venv\\Scripts\\python.exe -m uvicorn api.main:app --port 8000
Затем:
    .\\.venv\\Scripts\\python.exe scripts\\benchmark_api.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.load_raw import RAW_DIR, read_readings  # noqa: E402

OUT = ROOT / "docs" / "h4_latency.json"


def stream(df):
    """Показания в порядке «все циклы двигателя 1, затем двигателя 2, ...» с уникальными номерами."""
    for r in df.sort_values(["unit_id", "cycle"]).to_dict("records"):
        yield {k: (int(v) if k in ("unit_id", "cycle") else float(v)) for k, v in r.items()}


def main(url: str, n: int, warmup: int, unit_offset: int) -> dict:
    df = read_readings(RAW_DIR / "test_FD001.txt")
    df["unit_id"] += unit_offset   # чтобы не пересечься с историей, уже накопленной сервисом
    client_ms, server_ms, errors = [], [], 0
    with httpx.Client(base_url=url, timeout=10) as client:
        assert client.get("/health").status_code == 200, "сервис не отвечает"
        for i, body in enumerate(stream(df)):
            if i >= warmup + n:
                break
            t0 = time.perf_counter()
            r = client.post("/predict", json=body)
            dt = (time.perf_counter() - t0) * 1000
            if r.status_code != 200:
                errors += 1
                continue
            if i >= warmup:
                client_ms.append(dt)
                server_ms.append(r.json()["latency_ms"])
    pct = lambda a, q: round(float(np.percentile(a, q)), 2)
    result = {
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "requests": len(client_ms), "warmup": warmup, "errors": errors,
        "client_ms": {"p50": pct(client_ms, 50), "p95": pct(client_ms, 95), "p99": pct(client_ms, 99),
                      "max": round(max(client_ms), 2)},
        "server_ms": {"p50": pct(server_ms, 50), "p95": pct(server_ms, 95), "p99": pct(server_ms, 99)},
        "requirement_p95_ms": 50,
    }
    result["h4_confirmed"] = bool(result["client_ms"]["p95"] < 50 and errors == 0)
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("-n", type=int, default=1000)
    ap.add_argument("--warmup", type=int, default=50)
    # номера двигателей в API ограничены 1 000 000: смещение кратно 1000 и не больше 900 000
    ap.add_argument("--unit-offset", type=int, default=(int(time.time()) % 900 + 1) * 1000)
    a = ap.parse_args()
    res = main(a.url, a.n, a.warmup, a.unit_offset)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=2))
