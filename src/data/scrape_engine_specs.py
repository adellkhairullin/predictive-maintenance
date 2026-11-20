"""Web scraping: справочные характеристики больших турбовентиляторных двигателей из Википедии.

Из раздела «Specifications» статьи двигателя извлекаются тип, максимальная (взлётная) тяга, сухая масса,
степень двухконтурности и общая степень повышения давления. Поддерживаются оба формата раздела:
таблица вариантов (берётся первый вариант) и список «Ключ: значение». Значения приводятся к СИ
(кН, кг); для диапазона берётся верхняя граница.

C-MAPSS моделирует двигатель класса тяги около 90 000 lbf (≈ 400 кН, Saxena et al., 2008) —
столбец thrust_vs_cmapss_pct показывает, насколько реальный двигатель близок к смоделированному.

Правила вежливого сбора: проверка robots.txt, User-Agent с назначением запросов (Wikimedia требует
указать контакт — переменная SCRAPER_USER_AGENT), пауза между запросами, повторы при сбоях.
Тексты Википедии распространяются по лицензии CC BY-SA 4.0: в результате сохраняется ссылка на источник.

Результат сохраняется в data/external/engine_specs.csv (снимок для воспроизводимости ETL).
Обновить снимок:
    .\\.venv\\Scripts\\python.exe -m src.data.scrape_engine_specs
"""
from __future__ import annotations

import logging
import os
import re
import sys
import time
import urllib.robotparser
from datetime import datetime, timezone

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from src.utils.db import PROJECT_ROOT

log = logging.getLogger("scraper")

BASE_URL = "https://en.wikipedia.org/wiki/"
ENGINES = {
    "GE90": "General_Electric_GE90",
    "GEnx": "General_Electric_GEnx",
    "CF6": "General_Electric_CF6",
    "PW4000": "Pratt_%26_Whitney_PW4000",
    "Trent 800": "Rolls-Royce_Trent_800",
    "Trent 1000": "Rolls-Royce_Trent_1000",
    "CFM56": "CFM_International_CFM56",
}
SNAPSHOT = PROJECT_ROOT / "data" / "external" / "engine_specs.csv"
DEFAULT_UA = "pdm-coursework-scraper/1.0 (+https://github.com; educational, non-commercial)"
CMAPSS_THRUST_KN = 400.0     # ≈ 90 000 lbf
LBF_TO_KN = 0.0044482216
LB_TO_KG = 0.45359237

COLUMNS = ["engine", "variant", "engine_type", "max_thrust_kn", "dry_weight_kg", "bypass_ratio",
           "overall_pressure_ratio", "thrust_vs_cmapss_pct", "raw_thrust", "raw_weight", "raw_bypass",
           "raw_opr", "source_url", "license", "scraped_at"]

_NUM = r"(\d[\d,]*(?:\.\d+)?)"
_RANGE = _NUM + r"(?:\s*[–\-‐]\s*" + _NUM + r")?"


def clean_text(s: str) -> str:
    """Убирает сноски вида [ 12 ] / [a], неразрывные пробелы и лишние пробелы."""
    s = re.sub(r"\[\s*[\w\s]{1,6}\s*\]", "", s.replace("\xa0", " "))
    return re.sub(r"\s+", " ", s).strip()


def _num(x: str) -> float:
    return float(x.replace(",", ""))


def _upper(m: re.Match) -> float:
    return _num(m.group(2) or m.group(1))


def parse_quantity(text_: str, units: dict[str, float]) -> float | None:
    """Первое значение с одной из единиц (в порядке приоритета), умноженное на коэффициент; диапазон — верхняя граница."""
    for unit, factor in units.items():
        m = re.search(_RANGE + r"\s*" + unit + r"\b", text_)
        if m:
            return round(_upper(m) * factor, 3)
    return None


def parse_ratio(text_: str) -> float | None:
    """«6.4:1», «>10:1», «33.9-40.7:1» -> верхняя граница; без «:1» — первое число."""
    m = re.search(_RANGE + r"\s*:\s*1\b", text_) or re.search(_RANGE, text_)
    return _upper(m) if m else None


def _norm_key(k: str) -> str:
    return clean_text(k).rstrip(":").strip().lower()


def extract_spec_pairs(html: str) -> tuple[dict[str, str], str | None]:
    """Пары «характеристика -> значение» из раздела Specifications и название варианта двигателя."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    heading = next((h for h in soup.find_all("h2") if "specification" in h.get_text().lower()), None)
    if heading is None:
        return {}, None
    m = re.search(r"\((.+)\)", heading.get_text())
    variant = clean_text(m.group(1)) if m else None
    start = heading.find_parent("div", class_="mw-heading") or heading
    pairs: dict[str, str] = {}
    for el in start.find_next_siblings():
        if el.name == "h2" or (el.name == "div" and "mw-heading2" in (el.get("class") or [])):
            break
        tables = [el] if el.name == "table" else el.find_all("table")
        for table in tables:
            for tr in table.find_all("tr"):
                cells = [clean_text(c.get_text(" ", strip=True)) for c in tr.find_all(["th", "td"])]
                if len(cells) >= 2 and cells[0]:
                    key = _norm_key(cells[0])
                    if key == "variant":
                        variant = variant or cells[1]
                    pairs.setdefault(key, cells[1])
        for li in el.find_all("li") if el.name != "li" else [el]:
            t = clean_text(li.get_text(" ", strip=True))
            if ":" in t[:45]:
                k, v = t.split(":", 1)
                pairs.setdefault(_norm_key(k), v.strip())
    return pairs, variant


def _pick(pairs: dict[str, str], *candidates, exclude=()) -> str | None:
    """Значение по первому подходящему ключу: точное совпадение, затем вхождение подстроки."""
    for c in candidates:
        if c in pairs:
            return pairs[c]
    for c in candidates:
        for k, v in pairs.items():
            if c in k and not any(e in k for e in exclude):
                return v
    return None


def normalize_specs(pairs: dict[str, str]) -> dict:
    thrust = _pick(pairs, "maximum thrust", "takeoff thrust", "thrust",
                   exclude=("weight", "specific", "/", "reverse"))
    weight = _pick(pairs, "dry weight", "weight", exclude=("thrust", "/"))
    bypass = _pick(pairs, "bypass ratio", "bpr")
    opr = _pick(pairs, "overall pressure ratio", "pressure ratio", "opr")
    thrust_kn = parse_quantity(thrust, {"kN": 1.0, "lbf": LBF_TO_KN}) if thrust else None
    return {
        "engine_type": pairs.get("type"),
        "max_thrust_kn": thrust_kn,
        "dry_weight_kg": parse_quantity(weight, {"kg": 1.0, "lb": LB_TO_KG}) if weight else None,
        "bypass_ratio": parse_ratio(bypass) if bypass else None,
        "overall_pressure_ratio": parse_ratio(opr) if opr else None,
        "thrust_vs_cmapss_pct": round(100 * (thrust_kn - CMAPSS_THRUST_KN) / CMAPSS_THRUST_KN, 1)
        if thrust_kn else None,
        "raw_thrust": thrust, "raw_weight": weight, "raw_bypass": bypass, "raw_opr": opr,
    }


class PoliteClient:
    """HTTP-клиент: robots.txt, User-Agent, пауза между запросами, повторы при сбоях."""

    def __init__(self, user_agent: str | None = None, delay: float = 1.0, retries: int = 3):
        import httpx
        self.ua = user_agent or os.getenv("SCRAPER_USER_AGENT", DEFAULT_UA)
        self.client = httpx.Client(headers={"User-Agent": self.ua}, timeout=30, follow_redirects=True)
        self.delay, self.retries = delay, retries
        self._robots: dict[str, urllib.robotparser.RobotFileParser] = {}
        self._last = 0.0

    def allowed(self, url: str) -> bool:
        root = "/".join(url.split("/")[:3])
        if root not in self._robots:
            rp = urllib.robotparser.RobotFileParser()
            rp.parse(self._get(root + "/robots.txt").text.splitlines())
            self._robots[root] = rp
        return self._robots[root].can_fetch(self.ua, url)

    def _get(self, url: str):
        for attempt in range(1, self.retries + 1):
            wait = self.delay - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
            try:
                r = self.client.get(url)
                if r.status_code == 200:
                    return r
                log.warning("%s -> HTTP %s (попытка %s)", url, r.status_code, attempt)
            except Exception as exc:
                log.warning("%s -> %s (попытка %s)", url, exc, attempt)
            time.sleep(2 ** attempt)
        raise RuntimeError(f"Не удалось загрузить {url}")

    def get(self, url: str) -> str:
        if not self.allowed(url):
            raise PermissionError(f"robots.txt запрещает {url}")
        return self._get(url).text


def scrape(client: PoliteClient | None = None, engines: dict[str, str] = ENGINES) -> pd.DataFrame:
    client = client or PoliteClient()
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rows = []
    for name, page in engines.items():
        url = BASE_URL + page
        try:
            pairs, variant = extract_spec_pairs(client.get(url))
        except Exception as exc:
            log.error("%s: %s", name, exc)
            continue
        rows.append({"engine": name, "variant": variant, **normalize_specs(pairs), "source_url": url,
                     "license": "CC BY-SA 4.0 (Wikipedia)", "scraped_at": now})
        log.info("%s: %s", name, {k: rows[-1][k] for k in ("max_thrust_kn", "bypass_ratio", "overall_pressure_ratio")})
    return pd.DataFrame(rows, columns=COLUMNS)


def load_reference_specs(engine: Engine, path=SNAPSHOT) -> int:
    """Загружает снимок в enriched.engine_reference_specs (таблица пересоздаётся)."""
    df = pd.read_csv(path)
    with engine.begin() as conn:
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS enriched"))
        df.to_sql("engine_reference_specs", conn, schema="enriched", if_exists="replace", index=False)
        conn.execute(text("ALTER TABLE enriched.engine_reference_specs ADD PRIMARY KEY (engine)"))
    return len(df)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stdout)
    result = scrape()
    if result.empty:
        sys.exit("Ничего не собрано — снимок не перезаписан")
    SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(SNAPSHOT, index=False)
    print(result[["engine", "variant", "max_thrust_kn", "dry_weight_kg", "bypass_ratio",
                  "overall_pressure_ratio", "thrust_vs_cmapss_pct"]].to_string(index=False))
    print(f"\nСохранено: {SNAPSHOT}")
