"""Тесты парсера справочных характеристик. Сеть не используется: HTML-фрагменты повторяют
два формата раздела Specifications в статьях Википедии (таблица вариантов и список «Ключ: значение»)."""
import pandas as pd
import pytest

from src.data.scrape_engine_specs import (COLUMNS, SNAPSHOT, clean_text, extract_spec_pairs, normalize_specs,
                                          parse_quantity, parse_ratio, scrape)

TABLE_PAGE = """
<html><body>
<div class="mw-heading mw-heading2"><h2>History</h2></div><p>…</p>
<div class="mw-heading mw-heading2"><h2>Specifications</h2></div>
<table class="wikitable">
<tr><th>Variant</th><th>-94B</th><th>-115B</th></tr>
<tr><th>Type</th><td>Dual rotor, axial flow, high-bypass turbofan</td></tr>
<tr><th>Weight [ b ]</th><td>17,400 lb (7,893 kg )</td><td>19,316 lb (8,762 kg)</td></tr>
<tr><th>Takeoff thrust</th><td>81,070–97,300 lbf (360.6–432.8 kN)</td><td>115,540 lbf</td></tr>
<tr><th>Thrust/weight</th><td>5.6</td></tr>
<tr><th>Bypass ratio</th><td>8.7–9:1</td></tr>
<tr><th>Overall pressure ratio</th><td>40</td></tr>
</table>
<div class="mw-heading mw-heading2"><h2>See also</h2></div>
<table class="wikitable"><tr><th>Takeoff thrust</th><td>1 kN</td></tr></table>
</body></html>"""

LIST_PAGE = """
<html><body>
<div class="mw-heading mw-heading2"><h2>Specifications (Trent 800)</h2></div>
<p>Data from EASA</p>
<div class="mw-heading mw-heading4"><h4>General characteristics</h4></div>
<ul><li><b>Type:</b> Three-shaft high bypass turbofan engine</li>
<li><b>Dry weight:</b> 6,078&nbsp;kg (13,400&nbsp;lb)</li></ul>
<ul><li><b>Maximum thrust</b> : 340.6-413.4 kN (76,580-92,940 lbf)</li>
<li><b>Overall pressure ratio</b> : 33.9-40.7:1</li>
<li><b>Bypass ratio</b> : 6.4:1 [ 21 ]</li>
<li><b>Thrust-to-weight ratio</b> : 5.7-6.9</li></ul>
</body></html>"""


def test_table_format_takes_first_variant_and_ignores_other_sections():
    pairs, variant = extract_spec_pairs(TABLE_PAGE)
    assert variant == "-94B"
    s = normalize_specs(pairs)
    assert s["max_thrust_kn"] == 432.8          # верхняя граница диапазона, кН в приоритете
    assert s["dry_weight_kg"] == 7893.0          # сноска [ b ] в ключе убрана
    assert s["bypass_ratio"] == 9.0 and s["overall_pressure_ratio"] == 40.0
    assert s["thrust_vs_cmapss_pct"] == 8.2
    assert s["engine_type"].startswith("Dual rotor")


def test_list_format_and_variant_from_heading():
    pairs, variant = extract_spec_pairs(LIST_PAGE)
    assert variant == "Trent 800"
    s = normalize_specs(pairs)
    assert (s["max_thrust_kn"], s["dry_weight_kg"], s["bypass_ratio"], s["overall_pressure_ratio"]) == \
           (413.4, 6078.0, 6.4, 40.7)
    assert "thrust-to-weight ratio" in pairs and s["raw_thrust"].startswith("340.6")


def test_page_without_specifications():
    assert extract_spec_pairs("<html><h2>History</h2></html>") == ({}, None)
    assert normalize_specs({})["max_thrust_kn"] is None


@pytest.mark.parametrize("text,units,expected", [
    ("97,300 lbf", {"kN": 1.0, "lbf": 0.0044482216}, 432.812),
    ("2,139–2,200 kg 4,716–4,850 lb", {"kg": 1.0, "lb": 0.45359237}, 2200.0),
    ("13,400 lb", {"kg": 1.0, "lb": 0.45359237}, 6078.138),   # 13 400 × 0,45359237 = 6078,1378
    ("no number", {"kN": 1.0}, None),
])
def test_parse_quantity(text, units, expected):
    assert parse_quantity(text, units) == expected


@pytest.mark.parametrize("text,expected", [("6.4:1", 6.4), (">10:1", 10.0), ("33.9-40.7:1", 40.7), ("40", 40.0),
                                           ("n/a", None)])
def test_parse_ratio(text, expected):
    assert parse_ratio(text) == expected


def test_clean_text_removes_footnotes():
    assert clean_text("Bypass ratio [ 55 ] :\xa0 6.4 [a]") == "Bypass ratio : 6.4"


class FakeClient:
    def __init__(self, pages):
        self.pages = pages

    def get(self, url):
        if url.endswith("Broken"):
            raise RuntimeError("HTTP 500")
        return self.pages[url.rsplit("/", 1)[1]]


def test_scrape_skips_failed_pages():
    df = scrape(FakeClient({"A": TABLE_PAGE, "B": LIST_PAGE}), engines={"A": "A", "X": "Broken", "B": "B"})
    assert list(df.columns) == COLUMNS
    assert df["engine"].tolist() == ["A", "B"]
    assert df["source_url"].str.startswith("https://en.wikipedia.org/wiki/").all()


def test_snapshot_in_repository_is_valid():
    df = pd.read_csv(SNAPSHOT)
    assert list(df.columns) == COLUMNS and df["engine"].is_unique and len(df) >= 5
    assert df["max_thrust_kn"].between(50, 600).all()
    assert df["bypass_ratio"].between(1, 15).all()
