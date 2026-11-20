"""Сборка PDF отчёта и презентации из HTML (docs/src) печатью в Microsoft Edge (headless).

Запуск из корня проекта:
    .\\.venv\\Scripts\\python.exe scripts\\build_docs.py            # report и presentation
    .\\.venv\\Scripts\\python.exe scripts\\build_docs.py report     # только отчёт
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC, OUT = ROOT / "docs" / "src", ROOT / "docs"
EDGE_PATHS = [Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
              Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe")]
DOCS = {"report": "report.html", "presentation": "presentation.html"}


def build(name: str) -> Path:
    edge = next((p for p in EDGE_PATHS if p.exists()), None)
    if edge is None:
        sys.exit("Не найден Microsoft Edge")
    html, pdf = SRC / DOCS[name], OUT / f"{name}.pdf"
    with tempfile.TemporaryDirectory() as profile:
        subprocess.run([str(edge), "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                        f"--user-data-dir={profile}", "--virtual-time-budget=5000",
                        f"--print-to-pdf={pdf}", html.as_uri()], check=True, timeout=180)
    pages = None
    try:
        from pypdf import PdfReader
        pages = len(PdfReader(pdf).pages)
    except ImportError:
        pass
    print(f"{pdf.relative_to(ROOT)}: {pdf.stat().st_size // 1024} КБ" + (f", страниц: {pages}" if pages else ""))
    return pdf


if __name__ == "__main__":
    for n in (sys.argv[1:] or list(DOCS)):
        build(n)
