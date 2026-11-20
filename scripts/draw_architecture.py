"""Рисует архитектурную диаграмму проекта в docs/architecture.png."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams["font.family"] = "DejaVu Sans"
fig, ax = plt.subplots(figsize=(18, 9.6))
ax.set_xlim(0, 18); ax.set_ylim(0.3, 9.9); ax.axis("off")

INK = "#333333"
C = {"src": ("#EAF2FB", "#7A9CC6"), "proc": ("#FDF3E1", "#D9A441"), "pg": ("#E8F5E9", "#6BAF73"),
     "ml": ("#F3E8FA", "#A77BC4"), "srv": ("#FDE9E7", "#D07A70"), "mon": ("#FFF9D6", "#C9B037"),
     "ext": ("#F2F2F2", "#999999")}

def group(x0, y0, x1, y1, title, key):
    fill, edge = C[key]
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=0.15",
                                fc=fill, ec=edge, lw=1.5, zorder=1))
    ax.text((x0 + x1) / 2, y1 - 0.28, title, ha="center", va="center", fontsize=11, weight="bold", color=INK, zorder=2)

def box(cx, cy, w, title, sub="", h=0.9):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.08",
                                fc="white", ec=INK, lw=1.1, zorder=3))
    if sub:
        ax.text(cx, cy + 0.19, title, ha="center", va="center", fontsize=11, weight="bold", color=INK, zorder=4)
        ax.text(cx, cy - 0.17, sub, ha="center", va="center", fontsize=9, color=INK, zorder=4, linespacing=1.15)
    else:
        ax.text(cx, cy, title, ha="center", va="center", fontsize=11, weight="bold", color=INK, zorder=4)

def arrow(pts, label="", lpos=None, dashed=False, both=False, la="center", rot=0):
    xs, ys = zip(*pts)
    ls = (0, (5, 4)) if dashed else "-"
    if len(pts) > 2:
        ax.plot(xs[:-1], ys[:-1], color=INK, lw=1.2, ls=ls, zorder=5, solid_capstyle="round")
    ax.annotate("", xy=pts[-1], xytext=pts[-2], zorder=5,
                arrowprops=dict(arrowstyle="<|-|>" if both else "-|>", color=INK, lw=1.2, ls=ls, shrinkA=0, shrinkB=0))
    if label:
        ax.text(*lpos, label, ha=la, va="center", rotation=rot, fontsize=9, color=INK, style="italic", zorder=6,
                bbox=dict(fc="white", ec="none", pad=1.2, alpha=0.9))

ax.text(9, 9.6, "Архитектура системы предиктивного обслуживания (NASA C-MAPSS, FD001)",
        ha="center", va="center", fontsize=15, weight="bold", color=INK)

# --- группы
group(0.3, 3.3, 3.5, 9.1, "1. Источники данных", "src")
group(4.1, 3.3, 7.3, 9.1, "2. Обработка данных", "proc")
group(7.9, 0.7, 11.1, 9.1, "3. Хранилище PostgreSQL", "pg")
group(11.7, 4.7, 14.9, 9.1, "4. ML-компоненты", "ml")
group(11.7, 0.7, 14.9, 4.3, "5. Сервис предсказаний", "srv")
group(15.5, 4.7, 17.7, 7.2, "6. Мониторинг", "mon")

# --- узлы
W = 2.7
box(1.9, 7.9, W, "Симулятор датчиков", "Kafka Producer: из БД,\nасинхронно, ключ = двигатель", h=1.0)
box(1.9, 5.6, W, "Файлы C-MAPSS", "train / test / RUL (.txt)")
box(1.9, 4.2, W, "Web scraping", "характеристики двигателей\n(Википедия) -> enriched", h=1.0)

box(5.7, 8.2, W, "Apache Kafka", "топик sensor-readings", h=0.8)
box(5.7, 6.9, W, "Kafka Consumer", "группа потребителей, commit\nпосле записи в БД; прогноз", h=1.0)
box(5.7, 4.9, W, "ETL-пайплайн", "Python, pandas, SQL:\nзагрузка, очистка, признаки", h=1.0)

box(9.5, 7.9, W, "raw", "исходные данные")
box(9.5, 6.6, W, "cleaned", "очищенные данные")
box(9.5, 5.3, W, "enriched", "обогащённые данные")
box(9.5, 4.0, W, "features", "offline feature store:\n71 признак, SQL-функции", h=1.0)
box(9.5, 1.5, W, "predictions", "журнал предсказаний")

box(13.3, 7.9, W, "Обучение", "LR, RF, бустинг, MLP;\nCV по двигателям", h=1.0)
box(13.3, 6.6, W, "Оценка модели", "Recall, FPR, PR-AUC,\nпорог, анализ ошибок", h=1.0)
box(13.3, 5.3, W, "Хранилище моделей", "joblib + метаданные версии")
box(12.75, 3.1, 1.6, "Redis", "online feature\nstore", h=1.0)
box(13.3, 1.5, W, "REST API (FastAPI)", "/predict, /features,\nPydantic, p95 = 21 мс", h=1.0)

box(16.6, 5.7, 1.9, "", "", h=1.7)
ax.text(16.6, 5.7, "задержка и\nошибки API,\nдрейф данных,\nкачество\nмодели", ha="center", va="center", fontsize=10, color=INK, zorder=4)
fill, edge = C["ext"]
ax.add_patch(FancyBboxPatch((15.5, 0.95), 2.2, 1.1, boxstyle="round,pad=0,rounding_size=0.08", fc=fill, ec=edge, lw=1.3, zorder=3))
ax.text(16.6, 1.5, "Потребитель:\nслужба ТО,\nвнешние системы", ha="center", va="center", fontsize=10, color=INK, zorder=4)

# --- связи
arrow([(3.25, 7.9), (3.8, 7.9), (3.8, 8.2), (4.35, 8.2)])
arrow([(5.7, 7.8), (5.7, 7.4)])
arrow([(7.05, 6.9), (7.6, 6.9), (7.6, 7.9), (8.15, 7.9)], "поток", (7.6, 7.45), rot=90)
arrow([(3.25, 5.6), (3.8, 5.6), (3.8, 5.05), (4.35, 5.05)])
arrow([(3.25, 4.2), (3.8, 4.2), (3.8, 4.75), (4.35, 4.75)])
arrow([(7.05, 4.9), (7.9, 4.9)], "загрузка,\nслои", (7.48, 5.35))
arrow([(9.5, 7.45), (9.5, 7.05)]); arrow([(9.5, 6.15), (9.5, 5.75)]); arrow([(9.5, 4.85), (9.5, 4.5)])
arrow([(10.85, 4.0), (11.4, 4.0), (11.4, 7.9), (11.95, 7.9)], "обучающая выборка", (11.4, 6.0), rot=90)
arrow([(9.5, 3.5), (9.5, 3.0), (11.95, 3.0)], "материализация", (10.35, 3.0))
arrow([(13.3, 7.4), (13.3, 7.1)]); arrow([(13.3, 6.1), (13.3, 5.75)])
arrow([(14.2, 4.85), (14.2, 2.0)], "загрузка\nмодели", (14.2, 3.3))
arrow([(12.75, 2.6), (12.75, 2.0)], "признаки", (12.75, 2.3))
arrow([(11.95, 1.5), (10.85, 1.5)], "лог", (11.4, 1.5))
arrow([(14.65, 1.5), (15.5, 1.5)], "запрос /\nответ", (15.07, 1.1), both=True)
arrow([(10.85, 1.25), (11.4, 1.25), (11.4, 0.5), (17.9, 0.5), (17.9, 5.7), (17.55, 5.7)], "фактические исходы и предсказания", (14.6, 0.5))
arrow([(14.65, 1.75), (15.2, 1.75), (15.2, 5.4), (15.65, 5.4)], "метрики API", (15.2, 3.4), rot=90)
arrow([(16.6, 6.55), (16.6, 7.9), (14.65, 7.9)], "сигнал на переобучение", (16.3, 8.2), dashed=True)

fig.savefig("docs/architecture.png", dpi=170, bbox_inches="tight", facecolor="white")
