# nlp-fin

Система автоматизированного анализа и извлечения данных из финансовой отчётности
эмитентов: OCR-парсинг, предобработка, дообученная трансформерная модель FinRuBERT,
микросервис (FastAPI + Celery) и ONNX-квантование инференса.

## Структура проекта

```
src/nlp_fin/
├── config.py          # настройки приложения
├── constants.py       # константы предметной области
├── ingestion/         # парсинг PDF/скан-OCR/XLSX/DOCX/HTML
├── preprocessing/     # нормализация, очистка, лемматизация, чанкинг
├── model/             # FinRuBERT (NER, риски), инференс, квантование
├── training/          # обучение и оценка моделей
├── api/               # FastAPI-приложение
├── workers/           # Celery-очереди и задачи
├── services/          # сервисный слой
└── utils/             # logging и вспомогательные утилиты
```

## Быстрый старт

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
```

## Проверка качества кода

```bash
ruff check .
mypy src
pytest
```