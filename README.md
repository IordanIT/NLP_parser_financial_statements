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

## Системные зависимости

Для распознавания отсканированных PDF требуется Tesseract OCR с русским языком.
На Windows: `winget install --id UB-Mannheim.TesseractOCR -e` (добавить язык Russian).
Если исполняемый файл не в PATH, укажите его путь в `.env`: `OCR_TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe`.

Для лемматизации нужна русская модель spaCy:

```bash
python -m spacy download ru_core_news_md
```

Без неё тесты лемматизации пропускаются (`pytest.mark.skipif`).

## Проверка качества кода

```bash
ruff check .
mypy src
pytest
```