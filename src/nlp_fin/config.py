"""Конфигурация приложения.

Настройки читаются из переменных окружения и файла .env.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parents[2]


class OcrSettings(BaseModel):
    """Настройки OCR-распознавания Tesseract."""

    tesseract_cmd: str | None = None
    tesseract_lang: str = "rus"
    tessdata_prefix: str | None = None
    psm: int = 4
    oem: int = 1
    image_dpi: int = 300


class PdfSettings(BaseModel):
    """Настройки определения наличия текстового слоя в PDF."""

    text_min_chars: int = 50
    sample_pages: int = 3


class Settings(BaseSettings):
    """Настройки приложения."""

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "nlp-fin"
    debug: bool = False
    log_level: str = "INFO"

    data_dir: Path = BASE_DIR / "data"
    raw_dir: Path = BASE_DIR / "data" / "raw"
    processed_dir: Path = BASE_DIR / "data" / "processed"
    chunks_dir: Path = BASE_DIR / "data" / "chunks"
    annotations_dir: Path = BASE_DIR / "data" / "annotations"
    models_dir: Path = BASE_DIR / "data" / "models"
    dictionaries_dir: Path = BASE_DIR / "data" / "dictionaries"

    ocr_tesseract_cmd: str | None = None
    ocr_tesseract_lang: str = "rus"
    ocr_tessdata_prefix: str | None = None
    ocr_psm: int = 4
    ocr_oem: int = 1
    ocr_image_dpi: int = 300

    pdf_text_min_chars: int = 50
    pdf_sample_pages: int = 3

    @property
    def ocr(self) -> OcrSettings:
        """Настройки OCR-распознавания."""
        return OcrSettings(
            tesseract_cmd=self.ocr_tesseract_cmd,
            tesseract_lang=self.ocr_tesseract_lang,
            tessdata_prefix=self.ocr_tessdata_prefix,
            psm=self.ocr_psm,
            oem=self.ocr_oem,
            image_dpi=self.ocr_image_dpi,
        )

    @property
    def pdf(self) -> PdfSettings:
        """Настройки определения текстового слоя в PDF."""
        return PdfSettings(
            text_min_chars=self.pdf_text_min_chars, sample_pages=self.pdf_sample_pages
        )

    @property
    def all_data_dirs(self) -> tuple[Path, ...]:
        """Все директории данных проекта."""
        return (
            self.data_dir,
            self.raw_dir,
            self.processed_dir,
            self.chunks_dir,
            self.annotations_dir,
            self.models_dir,
            self.dictionaries_dir,
        )


@lru_cache
def get_settings() -> Settings:
    """Возвращает кэшированный экземпляр настроек приложения."""
    return Settings()


def ensure_data_dirs(settings: Settings) -> None:
    """Создаёт директории данных при их отсутствии."""
    for path in settings.all_data_dirs:
        path.mkdir(parents=True, exist_ok=True)