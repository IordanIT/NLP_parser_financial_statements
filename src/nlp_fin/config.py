"""Конфигурация приложения.

Настройки читаются из переменных окружения и файла .env.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


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