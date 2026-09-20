"""Тесты конфигурации приложения."""

from __future__ import annotations

from nlp_fin.config import BASE_DIR, Settings, ensure_data_dirs, get_settings


def test_base_dir_is_project_root() -> None:
    assert (BASE_DIR / "pyproject.toml").is_file()


def test_settings_defaults() -> None:
    settings = Settings(_env_file=None)
    assert settings.app_name == "nlp-fin"
    assert settings.log_level == "INFO"
    assert settings.data_dir == BASE_DIR / "data"
    assert settings.raw_dir == BASE_DIR / "data" / "raw"
    assert settings.ocr.tesseract_cmd is None
    assert settings.chunk_max_words == 128
    assert settings.chunk_overlap_ratio == 0.05
    assert settings.noise_repeat_ratio == 0.7
    assert settings.noise_header_zone == 0.05
    assert settings.noise_footer_zone == 0.95


def test_get_settings_returns_singleton() -> None:
    assert get_settings() is get_settings()


def test_ensure_data_dirs_creates_folders(settings: Settings) -> None:
    ensure_data_dirs(settings)
    for path in settings.all_data_dirs:
        assert path.is_dir()