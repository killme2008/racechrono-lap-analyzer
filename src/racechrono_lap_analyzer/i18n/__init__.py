"""
Internationalization (i18n) module for RaceChrono Lap Analyzer.

Provides a simple JSON-based translation system.
"""

import json
from pathlib import Path
from typing import Any

# Module state
_current_language: str = "en"
_translations: dict[str, dict] = {}
_i18n_dir = Path(__file__).parent

# Supported languages
SUPPORTED_LANGUAGES = {
    "en": "English",
    "zh-CN": "简体中文",
}


def _load_translations(lang: str) -> dict:
    """Load translation file for a language."""
    file_path = _i18n_dir / f"{lang}.json"
    if file_path.exists():
        with open(file_path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _get_nested(data: dict, key: str) -> Any:
    """Get nested value using dot notation (e.g., 'app.title')."""
    parts = key.split(".")
    current = data
    for part in parts:
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return None
    return current


def set_language(lang: str) -> None:
    """
    Set the current language.

    Args:
        lang: Language code (e.g., 'en', 'zh-CN')
    """
    global _current_language, _translations

    if lang not in SUPPORTED_LANGUAGES:
        lang = "en"

    _current_language = lang

    # Load translations if not already loaded
    if lang not in _translations:
        _translations[lang] = _load_translations(lang)

    # Also ensure English is loaded for fallback
    if "en" not in _translations:
        _translations["en"] = _load_translations("en")


def get_language() -> str:
    """Get the current language code."""
    return _current_language


def get_language_name() -> str:
    """Get the display name of the current language."""
    return SUPPORTED_LANGUAGES.get(_current_language, "English")


def t(key: str, **kwargs: Any) -> str:
    """
    Get translated string by key.

    Supports dot notation for nested keys (e.g., 'app.sidebar.title').
    Supports format parameters (e.g., t('msg', name='John') for '{name}').

    Fallback order:
    1. Current language
    2. English
    3. Key itself (for debugging)

    Args:
        key: Translation key in dot notation
        **kwargs: Format parameters

    Returns:
        Translated string
    """
    # Ensure translations are loaded
    if _current_language not in _translations:
        set_language(_current_language)

    # Try current language
    value = _get_nested(_translations.get(_current_language, {}), key)

    # Fallback to English
    if value is None and _current_language != "en":
        value = _get_nested(_translations.get("en", {}), key)

    # Fallback to key itself
    if value is None:
        return key

    # Format with kwargs if provided
    if kwargs and isinstance(value, str):
        try:
            return value.format(**kwargs)
        except (KeyError, ValueError):
            return value

    return str(value)


def get_available_languages() -> dict[str, str]:
    """Get dict of available languages {code: name}."""
    return SUPPORTED_LANGUAGES.copy()


# Initialize with default language on import
set_language("en")
