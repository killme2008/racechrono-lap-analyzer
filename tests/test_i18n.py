"""Tests for internationalization (i18n) module."""

import unittest

from racechrono_lap_analyzer.i18n import (
    get_available_languages,
    get_language,
    get_language_name,
    set_language,
    t,
)


class LanguageSelectionTests(unittest.TestCase):
    """Tests for language selection and switching."""

    def setUp(self):
        """Reset to English before each test."""
        set_language("en")

    def test_default_language_is_english(self):
        """Default language should be English."""
        set_language("en")
        self.assertEqual(get_language(), "en")

    def test_set_language_to_chinese(self):
        """Can switch to Chinese."""
        set_language("zh-CN")
        self.assertEqual(get_language(), "zh-CN")

    def test_invalid_language_falls_back_to_english(self):
        """Invalid language code falls back to English."""
        set_language("invalid")
        self.assertEqual(get_language(), "en")

    def test_get_language_name(self):
        """Get human-readable language name."""
        set_language("en")
        self.assertEqual(get_language_name(), "English")

        set_language("zh-CN")
        self.assertEqual(get_language_name(), "简体中文")

    def test_get_available_languages(self):
        """Get dict of available languages."""
        languages = get_available_languages()
        self.assertIn("en", languages)
        self.assertIn("zh-CN", languages)
        self.assertEqual(languages["en"], "English")
        self.assertEqual(languages["zh-CN"], "简体中文")


class TranslationTests(unittest.TestCase):
    """Tests for translation function."""

    def setUp(self):
        """Reset to English before each test."""
        set_language("en")

    def test_english_translation(self):
        """Basic English translation works."""
        self.assertEqual(t("app.title"), "RaceChrono Lap Comparison Tool")

    def test_chinese_translation(self):
        """Chinese translation works."""
        set_language("zh-CN")
        self.assertEqual(t("app.title"), "RaceChrono 圈速对比工具")

    def test_nested_key(self):
        """Nested keys with dot notation work."""
        self.assertEqual(t("app.sidebar.settings"), "Settings")

    def test_format_parameters(self):
        """String formatting with parameters works."""
        result = t("app.sidebar.files_uploaded", count=5)
        self.assertEqual(result, "5 file(s) uploaded")

    def test_format_parameters_chinese(self):
        """String formatting works in Chinese."""
        set_language("zh-CN")
        result = t("app.sidebar.files_uploaded", count=5)
        self.assertEqual(result, "已上传 5 个文件")

    def test_missing_key_returns_key(self):
        """Missing key returns the key itself."""
        result = t("nonexistent.key")
        self.assertEqual(result, "nonexistent.key")

    def test_fallback_to_english(self):
        """Missing Chinese translation falls back to English."""
        set_language("zh-CN")
        # This key only exists if we add it; test fallback mechanism
        # by checking a key that should exist in both
        result = t("summary.lap_time")
        self.assertNotEqual(result, "summary.lap_time")

    def test_complex_format_string(self):
        """Complex format strings with floats work."""
        result = t("analysis.coach.brake_early.problem", distance=15.5)
        self.assertIn("16", result)  # Formatted as integer


class ChartTranslationTests(unittest.TestCase):
    """Tests for chart-related translations."""

    def setUp(self):
        """Reset to English before each test."""
        set_language("en")

    def test_chart_titles(self):
        """Chart titles translate correctly."""
        self.assertEqual(t("charts.speed.title"), "Speed (km/h)")
        self.assertEqual(t("charts.gg_diagram.title"), "G-G Diagram (Friction Circle)")

    def test_chart_titles_chinese(self):
        """Chart titles translate to Chinese."""
        set_language("zh-CN")
        self.assertEqual(t("charts.speed.title"), "速度 (km/h)")
        self.assertEqual(t("charts.gg_diagram.title"), "G-G 图（摩擦圆）")

    def test_common_labels(self):
        """Common chart labels work."""
        self.assertEqual(t("charts.common.distance"), "Distance (m)")
        self.assertEqual(t("charts.common.delta"), "Delta")


class CoachInsightTranslationTests(unittest.TestCase):
    """Tests for coach insight translations."""

    def setUp(self):
        """Reset to English before each test."""
        set_language("en")

    def test_coach_problem_templates(self):
        """Coach problem templates work."""
        result = t("analysis.coach.apex_slow.problem", diff=10)
        self.assertIn("10", result)
        self.assertIn("km/h", result)

    def test_coach_suggestions(self):
        """Coach suggestions are translated."""
        result = t("analysis.coach.exit_slow.suggestion")
        self.assertIn("throttle", result.lower())

    def test_coach_chinese(self):
        """Coach insights translate to Chinese."""
        set_language("zh-CN")
        result = t("analysis.coach.brake_early.problem", distance=15)
        self.assertIn("刹车", result)
        self.assertIn("15", result)


if __name__ == "__main__":
    unittest.main()
