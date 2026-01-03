"""Tests for rules module with adaptive thresholds."""

import unittest

from racechrono_lap_analyzer.rules import (
    DEFAULT_THRESHOLDS,
    CornerType,
    RuleConfig,
    estimate_time_benefit_physics,
    get_corner_config_by_index,
    get_corner_type,
    get_thresholds,
)


class CornerTypeTests(unittest.TestCase):
    """Tests for corner type classification."""

    def test_corner_type_from_explicit_config(self):
        """Corner type from explicit 'type' field."""
        config = {"type": "hairpin"}
        self.assertEqual(get_corner_type(config), CornerType.HAIRPIN)

        config = {"type": "fast"}
        self.assertEqual(get_corner_type(config), CornerType.FAST)

    def test_corner_type_from_typical_speed(self):
        """Corner type from typical_speed_kmh field."""
        config = {"typical_speed_kmh": 45}
        self.assertEqual(get_corner_type(config), CornerType.HAIRPIN)

        config = {"typical_speed_kmh": 70}
        self.assertEqual(get_corner_type(config), CornerType.SLOW)

        config = {"typical_speed_kmh": 90}
        self.assertEqual(get_corner_type(config), CornerType.MEDIUM)

        config = {"typical_speed_kmh": 120}
        self.assertEqual(get_corner_type(config), CornerType.FAST)

    def test_corner_type_from_min_speed(self):
        """Corner type from provided min_speed_kmh."""
        self.assertEqual(get_corner_type(None, 40), CornerType.HAIRPIN)
        self.assertEqual(get_corner_type(None, 75), CornerType.SLOW)
        self.assertEqual(get_corner_type(None, 95), CornerType.MEDIUM)
        self.assertEqual(get_corner_type(None, 110), CornerType.FAST)

    def test_corner_type_default_medium(self):
        """Default corner type is MEDIUM."""
        self.assertEqual(get_corner_type(None, None), CornerType.MEDIUM)

    def test_explicit_type_overrides_speed(self):
        """Explicit type takes priority over speed."""
        config = {"type": "hairpin", "typical_speed_kmh": 120}
        self.assertEqual(get_corner_type(config), CornerType.HAIRPIN)


class ThresholdsTests(unittest.TestCase):
    """Tests for threshold retrieval and application."""

    def test_default_thresholds_exist(self):
        """All corner types have default thresholds."""
        for corner_type in CornerType:
            self.assertIn(corner_type, DEFAULT_THRESHOLDS)

    def test_hairpin_has_lower_thresholds(self):
        """Hairpin corners have lower thresholds."""
        hairpin = DEFAULT_THRESHOLDS[CornerType.HAIRPIN]
        fast = DEFAULT_THRESHOLDS[CornerType.FAST]

        self.assertLess(hairpin.brake_early_m, fast.brake_early_m)
        self.assertLess(hairpin.apex_speed_low_kmh, fast.apex_speed_low_kmh)

    def test_get_thresholds_basic(self):
        """Basic threshold retrieval works."""
        thresholds = get_thresholds(CornerType.MEDIUM)
        self.assertEqual(thresholds.brake_early_m, 12.0)
        self.assertEqual(thresholds.apex_speed_low_kmh, 5.0)

    def test_sensitivity_multiplier(self):
        """Sensitivity multiplier scales thresholds."""
        rule_config = RuleConfig(sensitivity=2.0)
        thresholds = get_thresholds(CornerType.MEDIUM, rule_config)

        default = DEFAULT_THRESHOLDS[CornerType.MEDIUM]
        self.assertEqual(thresholds.brake_early_m, default.brake_early_m * 2.0)

    def test_user_override(self):
        """User overrides take effect."""
        rule_config = RuleConfig(brake_early_m=20.0)
        thresholds = get_thresholds(CornerType.MEDIUM, rule_config)

        self.assertEqual(thresholds.brake_early_m, 20.0)

    def test_corner_config_override(self):
        """Corner-specific config overrides user config."""
        rule_config = RuleConfig(brake_early_m=20.0)
        corner_config = {"thresholds_override": {"brake_early_m": 5.0}}

        thresholds = get_thresholds(CornerType.MEDIUM, rule_config, corner_config)

        self.assertEqual(thresholds.brake_early_m, 5.0)


class TimeBenefitEstimationTests(unittest.TestCase):
    """Tests for physics-based time benefit estimation."""

    def test_basic_time_benefit(self):
        """Basic time benefit calculation works."""
        # 5 km/h slower over 100m at 80 km/h base
        benefit = estimate_time_benefit_physics(
            speed_diff_kmh=-5.0,
            section_length_m=100.0,
            base_speed_kmh=80.0,
            corner_type=CornerType.MEDIUM,
        )
        self.assertGreater(benefit, 0)

    def test_faster_speed_gives_benefit(self):
        """Faster speed gives positive benefit."""
        benefit = estimate_time_benefit_physics(
            speed_diff_kmh=10.0,
            section_length_m=100.0,
            base_speed_kmh=80.0,
            corner_type=CornerType.MEDIUM,
        )
        self.assertGreater(benefit, 0)

    def test_zero_speed_diff_gives_zero_benefit(self):
        """Zero speed difference gives near-zero benefit."""
        benefit = estimate_time_benefit_physics(
            speed_diff_kmh=0.0,
            section_length_m=100.0,
            base_speed_kmh=80.0,
            corner_type=CornerType.MEDIUM,
        )
        self.assertAlmostEqual(benefit, 0.0, places=1)

    def test_hairpin_multiplier_higher(self):
        """Hairpin corners have higher downstream impact multiplier."""
        hairpin_benefit = estimate_time_benefit_physics(
            speed_diff_kmh=-5.0,
            section_length_m=100.0,
            base_speed_kmh=50.0,
            corner_type=CornerType.HAIRPIN,
        )
        fast_benefit = estimate_time_benefit_physics(
            speed_diff_kmh=-5.0,
            section_length_m=100.0,
            base_speed_kmh=50.0,
            corner_type=CornerType.FAST,
        )
        # Hairpin multiplier is 1.2, fast is 0.9
        self.assertGreater(hairpin_benefit, fast_benefit)

    def test_handles_zero_base_speed(self):
        """Handles zero base speed gracefully."""
        benefit = estimate_time_benefit_physics(
            speed_diff_kmh=-5.0,
            section_length_m=100.0,
            base_speed_kmh=0.0,
            corner_type=CornerType.MEDIUM,
        )
        self.assertEqual(benefit, 0.0)

    def test_handles_negative_compare_speed(self):
        """Handles negative compare speed gracefully."""
        benefit = estimate_time_benefit_physics(
            speed_diff_kmh=-100.0,  # Would make compare speed negative
            section_length_m=100.0,
            base_speed_kmh=50.0,
            corner_type=CornerType.MEDIUM,
        )
        # Should not raise, should return valid positive number
        self.assertGreater(benefit, 0)


class CornerConfigIndexTests(unittest.TestCase):
    """Tests for corner config lookup by index."""

    def test_get_corner_by_index(self):
        """Get corner config by 1-based index."""
        track_config = {
            "corners": [
                {"id": "T1", "apex_distance_m": 100},
                {"id": "T2", "apex_distance_m": 200},
                {"id": "T3", "apex_distance_m": 300},
            ]
        }

        corner = get_corner_config_by_index(1, track_config)
        self.assertEqual(corner["id"], "T1")

        corner = get_corner_config_by_index(3, track_config)
        self.assertEqual(corner["id"], "T3")

    def test_invalid_index_returns_none(self):
        """Invalid index returns None."""
        track_config = {"corners": [{"id": "T1"}]}

        self.assertIsNone(get_corner_config_by_index(0, track_config))
        self.assertIsNone(get_corner_config_by_index(5, track_config))

    def test_none_config_returns_none(self):
        """None track config returns None."""
        self.assertIsNone(get_corner_config_by_index(1, None))


class RuleConfigTests(unittest.TestCase):
    """Tests for RuleConfig dataclass."""

    def test_default_values(self):
        """Default RuleConfig values are correct."""
        config = RuleConfig()
        self.assertEqual(config.sensitivity, 1.0)
        self.assertIsNone(config.brake_early_m)
        self.assertEqual(config.min_time_benefit_ms, 10.0)
        self.assertEqual(config.max_insights, 5)

    def test_custom_values(self):
        """Custom RuleConfig values work."""
        config = RuleConfig(
            sensitivity=1.5,
            brake_early_m=15.0,
            max_insights=10,
        )
        self.assertEqual(config.sensitivity, 1.5)
        self.assertEqual(config.brake_early_m, 15.0)
        self.assertEqual(config.max_insights, 10)


if __name__ == "__main__":
    unittest.main()
