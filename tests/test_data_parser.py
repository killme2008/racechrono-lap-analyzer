"""Tests for data_parser module using real RaceChrono CSV data."""

import unittest
from pathlib import Path

from racechrono_lap_analyzer.data_parser import (
    LapData,
    SessionMetadata,
    format_lap_time,
    generate_friendly_name,
    load_lap,
    normalize_lap_data,
    parse_metadata,
    parse_racechrono_csv,
)

TEST_DATA_DIR = Path(__file__).parent / "data"
LAP9_CSV = TEST_DATA_DIR / "session_20251231_111318_tianma_lap9_v3.csv"
LAP13_CSV = TEST_DATA_DIR / "session_20251231_111959_tianma_lap13_v3.csv"


class ParseMetadataTests(unittest.TestCase):
    def test_parse_metadata_extracts_session_info(self) -> None:
        metadata = parse_metadata(str(LAP9_CSV))

        self.assertEqual(metadata.format_version, "3")
        self.assertEqual(metadata.session_title, "Tianma")
        self.assertEqual(metadata.session_type, "Lap timing")
        self.assertEqual(metadata.track_name, "Tianma")
        self.assertIn("31/12/2025", metadata.created_date)

    def test_parse_metadata_handles_different_files(self) -> None:
        metadata = parse_metadata(str(LAP13_CSV))

        self.assertEqual(metadata.track_name, "Tianma")
        self.assertEqual(metadata.format_version, "3")


class ParseRacechronoCsvTests(unittest.TestCase):
    def test_parse_csv_returns_dataframe_with_expected_columns(self) -> None:
        df, lap_number = parse_racechrono_csv(str(LAP9_CSV))

        # Core columns should exist
        self.assertIn("timestamp", df.columns)
        self.assertIn("lap_number", df.columns)
        self.assertIn("speed", df.columns)
        self.assertIn("latitude", df.columns)
        self.assertIn("longitude", df.columns)
        self.assertIn("lateral_acc", df.columns)
        self.assertIn("longitudinal_acc", df.columns)

    def test_parse_csv_returns_correct_lap_number(self) -> None:
        df, lap_number = parse_racechrono_csv(str(LAP9_CSV))

        # Should auto-detect the main lap (lap 9 based on filename)
        self.assertEqual(lap_number, 9)

    def test_parse_csv_filters_to_single_lap(self) -> None:
        df, lap_number = parse_racechrono_csv(str(LAP9_CSV))

        # All rows should be from the same lap
        unique_laps = df["lap_number"].unique()
        self.assertEqual(len(unique_laps), 1)
        self.assertEqual(unique_laps[0], lap_number)

    def test_parse_csv_with_target_lap(self) -> None:
        df, lap_number = parse_racechrono_csv(str(LAP9_CSV), target_lap=9)

        self.assertEqual(lap_number, 9)

    def test_parse_csv_has_obd_data(self) -> None:
        df, _ = parse_racechrono_csv(str(LAP9_CSV))

        # OBD columns should exist
        self.assertIn("throttle_pos", df.columns)
        self.assertIn("rpm", df.columns)
        self.assertIn("accelerator_pos", df.columns)


class NormalizeLapDataTests(unittest.TestCase):
    def test_normalize_resets_distance_to_zero(self) -> None:
        df, _ = parse_racechrono_csv(str(LAP9_CSV))
        normalized = normalize_lap_data(df)

        self.assertAlmostEqual(normalized["distance"].min(), 0.0, places=1)

    def test_normalize_resets_time_to_zero(self) -> None:
        df, _ = parse_racechrono_csv(str(LAP9_CSV))
        normalized = normalize_lap_data(df)

        self.assertAlmostEqual(normalized["time"].min(), 0.0, places=1)

    def test_normalize_converts_speed_to_kmh(self) -> None:
        df, _ = parse_racechrono_csv(str(LAP9_CSV))
        normalized = normalize_lap_data(df)

        self.assertIn("speed_kmh", normalized.columns)

        # Speed in km/h should be ~3.6x speed in m/s
        if "speed" in normalized.columns:
            ratio = normalized["speed_kmh"].mean() / normalized["speed"].mean()
            self.assertAlmostEqual(ratio, 3.6, places=1)


class LoadLapTests(unittest.TestCase):
    def test_load_lap_returns_lap_data(self) -> None:
        lap = load_lap(str(LAP9_CSV))

        self.assertIsInstance(lap, LapData)
        self.assertEqual(lap.lap_number, 9)

    def test_load_lap_computes_statistics(self) -> None:
        lap = load_lap(str(LAP9_CSV))

        # Lap time should be positive
        self.assertGreater(lap.lap_time, 0)

        # Max speed should be reasonable for a race track (50-200 km/h)
        self.assertGreater(lap.max_speed_kmh, 50)
        self.assertLess(lap.max_speed_kmh, 200)

        # Avg speed should be less than max
        self.assertLess(lap.avg_speed_kmh, lap.max_speed_kmh)
        self.assertGreater(lap.avg_speed_kmh, 0)

        # G-forces should be reasonable (0-2G)
        self.assertGreater(lap.max_lateral_g, 0)
        self.assertLess(lap.max_lateral_g, 2.5)
        self.assertGreater(lap.max_brake_g, 0)
        self.assertLess(lap.max_brake_g, 2.5)

    def test_load_lap_detects_obd_data(self) -> None:
        lap = load_lap(str(LAP9_CSV))

        self.assertTrue(lap.has_obd_data)
        self.assertTrue(lap.has_throttle)
        self.assertTrue(lap.has_rpm)

    def test_load_lap_has_valid_distance(self) -> None:
        lap = load_lap(str(LAP9_CSV))

        # Tianma circuit is ~2000m
        self.assertGreater(lap.lap_distance, 1800)
        self.assertLess(lap.lap_distance, 2200)

    def test_load_lap_dataframe_has_distance_column(self) -> None:
        lap = load_lap(str(LAP9_CSV))

        self.assertIn("distance", lap.df.columns)
        self.assertAlmostEqual(lap.df["distance"].min(), 0.0, places=1)


class GenerateFriendlyNameTests(unittest.TestCase):
    def test_generate_friendly_name_extracts_date_and_lap(self) -> None:
        metadata = SessionMetadata(filename="test")
        name = generate_friendly_name(
            "session_20251231_111318_tianma_lap9_v3", 9, metadata
        )

        self.assertIn("12/31", name)
        self.assertIn("L9", name)

    def test_generate_friendly_name_handles_different_laps(self) -> None:
        metadata = SessionMetadata(filename="test")
        name = generate_friendly_name(
            "session_20251222_102302_tianma_lap14_v3", 14, metadata
        )

        self.assertIn("12/22", name)
        self.assertIn("L14", name)

    def test_generate_friendly_name_fallback(self) -> None:
        metadata = SessionMetadata(filename="test")
        name = generate_friendly_name("unknown_format", 0, metadata)

        # Should return truncated filename when pattern doesn't match
        self.assertTrue(len(name) > 0)


class FormatLapTimeTests(unittest.TestCase):
    def test_format_lap_time_under_minute(self) -> None:
        result = format_lap_time(45.123)
        self.assertEqual(result, "0:45.123")

    def test_format_lap_time_over_minute(self) -> None:
        result = format_lap_time(90.456)
        self.assertEqual(result, "1:30.456")

    def test_format_lap_time_exact_minute(self) -> None:
        result = format_lap_time(60.0)
        self.assertEqual(result, "1:00.000")


class LoadMultipleLapsTests(unittest.TestCase):
    def test_load_two_laps_different_times(self) -> None:
        lap9 = load_lap(str(LAP9_CSV))
        lap13 = load_lap(str(LAP13_CSV))

        # Both should load successfully
        self.assertEqual(lap9.lap_number, 9)
        self.assertEqual(lap13.lap_number, 13)

        # Lap times should be different
        self.assertNotEqual(lap9.lap_time, lap13.lap_time)

    def test_laps_have_similar_distance(self) -> None:
        lap9 = load_lap(str(LAP9_CSV))
        lap13 = load_lap(str(LAP13_CSV))

        # Same track, should have similar distance (within 5%)
        diff_pct = abs(lap9.lap_distance - lap13.lap_distance) / lap9.lap_distance
        self.assertLess(diff_pct, 0.05)


if __name__ == "__main__":
    unittest.main()
