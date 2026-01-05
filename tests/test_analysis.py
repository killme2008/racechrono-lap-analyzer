"""Tests for analysis module."""

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from racechrono_lap_analyzer.analysis import (
    align_laps,
    analyze_gg_diagram,
    compare_corners,
    compute_time_delta,
    compute_tire_utilization,
    corners_from_track_config,
    detect_corners,
    detect_track,
    find_bottlenecks,
    generate_coach_insights,
    get_available_tracks,
    get_corner_difficulty_risk,
    load_track_config,
    resample_by_distance,
)
from racechrono_lap_analyzer.data_parser import LapData, SessionMetadata, load_lap

TEST_DATA_DIR = Path(__file__).parent / "data"
LAP9_CSV = TEST_DATA_DIR / "session_20251231_111318_tianma_lap9_v3.csv"
LAP13_CSV = TEST_DATA_DIR / "session_20251231_111959_tianma_lap13_v3.csv"


def _make_lap(name: str, df: pd.DataFrame, lap_distance: float) -> LapData:
    return LapData(
        name=name,
        metadata=SessionMetadata(filename=name),
        df=df,
        lap_distance=lap_distance,
        lap_time=float(df["time"].max()) if "time" in df.columns else 0.0,
    )


class ResampleByDistanceTests(unittest.TestCase):
    def test_resample_handles_non_monotonic_distance(self) -> None:
        df = pd.DataFrame(
            {
                "distance": [0.0, 1.0, 1.0, 0.5, 2.0],
                "speed_kmh": [0.0, 10.0, 20.0, 15.0, 30.0],
            }
        )
        distances = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        result = resample_by_distance(df, distances, ["speed_kmh"])
        self.assertTrue(np.isfinite(result["speed_kmh"]).any())


class AlignLapsTests(unittest.TestCase):
    def test_align_uses_maximum_lap_distance(self) -> None:
        df_short = pd.DataFrame(
            {"distance": [0.0, 1.0, 2.0], "time": [0.0, 1.0, 2.0]}
        )
        df_long = pd.DataFrame(
            {
                "distance": [0.0, 1.0, 2.0, 3.0, 4.0],
                "time": [0.0, 1.0, 2.0, 3.0, 4.0],
            }
        )
        lap_short = _make_lap("short", df_short, lap_distance=2.0)
        lap_long = _make_lap("long", df_long, lap_distance=4.0)

        distances, aligned = align_laps([lap_short, lap_long], resolution_m=1.0)

        # Uses max(lap.lap_distance) to avoid truncating longer laps
        # Shorter laps will have NaN for distances beyond their range
        self.assertEqual(len(distances), 4)
        self.assertEqual(len(aligned), 2)

    def test_align_real_laps(self) -> None:
        lap9 = load_lap(str(LAP9_CSV))
        lap13 = load_lap(str(LAP13_CSV))

        distances, aligned = align_laps([lap9, lap13], resolution_m=1.0)

        # Should have data for both laps
        self.assertEqual(len(aligned), 2)
        self.assertGreater(len(distances), 1000)  # Tianma is ~2000m

        # Aligned data should have same length
        self.assertEqual(len(aligned[0]), len(aligned[1]))


class DetectCornersTests(unittest.TestCase):
    def test_detect_corners_finds_corners_in_real_data(self) -> None:
        lap = load_lap(str(LAP9_CSV))
        corners = detect_corners(lap.df)

        # Tianma has 14 corners, should detect several
        self.assertGreater(len(corners), 5)
        self.assertLess(len(corners), 20)

    def test_detect_corners_has_valid_properties(self) -> None:
        lap = load_lap(str(LAP9_CSV))
        corners = detect_corners(lap.df)

        for corner in corners:
            # Each corner should have valid properties
            self.assertGreater(corner.apex_distance, 0)
            self.assertIn(corner.direction, ["left", "right"])
            self.assertGreater(corner.max_lateral_g, 0)
            self.assertGreater(corner.min_speed_kmh, 0)
            self.assertLess(corner.min_speed_kmh, 200)

    def test_detect_corners_ordered_by_distance(self) -> None:
        lap = load_lap(str(LAP9_CSV))
        corners = detect_corners(lap.df)

        # Corners should be in order of apex distance
        for i in range(1, len(corners)):
            self.assertGreater(
                corners[i].apex_distance, corners[i - 1].apex_distance
            )


class TrackConfigTests(unittest.TestCase):
    def test_get_available_tracks(self) -> None:
        tracks = get_available_tracks()

        self.assertIn("tianma", tracks)

    def test_load_track_config(self) -> None:
        config = load_track_config("tianma")

        self.assertIsNotNone(config)
        self.assertEqual(config["name"], "Shanghai Tianma Circuit")
        self.assertEqual(len(config["corners"]), 14)

    def test_load_track_config_returns_none_for_unknown(self) -> None:
        config = load_track_config("nonexistent_track")

        self.assertIsNone(config)

    def test_corners_from_track_config(self) -> None:
        lap = load_lap(str(LAP9_CSV))
        config = load_track_config("tianma")
        corners = corners_from_track_config(lap.df, config)

        # Should match track config (14 corners)
        self.assertEqual(len(corners), 14)

        # First corner should be T1 (left hairpin around 180m)
        self.assertEqual(corners[0].direction, "left")
        self.assertAlmostEqual(corners[0].apex_distance, 180, delta=10)


class DetectTrackTests(unittest.TestCase):
    def test_detect_track_by_gps(self) -> None:
        # Tianma GPS coordinates
        result = detect_track(center_lat=31.0775, center_lon=121.1155)
        self.assertEqual(result, "tianma")

    def test_detect_track_by_gps_nearby(self) -> None:
        # Slightly off but within radius
        result = detect_track(center_lat=31.078, center_lon=121.116)
        self.assertEqual(result, "tianma")

    def test_detect_track_by_distance_fallback(self) -> None:
        # No GPS, but distance matches Tianma (~2000m)
        result = detect_track(lap_distance=2000)
        self.assertEqual(result, "tianma")

    def test_detect_track_with_tolerance(self) -> None:
        # Should match with 10% tolerance
        result = detect_track(lap_distance=2100)  # 5% off
        self.assertEqual(result, "tianma")

        result = detect_track(lap_distance=1900)  # 5% off
        self.assertEqual(result, "tianma")

    def test_detect_track_no_match(self) -> None:
        # Way off - should not match any track
        result = detect_track(lap_distance=5000)
        self.assertIsNone(result)

        # Wrong GPS location
        result = detect_track(center_lat=40.0, center_lon=116.0)
        self.assertIsNone(result)

    def test_detect_track_from_real_lap(self) -> None:
        lap = load_lap(str(LAP13_CSV))
        center_lat = lap.df['latitude'].mean()
        center_lon = lap.df['longitude'].mean()
        result = detect_track(lap.lap_distance, center_lat, center_lon)
        self.assertEqual(result, "tianma")


class FindBottlenecksTests(unittest.TestCase):
    def test_find_bottlenecks_between_real_laps(self) -> None:
        lap9 = load_lap(str(LAP9_CSV))
        lap13 = load_lap(str(LAP13_CSV))

        # Sort by lap time (fastest first)
        laps = sorted([lap9, lap13], key=lambda x: x.lap_time)
        bottlenecks = find_bottlenecks(laps, threshold_kmh=5.0)

        # Should find some bottlenecks between different laps
        self.assertIsInstance(bottlenecks, list)

    def test_find_bottlenecks_has_valid_categories(self) -> None:
        lap9 = load_lap(str(LAP9_CSV))
        lap13 = load_lap(str(LAP13_CSV))

        laps = sorted([lap9, lap13], key=lambda x: x.lap_time)
        bottlenecks = find_bottlenecks(laps, threshold_kmh=3.0)

        valid_categories = {"braking", "cornering", "acceleration", "top_speed"}
        for b in bottlenecks:
            self.assertIn(b.category, valid_categories)
            self.assertGreater(b.end_m, b.start_m)


class ComputeTimeDeltaTests(unittest.TestCase):
    def test_compute_time_delta_real_laps(self) -> None:
        lap9 = load_lap(str(LAP9_CSV))
        lap13 = load_lap(str(LAP13_CSV))

        distances, time_delta = compute_time_delta(lap9, lap13)

        # Should have data for the full lap
        self.assertGreater(len(distances), 1000)
        self.assertEqual(len(distances), len(time_delta))

        # Final time delta should roughly match lap time difference
        lap_time_diff = lap13.lap_time - lap9.lap_time
        final_delta = time_delta[-1] if not np.isnan(time_delta[-1]) else 0

        # Allow some tolerance due to alignment
        self.assertAlmostEqual(final_delta, lap_time_diff, delta=1.0)


class CompareCornersTests(unittest.TestCase):
    def test_compare_corners_real_laps(self) -> None:
        lap9 = load_lap(str(LAP9_CSV))
        lap13 = load_lap(str(LAP13_CSV))

        comparisons = compare_corners(lap9, lap13)

        # Should have comparisons for detected corners
        self.assertGreater(len(comparisons), 0)

    def test_compare_corners_with_track_config(self) -> None:
        lap9 = load_lap(str(LAP9_CSV))
        lap13 = load_lap(str(LAP13_CSV))

        config = load_track_config("tianma")
        comparisons = compare_corners(lap9, lap13, track_config=config)

        # Should have 14 comparisons (one per corner)
        self.assertEqual(len(comparisons), 14)

        for cc in comparisons:
            self.assertIn(cc.direction, ["left", "right", "unknown"])
            self.assertGreater(cc.apex_distance, 0)


class GenerateCoachInsightsTests(unittest.TestCase):
    def test_generate_coach_insights_real_laps(self) -> None:
        lap9 = load_lap(str(LAP9_CSV))
        lap13 = load_lap(str(LAP13_CSV))

        # Use faster lap as reference
        laps = sorted([lap9, lap13], key=lambda x: x.lap_time)
        ref_lap = laps[0]
        compare_lap = laps[1]

        insights = generate_coach_insights(ref_lap, compare_lap, max_insights=5)

        # Should return list of insights
        self.assertIsInstance(insights, list)
        self.assertLessEqual(len(insights), 5)

    def test_coach_insights_have_valid_structure(self) -> None:
        lap9 = load_lap(str(LAP9_CSV))
        lap13 = load_lap(str(LAP13_CSV))

        laps = sorted([lap9, lap13], key=lambda x: x.lap_time)
        insights = generate_coach_insights(laps[0], laps[1], max_insights=10)

        valid_categories = {"braking", "corner_speed", "exit", "straight", "line"}
        for insight in insights:
            self.assertIn(insight.category, valid_categories)
            self.assertGreater(insight.priority, 0)
            self.assertTrue(len(insight.location) > 0)
            self.assertTrue(len(insight.problem) > 0)
            self.assertTrue(len(insight.suggestion) > 0)

    def test_coach_insights_with_track_config(self) -> None:
        lap9 = load_lap(str(LAP9_CSV))
        lap13 = load_lap(str(LAP13_CSV))

        laps = sorted([lap9, lap13], key=lambda x: x.lap_time)
        config = load_track_config("tianma")

        insights = generate_coach_insights(
            laps[0], laps[1], max_insights=5, track_config=config
        )

        self.assertIsInstance(insights, list)


class TireUtilizationTests(unittest.TestCase):
    def test_compute_tire_utilization_real_lap(self) -> None:
        lap13 = load_lap(str(LAP13_CSV))
        stats = compute_tire_utilization(lap13)

        # Should have valid values
        self.assertGreater(stats.avg_combined_g, 0)
        self.assertGreater(stats.max_combined_g, stats.avg_combined_g)
        self.assertGreaterEqual(stats.high_g_percentage, 0)
        self.assertLessEqual(stats.high_g_percentage, 100)
        self.assertGreaterEqual(stats.trail_brake_percentage, 0)
        self.assertLessEqual(stats.trail_brake_percentage, 100)

    def test_tire_utilization_ratings_are_valid(self) -> None:
        lap13 = load_lap(str(LAP13_CSV))
        stats = compute_tire_utilization(lap13)

        valid_ratings = {"low", "medium", "good", "excellent"}
        self.assertIn(stats.high_g_rating, valid_ratings)
        self.assertIn(stats.trail_brake_rating, valid_ratings)

    def test_tire_utilization_throttle_with_obd(self) -> None:
        lap13 = load_lap(str(LAP13_CSV))
        stats = compute_tire_utilization(lap13)

        # Lap 13 has OBD data
        if lap13.has_obd_data:
            self.assertIsNotNone(stats.full_throttle_percentage)
            self.assertIsNotNone(stats.throttle_rating)


class GGDiagramInsightTests(unittest.TestCase):
    def test_analyze_gg_diagram_real_lap(self) -> None:
        lap13 = load_lap(str(LAP13_CSV))
        insights = analyze_gg_diagram(lap13)

        # Should have valid values
        self.assertGreaterEqual(insights.left_turn_avg_g, 0)
        self.assertGreaterEqual(insights.right_turn_avg_g, 0)

        # Quadrant percentages should sum to less than 100
        # (not all time is in a quadrant - straight driving, etc.)
        total_quadrant = (
            insights.brake_left_pct
            + insights.brake_right_pct
            + insights.accel_left_pct
            + insights.accel_right_pct
        )
        self.assertLessEqual(total_quadrant, 100)

    def test_gg_diagram_insights_are_strings(self) -> None:
        lap13 = load_lap(str(LAP13_CSV))
        insights = analyze_gg_diagram(lap13)

        self.assertIsInstance(insights.insights, list)
        for insight in insights.insights:
            self.assertIsInstance(insight, str)

    def test_gg_diagram_left_right_balance(self) -> None:
        lap13 = load_lap(str(LAP13_CSV))
        insights = analyze_gg_diagram(lap13)

        # Balance is the difference between right and left
        expected_balance = insights.right_turn_avg_g - insights.left_turn_avg_g
        self.assertAlmostEqual(insights.left_right_balance, expected_balance, places=5)


class CoachInsightDifficultyRiskTests(unittest.TestCase):
    def test_corner_difficulty_risk_hairpin(self) -> None:
        difficulty, risk = get_corner_difficulty_risk("hairpin")
        self.assertEqual(difficulty, "high")
        self.assertEqual(risk, "low")

    def test_corner_difficulty_risk_fast(self) -> None:
        difficulty, risk = get_corner_difficulty_risk("fast")
        self.assertEqual(difficulty, "medium")
        self.assertEqual(risk, "high")

    def test_corner_difficulty_risk_medium(self) -> None:
        difficulty, risk = get_corner_difficulty_risk("medium")
        self.assertEqual(difficulty, "medium")
        self.assertEqual(risk, "medium")

    def test_corner_difficulty_risk_from_speed(self) -> None:
        # Low speed = hairpin-like
        difficulty, risk = get_corner_difficulty_risk(None, 50.0)
        self.assertEqual(difficulty, "high")
        self.assertEqual(risk, "low")

        # High speed = fast corner
        difficulty, risk = get_corner_difficulty_risk(None, 100.0)
        self.assertEqual(difficulty, "medium")
        self.assertEqual(risk, "high")

    def test_coach_insights_have_difficulty_risk(self) -> None:
        lap9 = load_lap(str(LAP9_CSV))
        lap13 = load_lap(str(LAP13_CSV))

        laps = sorted([lap9, lap13], key=lambda x: x.lap_time)
        config = load_track_config("tianma")

        insights = generate_coach_insights(
            laps[0], laps[1], max_insights=5, track_config=config
        )

        valid_levels = {"low", "medium", "high"}
        for insight in insights:
            self.assertIn(insight.difficulty, valid_levels)
            self.assertIn(insight.risk, valid_levels)


class HaversineDistanceTests(unittest.TestCase):
    def test_haversine_same_point_is_zero(self) -> None:
        from racechrono_lap_analyzer.analysis import haversine_distance

        dist = haversine_distance(31.0, 121.0, 31.0, 121.0)
        self.assertEqual(dist, 0.0)

    def test_haversine_small_distance(self) -> None:
        from racechrono_lap_analyzer.analysis import haversine_distance

        # ~111m difference (0.001 degree lat)
        dist = haversine_distance(31.0, 121.0, 31.001, 121.0)
        self.assertAlmostEqual(dist, 111, delta=5)

    def test_haversine_known_distance(self) -> None:
        from racechrono_lap_analyzer.analysis import haversine_distance

        # Shanghai to Beijing ~1068km
        dist = haversine_distance(31.23, 121.47, 39.90, 116.40)
        self.assertAlmostEqual(dist, 1068000, delta=10000)


class LineDeviationTests(unittest.TestCase):
    def test_deviation_identical_laps_is_zero(self) -> None:
        from racechrono_lap_analyzer.analysis import compute_line_deviations

        lap = load_lap(str(LAP9_CSV))
        offsets = compute_line_deviations(lap, lap)

        # Identical lap should have near-zero offset
        valid_offsets = offsets.dropna()
        if len(valid_offsets) > 0:
            self.assertTrue((valid_offsets < 0.1).all())  # < 10cm

    def test_deviation_real_laps_reasonable_range(self) -> None:
        from racechrono_lap_analyzer.analysis import compute_line_deviations

        lap9 = load_lap(str(LAP9_CSV))
        lap13 = load_lap(str(LAP13_CSV))
        offsets = compute_line_deviations(lap9, lap13)

        # Racing line differences can be up to ~20m in extreme cases
        valid_offsets = offsets.dropna()
        if len(valid_offsets) > 0:
            self.assertLess(valid_offsets.max(), 25.0)  # Max < 25m
            self.assertGreater(valid_offsets.max(), 0.01)  # Should have some difference


class DeviationZoneTests(unittest.TestCase):
    def test_detect_zones_filters_by_threshold(self) -> None:
        from racechrono_lap_analyzer.analysis import detect_deviation_zones

        offsets = pd.Series([0.5, 0.5, 2.0, 2.5, 2.0, 0.5, 0.5], index=range(7))
        zones = detect_deviation_zones(offsets, threshold_m=1.0, min_length_m=1.0)

        self.assertGreater(len(zones), 0)
        self.assertGreater(zones[0].max_offset_m, 1.0)

    def test_detect_zones_filters_short(self) -> None:
        from racechrono_lap_analyzer.analysis import detect_deviation_zones

        # Zone shorter than min_length should be filtered
        offsets = pd.Series([0] * 50 + [2] * 5 + [0] * 50, index=range(105))
        zones = detect_deviation_zones(offsets, threshold_m=1.0, min_length_m=10)

        self.assertEqual(len(zones), 0)

    def test_detect_zones_empty_input(self) -> None:
        from racechrono_lap_analyzer.analysis import detect_deviation_zones

        offsets = pd.Series(dtype=float)
        zones = detect_deviation_zones(offsets)

        self.assertEqual(len(zones), 0)


if __name__ == "__main__":
    unittest.main()
