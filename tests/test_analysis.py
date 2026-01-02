"""Tests for analysis module."""

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from racechrono_lap_analyzer.analysis import (
    align_laps,
    compare_corners,
    compute_time_delta,
    corners_from_track_config,
    detect_corners,
    find_bottlenecks,
    generate_coach_insights,
    get_available_tracks,
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


if __name__ == "__main__":
    unittest.main()
