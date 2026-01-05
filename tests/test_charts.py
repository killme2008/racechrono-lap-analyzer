import unittest

import numpy as np
import pandas as pd

from racechrono_lap_analyzer.charts import (
    create_gg_diagram,
    create_speed_comparison_chart,
    create_throttle_brake_chart,
    create_track_line_comparison,
    create_track_map,
)
from racechrono_lap_analyzer.data_parser import LapData, SessionMetadata


def _make_lap(name: str, df: pd.DataFrame, lap_distance: float, has_throttle: bool = False) -> LapData:
    return LapData(
        name=name,
        metadata=SessionMetadata(filename=name),
        df=df,
        lap_distance=lap_distance,
        lap_time=float(df['time'].max()) if 'time' in df.columns else 0.0,
        has_throttle=has_throttle,
    )


class TrackMapTests(unittest.TestCase):
    def test_track_map_ignores_laps_without_gps(self) -> None:
        df_gps = pd.DataFrame(
            {
                'latitude': [1.0, 1.1, 1.2],
                'longitude': [2.0, 2.1, 2.2],
                'speed_kmh': [10.0, 20.0, 30.0],
            }
        )
        df_no_gps = pd.DataFrame({'speed_kmh': [5.0, 6.0]})
        lap_gps = _make_lap('gps', df_gps, lap_distance=2.0)
        lap_no_gps = _make_lap('nogps', df_no_gps, lap_distance=1.0)

        fig = create_track_map([lap_no_gps, lap_gps])
        center = fig.layout.mapbox.center
        self.assertAlmostEqual(center.lat, np.mean(df_gps['latitude']))
        self.assertAlmostEqual(center.lon, np.mean(df_gps['longitude']))

    def test_track_map_no_gps_annotation(self) -> None:
        df_no_gps = pd.DataFrame({'speed_kmh': [5.0, 6.0]})
        lap_no_gps = _make_lap('nogps', df_no_gps, lap_distance=1.0)

        fig = create_track_map([lap_no_gps])
        self.assertTrue(fig.layout.annotations)
        self.assertEqual(fig.layout.annotations[0].text, 'No GPS data available')


class ThrottleBrakeTests(unittest.TestCase):
    def test_throttle_chart_uses_consistent_column(self) -> None:
        # Need enough data points for interpolation in align_laps
        df_throttle = pd.DataFrame(
            {
                'distance': [0.0, 1.0, 2.0, 3.0, 4.0],
                'time': [0.0, 1.0, 2.0, 3.0, 4.0],
                'throttle_pos': [10.0, 20.0, 30.0, 40.0, 50.0],
            }
        )
        df_pedal = pd.DataFrame(
            {
                'distance': [0.0, 1.0, 2.0, 3.0, 4.0],
                'time': [0.0, 1.0, 2.0, 3.0, 4.0],
                'accelerator_pos': [15.0, 25.0, 35.0, 45.0, 55.0],
            }
        )
        lap_throttle = _make_lap('throttle', df_throttle, lap_distance=4.0, has_throttle=True)
        lap_pedal = _make_lap('pedal', df_pedal, lap_distance=4.0, has_throttle=True)

        fig = create_throttle_brake_chart([lap_throttle, lap_pedal])
        self.assertIsNotNone(fig)
        trace_names = [trace.name for trace in fig.data]
        self.assertTrue(any('Throttle' in name for name in trace_names))


class GGDiagramTests(unittest.TestCase):
    def test_gg_diagram_uses_equal_scaling(self) -> None:
        df = pd.DataFrame(
            {
                'lateral_acc': [0.1, 0.2],
                'longitudinal_acc': [0.0, -0.1],
            }
        )
        lap1 = _make_lap('lap1', df, lap_distance=1.0)
        lap2 = _make_lap('lap2', df, lap_distance=1.0)

        fig = create_gg_diagram([lap1, lap2])
        self.assertEqual(fig.layout.xaxis.scaleanchor, 'y')
        self.assertEqual(fig.layout.xaxis2.scaleanchor, 'y2')


class SpeedDeltaTests(unittest.TestCase):
    def test_speed_delta_range_is_dynamic(self) -> None:
        df_slow = pd.DataFrame(
            {
                'distance': [0.0, 1.0, 2.0],
                'time': [0.0, 1.0, 2.0],
                'speed_kmh': [0.0, 0.0, 0.0],
            }
        )
        df_fast = pd.DataFrame(
            {
                'distance': [0.0, 1.0, 2.0],
                'time': [0.0, 1.0, 2.0],
                'speed_kmh': [100.0, 100.0, 100.0],
            }
        )
        lap_slow = _make_lap('slow', df_slow, lap_distance=2.0)
        lap_fast = _make_lap('fast', df_fast, lap_distance=2.0)

        fig = create_speed_comparison_chart([lap_slow, lap_fast], show_delta=True)
        self.assertGreater(fig.layout.yaxis2.range[1], 30)


class TrackLineComparisonTests(unittest.TestCase):
    def test_line_comparison_mode_uses_lines(self) -> None:
        df_gps = pd.DataFrame(
            {
                'distance': [0.0, 1.0, 2.0, 3.0, 4.0],
                'time': [0.0, 1.0, 2.0, 3.0, 4.0],
                'latitude': [31.0, 31.001, 31.002, 31.003, 31.004],
                'longitude': [121.0, 121.001, 121.002, 121.003, 121.004],
                'speed_kmh': [50.0, 60.0, 70.0, 60.0, 50.0],
            }
        )
        lap1 = _make_lap('lap1', df_gps, lap_distance=4.0)
        lap2 = _make_lap('lap2', df_gps, lap_distance=4.0)

        fig = create_track_line_comparison([lap1, lap2], mode="line_comparison")

        # Should have line traces
        line_traces = [t for t in fig.data if hasattr(t, 'mode') and t.mode == 'lines']
        self.assertGreater(len(line_traces), 0)

    def test_speed_coloring_mode_delegates_to_track_map(self) -> None:
        df_gps = pd.DataFrame(
            {
                'latitude': [31.0, 31.001, 31.002],
                'longitude': [121.0, 121.001, 121.002],
                'speed_kmh': [50.0, 60.0, 70.0],
            }
        )
        lap1 = _make_lap('lap1', df_gps, lap_distance=2.0)

        fig = create_track_line_comparison([lap1], mode="speed_coloring")

        # Should be marker-based (from create_track_map)
        marker_traces = [t for t in fig.data if hasattr(t, 'mode') and t.mode == 'markers']
        self.assertGreater(len(marker_traces), 0)

    def test_no_gps_shows_annotation(self) -> None:
        df_no_gps = pd.DataFrame({'speed_kmh': [5.0, 6.0]})
        lap = _make_lap('nogps', df_no_gps, lap_distance=1.0)

        fig = create_track_line_comparison([lap], mode="line_comparison")

        self.assertTrue(fig.layout.annotations)

    def test_single_lap_no_deviation_highlights(self) -> None:
        df_gps = pd.DataFrame(
            {
                'distance': [0.0, 1.0, 2.0, 3.0, 4.0],
                'time': [0.0, 1.0, 2.0, 3.0, 4.0],
                'latitude': [31.0, 31.001, 31.002, 31.003, 31.004],
                'longitude': [121.0, 121.001, 121.002, 121.003, 121.004],
                'speed_kmh': [50.0, 60.0, 70.0, 60.0, 50.0],
            }
        )
        lap = _make_lap('lap1', df_gps, lap_distance=4.0)

        fig = create_track_line_comparison([lap], mode="line_comparison")

        # Single lap should not have deviation highlight traces
        deviation_traces = [
            t for t in fig.data
            if hasattr(t, 'name') and t.name and 'deviation' in t.name.lower()
        ]
        self.assertEqual(len(deviation_traces), 0)


if __name__ == '__main__':
    unittest.main()
