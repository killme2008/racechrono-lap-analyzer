"""
Lap data analysis module.

Provides functions for:
- Data alignment by distance
- Bottleneck detection
- Racing technique analysis (braking, cornering, acceleration)
- Track configuration support
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import find_peaks

from racechrono_lap_analyzer.data_parser import LapData
from racechrono_lap_analyzer.i18n import t

if TYPE_CHECKING:
    from racechrono_lap_analyzer.rules import RuleConfig

# Track configurations directory
TRACKS_DIR = Path(__file__).parent / "tracks"


def load_track_config(track_name: str) -> dict | None:
    """Load track configuration from JSON file."""
    config_path = TRACKS_DIR / f"{track_name}.json"
    if config_path.exists():
        with open(config_path, encoding='utf-8') as f:
            return json.load(f)
    return None


def get_available_tracks() -> list[str]:
    """List available track configurations."""
    if not TRACKS_DIR.exists():
        return []
    return [f.stem for f in TRACKS_DIR.glob("*.json")]


def corners_from_track_config(
    df: pd.DataFrame,
    track_config: dict
) -> list[Corner]:
    """
    Create Corner objects from track configuration.

    Uses predefined corner ranges (start_m, end_m) and enriches with actual data.
    """
    corners = []
    distance = df['distance'].values
    speed = df['speed_kmh'].values if 'speed_kmh' in df.columns else np.zeros_like(distance)
    lat_acc = df['lateral_acc'].values if 'lateral_acc' in df.columns else np.zeros_like(distance)
    lon_acc = df['longitudinal_acc'].values if 'longitudinal_acc' in df.columns else np.zeros_like(distance)

    for i, corner_cfg in enumerate(track_config.get('corners', [])):
        apex_dist = corner_cfg['apex_distance_m']

        # Use start_m/end_m if available, otherwise fallback to apex +/- 50m
        start_m = corner_cfg.get('start_m', apex_dist - 50)
        end_m = corner_cfg.get('end_m', apex_dist + 50)

        corner_mask = (distance >= start_m) & (distance <= end_m)

        if corner_mask.sum() == 0:
            continue

        # Extract actual values from data
        corner_speed = speed[corner_mask]
        corner_lat = lat_acc[corner_mask]
        lon_acc[corner_mask]

        min_speed = np.nanmin(corner_speed) if len(corner_speed) > 0 else corner_cfg.get('typical_speed_kmh', 0)
        max_lat_g = np.nanmax(np.abs(corner_lat)) if len(corner_lat) > 0 else 0

        # Find brake point (where lon_acc < -0.3 before apex)
        pre_apex_mask = (distance >= start_m - 50) & (distance <= apex_dist)
        brake_point = start_m  # default
        if pre_apex_mask.sum() > 0:
            pre_lon = lon_acc[pre_apex_mask]
            pre_dist = distance[pre_apex_mask]
            brake_idx = np.where(pre_lon < -0.3)[0]
            if len(brake_idx) > 0:
                brake_point = pre_dist[brake_idx[0]]

        # Find throttle point (where lon_acc > 0.1 after apex)
        post_apex_mask = (distance >= apex_dist) & (distance <= end_m + 50)
        throttle_point = end_m  # default
        if post_apex_mask.sum() > 0:
            post_lon = lon_acc[post_apex_mask]
            post_dist = distance[post_apex_mask]
            throttle_idx = np.where(post_lon > 0.1)[0]
            if len(throttle_idx) > 0:
                throttle_point = post_dist[throttle_idx[0]]

        corners.append(Corner(
            index=i + 1,
            start_distance=start_m,
            apex_distance=apex_dist,
            end_distance=end_m,
            direction=corner_cfg.get('direction', 'unknown'),
            max_lateral_g=max_lat_g,
            min_speed_kmh=min_speed,
            brake_point=brake_point,
            throttle_point=throttle_point
        ))

    return corners


@dataclass
class Corner:
    """Detected corner information."""

    index: int  # Corner number
    start_distance: float  # Entry point (m)
    apex_distance: float  # Apex point (m)
    end_distance: float  # Exit point (m)
    direction: str  # 'left' or 'right'
    max_lateral_g: float  # Peak lateral G
    min_speed_kmh: float  # Minimum speed in corner
    brake_point: float  # Braking start distance
    throttle_point: float  # Throttle application distance


@dataclass
class BottleneckSection:
    """A section where significant time is lost/gained."""

    start_m: float
    end_m: float
    speed_diff_kmh: float  # Positive = faster, negative = slower
    time_diff_ms: float  # Time gained/lost in ms
    category: str  # 'braking', 'cornering', 'acceleration', 'top_speed'
    description: str  # Human-readable description


@dataclass
class RacingInsight:
    """An observation about racing technique."""

    category: str  # 'braking', 'cornering', 'acceleration', 'line', 'grip'
    severity: str  # 'info', 'suggestion', 'warning'
    title: str  # English fallback or translation key
    description: str  # English fallback or translation key
    distance_range: tuple[float, float] | None = None
    # i18n support: if set, use these keys for translation
    title_key: str | None = None
    description_key: str | None = None
    format_args: dict | None = None  # Args for string formatting


@dataclass
class CornerComparison:
    """Comparison of a single corner between two laps."""

    corner_index: int
    direction: str
    apex_distance: float

    # Speed differences (compare_lap - ref_lap, negative = slower)
    entry_speed_diff: float
    apex_speed_diff: float
    exit_speed_diff: float

    # Braking differences
    brake_point_diff: float  # positive = later braking
    brake_peak_g_diff: float

    # Lateral G difference
    max_lateral_g_diff: float

    # Time impact (positive = slower)
    time_lost_ms: float


@dataclass
class CoachInsight:
    """A prioritized, actionable coaching suggestion."""

    priority: int  # 1 = highest
    category: str  # 'braking', 'corner_speed', 'exit', 'line', 'straight'
    location: str  # e.g., "T3 (180m)" or "Main straight"
    time_benefit_ms: float
    problem: str  # Short description of the issue
    suggestion: str  # What to do about it
    difficulty: str = "medium"  # 'low', 'medium', 'high'
    risk: str = "medium"  # 'low', 'medium', 'high'


@dataclass
class TireUtilization:
    """Tire utilization statistics with evaluation."""

    avg_combined_g: float
    max_combined_g: float
    high_g_percentage: float  # Time spent >1.0G (%)
    trail_brake_percentage: float  # Time braking while turning (%)
    full_throttle_percentage: float | None  # Time at >90% throttle (%), None if no OBD

    # Evaluation ratings: "low" | "medium" | "good" | "excellent"
    high_g_rating: str
    trail_brake_rating: str
    throttle_rating: str | None


@dataclass
class GGDiagramInsight:
    """Insights derived from G-G diagram analysis."""

    # Left/right turn utilization
    left_turn_avg_g: float  # Average |lateral G| in left turns (lat < -0.3)
    right_turn_avg_g: float  # Average |lateral G| in right turns (lat > 0.3)
    left_right_balance: float  # right - left, positive = right stronger

    # Quadrant fill rates (percentage of time in each quadrant)
    brake_left_pct: float  # Braking + left turn
    brake_right_pct: float  # Braking + right turn
    accel_left_pct: float  # Accelerating + left turn
    accel_right_pct: float  # Accelerating + right turn

    # Generated insights
    insights: list[str]  # Human-readable insight strings


def resample_by_distance(
    df: pd.DataFrame,
    distance_points: np.ndarray,
    columns: list[str]
) -> pd.DataFrame:
    """
    Resample data to uniform distance intervals for comparison.
    Uses linear interpolation.
    """
    result = {'distance': distance_points}

    for col in columns:
        if col not in df.columns or df[col].isna().all():
            result[col] = np.full_like(distance_points, np.nan)
            continue

        valid_mask = ~df[col].isna() & ~df['distance'].isna()
        if valid_mask.sum() <= 2:
            result[col] = np.full_like(distance_points, np.nan)
            continue

        x = df.loc[valid_mask, 'distance'].to_numpy()
        y = df.loc[valid_mask, col].to_numpy()
        finite_mask = np.isfinite(x) & np.isfinite(y)
        if finite_mask.sum() <= 2:
            result[col] = np.full_like(distance_points, np.nan)
            continue

        x = x[finite_mask]
        y = y[finite_mask]
        order = np.argsort(x)
        x = x[order]
        y = y[order]
        if len(x) <= 2:
            result[col] = np.full_like(distance_points, np.nan)
            continue

        unique_mask = np.concatenate(([True], np.diff(x) != 0))
        x = x[unique_mask]
        y = y[unique_mask]
        if len(x) <= 2:
            result[col] = np.full_like(distance_points, np.nan)
            continue

        try:
            interp_func = interp1d(
                x,
                y,
                kind='linear',
                bounds_error=False,
                fill_value=np.nan
            )
            result[col] = interp_func(distance_points)
        except Exception:
            result[col] = np.full_like(distance_points, np.nan)

    return pd.DataFrame(result)


def align_laps(
    laps: list[LapData],
    resolution_m: float = 1.0
) -> tuple[np.ndarray, list[pd.DataFrame]]:
    """
    Align multiple laps by distance for comparison.

    Returns:
        - Common distance array
        - List of resampled DataFrames
    """
    if not laps:
        return np.array([]), []

    # Use longest distance to avoid truncating longer laps; shorter laps will yield NaNs.
    max_dist = max(lap.lap_distance for lap in laps)
    if max_dist <= 0:
        return np.array([]), []

    distances = np.arange(0, max_dist, resolution_m)

    # Columns to resample
    columns = [
        'speed_kmh', 'time',
        'lateral_acc', 'longitudinal_acc', 'combined_acc',
        'latitude', 'longitude',
        'throttle_pos', 'accelerator_pos', 'brake_pos',
        'rpm', 'lean_angle', 'bearing'
    ]

    aligned = []
    for lap in laps:
        resampled = resample_by_distance(lap.df, distances, columns)
        aligned.append(resampled)

    return distances, aligned


def detect_corners(
    df: pd.DataFrame,
    lateral_g_threshold: float = 0.4,
    min_corner_distance: float = 80.0
) -> list[Corner]:
    """
    Detect corners based on lateral acceleration.

    A corner is detected when lateral G exceeds threshold.
    Default parameters tuned for Tianma Circuit (14 corners).
    """
    if 'lateral_acc' not in df.columns or 'distance' not in df.columns:
        return []

    if len(df) < 10:
        return []

    corners = []
    lat_acc = df['lateral_acc'].values
    distance = df['distance'].values
    speed = df['speed_kmh'].values if 'speed_kmh' in df.columns else np.zeros_like(distance)
    lon_acc = df['longitudinal_acc'].values if 'longitudinal_acc' in df.columns else np.zeros_like(distance)

    # Estimate samples per meter based on actual data
    total_distance = distance[-1] - distance[0]
    if total_distance <= 0:
        return []
    samples_per_meter = len(distance) / total_distance
    min_samples = max(10, int(min_corner_distance * samples_per_meter))

    # Find peaks in absolute lateral acceleration
    abs_lat = np.abs(lat_acc)
    peaks, properties = find_peaks(abs_lat, height=lateral_g_threshold, distance=min_samples)

    for i, peak_idx in enumerate(peaks):
        # Find corner boundaries (where lateral G drops below threshold/2)
        threshold = lateral_g_threshold / 2

        # Find start
        start_idx = peak_idx
        while start_idx > 0 and abs(lat_acc[start_idx]) > threshold:
            start_idx -= 1

        # Find end
        end_idx = peak_idx
        while end_idx < len(lat_acc) - 1 and abs(lat_acc[end_idx]) > threshold:
            end_idx += 1

        # Determine direction
        direction = 'right' if lat_acc[peak_idx] > 0 else 'left'

        # Find minimum speed in corner
        corner_speed = speed[start_idx:end_idx+1]
        min_speed = np.min(corner_speed) if len(corner_speed) > 0 else 0

        # Find brake point (where longitudinal G goes negative before corner)
        brake_point = distance[start_idx]
        for j in range(start_idx, max(0, start_idx - 500), -1):
            if lon_acc[j] < -0.2:  # Braking detected
                brake_point = distance[j]
                break

        # Find throttle point (where longitudinal G goes positive after apex)
        throttle_point = distance[end_idx]
        for j in range(peak_idx, min(len(lon_acc), end_idx + 200)):
            if lon_acc[j] > 0.1:  # Acceleration detected
                throttle_point = distance[j]
                break

        corners.append(Corner(
            index=i + 1,
            start_distance=distance[start_idx],
            apex_distance=distance[peak_idx],
            end_distance=distance[end_idx],
            direction=direction,
            max_lateral_g=abs(lat_acc[peak_idx]),
            min_speed_kmh=min_speed,
            brake_point=brake_point,
            throttle_point=throttle_point
        ))

    return corners


def find_bottlenecks(
    laps: list[LapData],
    threshold_kmh: float = 5.0,
    min_section_length: float = 20.0
) -> list[BottleneckSection]:
    """
    Find sections where there are significant speed differences.

    Compares all laps against the fastest lap.
    """
    if len(laps) < 2:
        return []

    # Sort by lap time (fastest first)
    sorted_laps = sorted(laps, key=lambda x: x.lap_time)
    ref_lap = sorted_laps[0]

    # Align laps
    distances, aligned = align_laps([ref_lap] + sorted_laps[1:])
    ref_data = aligned[0]

    bottlenecks = []

    for i, _lap in enumerate(sorted_laps[1:], 1):
        lap_data = aligned[i]

        # Calculate speed difference
        speed_diff = lap_data['speed_kmh'].values - ref_data['speed_kmh'].values

        # Find sections with significant gaps
        in_gap = False
        gap_start = 0
        gap_speeds = []

        for j, (dist, diff) in enumerate(zip(distances, speed_diff, strict=False)):
            if np.isnan(diff):
                continue

            if abs(diff) > threshold_kmh and not in_gap:
                in_gap = True
                gap_start = dist
                gap_speeds = [diff]
            elif in_gap:
                gap_speeds.append(diff)
                if abs(diff) <= threshold_kmh or j == len(distances) - 1:
                    in_gap = False
                    gap_end = dist

                    if gap_end - gap_start >= min_section_length:
                        avg_diff = np.nanmean(gap_speeds)

                        # Categorize the bottleneck
                        ref_lon_acc = ref_data['longitudinal_acc'].values
                        section_mask = (distances >= gap_start) & (distances <= gap_end)
                        avg_lon_acc = np.nanmean(ref_lon_acc[section_mask])

                        if avg_lon_acc < -0.3:
                            category = 'braking'
                        elif avg_lon_acc > 0.2:
                            category = 'acceleration'
                        elif abs(np.nanmean(ref_data['lateral_acc'].values[section_mask])) > 0.3:
                            category = 'cornering'
                        else:
                            category = 'top_speed'

                        # Estimate time difference (with bounds check)
                        ref_time = ref_data['time'].values
                        lap_time_arr = lap_data['time'].values
                        ref_section = ref_time[section_mask]
                        lap_section = lap_time_arr[section_mask]

                        if len(ref_section) > 0 and len(lap_section) > 0:
                            time_diff = (lap_section[-1] - lap_section[0]) - \
                                       (ref_section[-1] - ref_section[0])
                        else:
                            time_diff = 0.0

                        description = _generate_bottleneck_description(
                            category, avg_diff, gap_start, gap_end
                        )

                        bottlenecks.append(BottleneckSection(
                            start_m=gap_start,
                            end_m=gap_end,
                            speed_diff_kmh=avg_diff,
                            time_diff_ms=time_diff * 1000,
                            category=category,
                            description=description
                        ))

    return bottlenecks


def _generate_bottleneck_description(
    category: str,
    speed_diff: float,
    start: float,
    end: float
) -> str:
    """Generate human-readable description for a bottleneck."""
    faster = speed_diff > 0
    abs_diff = abs(speed_diff)

    if category == 'braking':
        if faster:
            return f"Later braking point, carrying {abs_diff:.1f} km/h more into corner"
        else:
            return f"Earlier braking, {abs_diff:.1f} km/h slower at turn-in"
    elif category == 'cornering':
        if faster:
            return f"Higher corner speed by {abs_diff:.1f} km/h"
        else:
            return f"Lower corner speed, losing {abs_diff:.1f} km/h through corner"
    elif category == 'acceleration':
        if faster:
            return f"Better exit acceleration, {abs_diff:.1f} km/h faster"
        else:
            return f"Slower exit, losing {abs_diff:.1f} km/h on acceleration"
    else:  # top_speed
        if faster:
            return f"Higher straight-line speed by {abs_diff:.1f} km/h"
        else:
            return f"Lower top speed by {abs_diff:.1f} km/h"


def analyze_racing_technique(laps: list[LapData]) -> list[RacingInsight]:
    """
    Analyze racing technique and generate insights.

    Based on racing principles:
    - Trail braking technique
    - Grip utilization (G-G diagram)
    - Throttle application timing
    - Consistency
    """
    insights = []

    if len(laps) < 2:
        return insights

    # Sort by lap time
    sorted_laps = sorted(laps, key=lambda x: x.lap_time)
    fastest = sorted_laps[0]
    slowest = sorted_laps[-1]

    # Align for comparison
    distances, aligned = align_laps(laps)

    # 1. Grip utilization analysis (G-G diagram)
    for _i, lap in enumerate(laps):
        df = lap.df
        if 'combined_acc' in df.columns and len(df) > 0:
            combined_g = np.abs(df['combined_acc'].values)
            high_g_pct = np.sum(combined_g > 0.8) / len(combined_g) * 100

            if high_g_pct < 30:
                insights.append(RacingInsight(
                    category='grip',
                    severity='suggestion',
                    title=f'{lap.name}: Low grip utilization',
                    description=f'Only {high_g_pct:.1f}% of lap at >0.8G combined. '
                               f'The tires have more grip available. Consider carrying more speed or braking later.',
                    title_key='analysis.technique.grip_low.title',
                    description_key='analysis.technique.grip_low.description',
                    format_args={'name': lap.name, 'pct': high_g_pct}
                ))

    # 2. Trail braking analysis
    for _i, lap in enumerate(laps):
        df = lap.df
        if 'lateral_acc' in df.columns and 'longitudinal_acc' in df.columns:
            lat = df['lateral_acc'].values
            lon = df['longitudinal_acc'].values

            # Find corner entry zones (increasing lateral G)
            corners = detect_corners(df)
            good_trail_brake = 0
            total_corners = len(corners)

            for corner in corners:
                # Check if braking continues into corner entry
                mask = (df['distance'] >= corner.brake_point) & (df['distance'] <= corner.apex_distance)
                if mask.sum() > 0:
                    entry_lon = lon[mask]
                    entry_lat = lat[mask]

                    # Good trail braking: braking continues as lateral G builds
                    if np.any((entry_lon < -0.1) & (np.abs(entry_lat) > 0.2)):
                        good_trail_brake += 1

            if total_corners > 0:
                trail_brake_pct = good_trail_brake / total_corners * 100
                if trail_brake_pct < 50:
                    insights.append(RacingInsight(
                        category='braking',
                        severity='suggestion',
                        title=f'{lap.name}: Trail braking opportunity',
                        description=f'Trail braking detected in only {trail_brake_pct:.0f}% of corners. '
                                   f'Maintaining brake pressure into corner entry can help rotate the car and improve lap time.',
                        title_key='analysis.technique.trail_brake.title',
                        description_key='analysis.technique.trail_brake.description',
                        format_args={'name': lap.name, 'pct': trail_brake_pct}
                    ))

    # 3. Throttle application analysis (if OBD data available)
    for _i, lap in enumerate(laps):
        df = lap.df
        throttle_col = None
        for col in ['throttle_pos', 'accelerator_pos']:
            if col in df.columns and not df[col].isna().all():
                throttle_col = col
                break

        if throttle_col:
            throttle = df[throttle_col].values
            distance = df['distance'].values

            # Find corners
            corners = detect_corners(df)

            for corner in corners:
                # Time from apex to full throttle
                apex_idx = np.argmin(np.abs(distance - corner.apex_distance))
                post_apex = throttle[apex_idx:]

                full_throttle_idx = np.where(post_apex > 90)[0]
                if len(full_throttle_idx) > 0:
                    dist_to_full = distance[apex_idx + full_throttle_idx[0]] - corner.apex_distance

                    if dist_to_full > 50:  # More than 50m to full throttle
                        insights.append(RacingInsight(
                            category='acceleration',
                            severity='info',
                            title=f'{lap.name}: Delayed throttle in corner {corner.index}',
                            description=f'Full throttle reached {dist_to_full:.0f}m after apex. '
                                       f'Earlier throttle application (if traction allows) could improve exit speed.',
                            distance_range=(corner.apex_distance, corner.end_distance),
                            title_key='analysis.technique.throttle_delayed.title',
                            description_key='analysis.technique.throttle_delayed.description',
                            format_args={'name': lap.name, 'corner': corner.index, 'dist': dist_to_full}
                        ))

    # 4. Compare fastest vs slowest lap
    if len(laps) >= 2:
        slowest.lap_time - fastest.lap_time

        # Find where time is lost
        bottlenecks = find_bottlenecks([fastest, slowest])
        braking_bottlenecks = [b for b in bottlenecks if b.category == 'braking']
        corner_bottlenecks = [b for b in bottlenecks if b.category == 'cornering']
        accel_bottlenecks = [b for b in bottlenecks if b.category == 'acceleration']

        if braking_bottlenecks:
            total_brake_loss = sum(b.time_diff_ms for b in braking_bottlenecks if b.speed_diff_kmh < 0)
            if total_brake_loss > 100:  # > 100ms lost
                insights.append(RacingInsight(
                    category='braking',
                    severity='warning',
                    title='Braking is a key area for improvement',
                    description=f'Approximately {total_brake_loss:.0f}ms lost in braking zones. '
                               f'Focus on braking later and trail braking into corners.',
                    title_key='analysis.technique.braking_key.title',
                    description_key='analysis.technique.braking_key.description',
                    format_args={'time': total_brake_loss}
                ))

        if corner_bottlenecks:
            total_corner_loss = sum(b.time_diff_ms for b in corner_bottlenecks if b.speed_diff_kmh < 0)
            if total_corner_loss > 100:
                insights.append(RacingInsight(
                    category='cornering',
                    severity='warning',
                    title='Corner speed needs improvement',
                    description=f'Approximately {total_corner_loss:.0f}ms lost in corners. '
                               f'Consider a wider line or more confidence through mid-corner.',
                    title_key='analysis.technique.corner_speed_key.title',
                    description_key='analysis.technique.corner_speed_key.description',
                    format_args={'time': total_corner_loss}
                ))

        if accel_bottlenecks:
            total_accel_loss = sum(b.time_diff_ms for b in accel_bottlenecks if b.speed_diff_kmh < 0)
            if total_accel_loss > 100:
                insights.append(RacingInsight(
                    category='acceleration',
                    severity='warning',
                    title='Exit acceleration is losing time',
                    description=f'Approximately {total_accel_loss:.0f}ms lost on corner exits. '
                               f'Focus on getting on throttle earlier and maximizing traction.',
                    title_key='analysis.technique.exit_accel_key.title',
                    description_key='analysis.technique.exit_accel_key.description',
                    format_args={'time': total_accel_loss}
                ))

    # 5. Consistency analysis
    if len(laps) >= 2:
        lap_times = [lap.lap_time for lap in laps]
        std_dev = np.std(lap_times)
        mean_time = np.mean(lap_times)
        cv = std_dev / mean_time * 100  # Coefficient of variation

        if cv > 2:
            insights.append(RacingInsight(
                category='consistency',
                severity='info',
                title='Lap time variation detected',
                description=f'Lap times vary by {cv:.1f}% (std dev: {std_dev:.2f}s). '
                           f'Focus on repeating the techniques from your fastest lap.',
                title_key='analysis.technique.consistency.title',
                description_key='analysis.technique.consistency.description',
                format_args={'cv': cv, 'std': std_dev}
            ))

    return insights


def compute_time_delta(
    ref_lap: LapData,
    compare_lap: LapData,
    resolution_m: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute cumulative time delta between two laps.

    Returns:
        - Distance array
        - Cumulative time delta (positive = compare is slower)
    """
    distances, aligned = align_laps([ref_lap, compare_lap], resolution_m)
    ref_time = aligned[0]['time'].values
    cmp_time = aligned[1]['time'].values

    # Cumulative time delta
    time_delta = cmp_time - ref_time

    return distances, time_delta


def compare_corners(
    ref_lap: LapData,
    compare_lap: LapData,
    track_config: dict | None = None
) -> list[CornerComparison]:
    """
    Compare corners between two laps.

    Returns list of CornerComparison objects for each detected corner.
    If track_config is provided, uses predefined corner positions.
    """
    # Use track config if provided, otherwise detect corners
    if track_config:
        ref_corners = corners_from_track_config(ref_lap.df, track_config)
    else:
        ref_corners = detect_corners(ref_lap.df)

    if not ref_corners:
        return []

    # Align laps for comparison
    distances, aligned = align_laps([ref_lap, compare_lap], resolution_m=1.0)
    if len(distances) == 0:
        return []

    ref_data = aligned[0]
    cmp_data = aligned[1]

    comparisons = []

    for corner in ref_corners:
        # Define corner zones
        entry_start = corner.brake_point
        entry_end = corner.start_distance
        apex_zone_start = corner.start_distance
        apex_zone_end = corner.apex_distance
        exit_start = corner.apex_distance
        exit_end = corner.end_distance

        # Get masks for each zone
        entry_mask = (distances >= entry_start) & (distances <= entry_end)
        apex_mask = (distances >= apex_zone_start) & (distances <= apex_zone_end)
        exit_mask = (distances >= exit_start) & (distances <= exit_end)

        # Calculate speed differences
        ref_speed = ref_data['speed_kmh'].values
        cmp_speed = cmp_data['speed_kmh'].values

        entry_speed_diff = 0.0
        if entry_mask.sum() > 0:
            ref_entry = np.nanmean(ref_speed[entry_mask])
            cmp_entry = np.nanmean(cmp_speed[entry_mask])
            if not np.isnan(ref_entry) and not np.isnan(cmp_entry):
                entry_speed_diff = cmp_entry - ref_entry

        apex_speed_diff = 0.0
        if apex_mask.sum() > 0:
            ref_apex = np.nanmin(ref_speed[apex_mask])
            cmp_apex = np.nanmin(cmp_speed[apex_mask])
            if not np.isnan(ref_apex) and not np.isnan(cmp_apex):
                apex_speed_diff = cmp_apex - ref_apex

        exit_speed_diff = 0.0
        if exit_mask.sum() > 0:
            ref_exit = np.nanmean(ref_speed[exit_mask])
            cmp_exit = np.nanmean(cmp_speed[exit_mask])
            if not np.isnan(ref_exit) and not np.isnan(cmp_exit):
                exit_speed_diff = cmp_exit - ref_exit

        # Compare brake points (find where longitudinal G < -0.3)
        ref_lon = ref_data['longitudinal_acc'].values
        cmp_lon = cmp_data['longitudinal_acc'].values

        brake_zone_mask = (distances >= entry_start - 100) & (distances <= entry_end)
        brake_zone_dist = distances[brake_zone_mask]

        ref_brake_point = entry_start
        cmp_brake_point = entry_start

        if len(brake_zone_dist) > 0:
            ref_lon_zone = ref_lon[brake_zone_mask]
            cmp_lon_zone = cmp_lon[brake_zone_mask]

            ref_brake_idx = np.where(ref_lon_zone < -0.3)[0]
            if len(ref_brake_idx) > 0:
                ref_brake_point = brake_zone_dist[ref_brake_idx[0]]

            cmp_brake_idx = np.where(cmp_lon_zone < -0.3)[0]
            if len(cmp_brake_idx) > 0:
                cmp_brake_point = brake_zone_dist[cmp_brake_idx[0]]

        brake_point_diff = cmp_brake_point - ref_brake_point  # positive = later braking

        # Compare peak braking G
        brake_peak_g_diff = 0.0
        full_entry_mask = (distances >= entry_start - 50) & (distances <= apex_zone_end)
        if full_entry_mask.sum() > 0:
            ref_peak = np.nanmin(ref_lon[full_entry_mask])
            cmp_peak = np.nanmin(cmp_lon[full_entry_mask])
            if not np.isnan(ref_peak) and not np.isnan(cmp_peak):
                brake_peak_g_diff = cmp_peak - ref_peak  # negative means stronger braking

        # Compare max lateral G
        ref_lat = ref_data['lateral_acc'].values
        cmp_lat = cmp_data['lateral_acc'].values

        corner_mask = (distances >= corner.start_distance) & (distances <= corner.end_distance)
        max_lateral_g_diff = 0.0
        if corner_mask.sum() > 0:
            ref_max_lat = np.nanmax(np.abs(ref_lat[corner_mask]))
            cmp_max_lat = np.nanmax(np.abs(cmp_lat[corner_mask]))
            if not np.isnan(ref_max_lat) and not np.isnan(cmp_max_lat):
                max_lateral_g_diff = cmp_max_lat - ref_max_lat

        # Calculate time lost in this corner
        ref_time = ref_data['time'].values
        cmp_time = cmp_data['time'].values

        full_corner_mask = (distances >= entry_start) & (distances <= exit_end)
        time_lost_ms = 0.0
        if full_corner_mask.sum() > 0:
            ref_times = ref_time[full_corner_mask]
            cmp_times = cmp_time[full_corner_mask]
            valid_ref = ref_times[~np.isnan(ref_times)]
            valid_cmp = cmp_times[~np.isnan(cmp_times)]
            if len(valid_ref) > 1 and len(valid_cmp) > 1:
                ref_duration = valid_ref[-1] - valid_ref[0]
                cmp_duration = valid_cmp[-1] - valid_cmp[0]
                time_lost_ms = (cmp_duration - ref_duration) * 1000

        comparisons.append(CornerComparison(
            corner_index=corner.index,
            direction=corner.direction,
            apex_distance=corner.apex_distance,
            entry_speed_diff=entry_speed_diff,
            apex_speed_diff=apex_speed_diff,
            exit_speed_diff=exit_speed_diff,
            brake_point_diff=brake_point_diff,
            brake_peak_g_diff=brake_peak_g_diff,
            max_lateral_g_diff=max_lateral_g_diff,
            time_lost_ms=time_lost_ms
        ))

    return comparisons


def generate_coach_insights(
    ref_lap: LapData,
    compare_lap: LapData,
    max_insights: int = 5,
    track_config: dict | None = None,
    rule_config: RuleConfig | None = None,
) -> list[CoachInsight]:
    """
    Generate prioritized coach-style insights.

    Analyzes the differences between two laps and returns
    actionable suggestions sorted by potential time benefit.

    Uses adaptive thresholds based on corner type if rule_config is provided.
    """
    # Import here to avoid circular imports
    from racechrono_lap_analyzer.rules import (
        RuleConfig,
        estimate_time_benefit_physics,
        get_corner_config_by_index,
        get_corner_type,
        get_thresholds,
    )

    if rule_config is None:
        rule_config = RuleConfig()

    insights = []

    # Get corner comparisons (use track config if provided)
    corner_comparisons = compare_corners(ref_lap, compare_lap, track_config)

    # Analyze each corner for issues
    for cc in corner_comparisons:
        # Skip corners where compare lap is faster
        if cc.time_lost_ms <= rule_config.min_time_benefit_ms:
            continue

        location = f"T{cc.corner_index} ({cc.apex_distance:.0f}m)"

        # Get corner config and adaptive thresholds
        corner_cfg = get_corner_config_by_index(cc.corner_index, track_config)
        corner_type = get_corner_type(corner_cfg, None)
        thresholds = get_thresholds(corner_type, rule_config, corner_cfg)

        # Get difficulty and risk based on corner type
        # Use corner config type if available, otherwise infer from apex speed
        cfg_corner_type = corner_cfg.get('type') if corner_cfg else None
        difficulty, risk = get_corner_difficulty_risk(cfg_corner_type, cc.apex_speed_diff)

        # Estimate section length for physics-based time calculation
        section_length = 100.0  # Default estimate

        # Early braking detection
        if cc.brake_point_diff < -thresholds.brake_early_m:
            time_benefit = estimate_time_benefit_physics(
                cc.apex_speed_diff, section_length, 80.0, corner_type
            ) if cc.apex_speed_diff else abs(cc.time_lost_ms * 0.4)

            insights.append(CoachInsight(
                priority=0,
                category='braking',
                location=location,
                time_benefit_ms=time_benefit,
                problem=t("analysis.coach.brake_early.problem", distance=abs(cc.brake_point_diff)),
                suggestion=t("analysis.coach.brake_early.suggestion", distance=abs(cc.brake_point_diff)),
                difficulty=difficulty,
                risk=risk
            ))

        # Low mid-corner speed
        if cc.apex_speed_diff < -thresholds.apex_speed_low_kmh:
            time_benefit = estimate_time_benefit_physics(
                cc.apex_speed_diff, section_length, 60.0, corner_type
            )
            insights.append(CoachInsight(
                priority=0,
                category='corner_speed',
                location=location,
                time_benefit_ms=time_benefit,
                problem=t("analysis.coach.apex_slow.problem", diff=abs(cc.apex_speed_diff)),
                suggestion=t("analysis.coach.apex_slow.suggestion"),
                difficulty=difficulty,
                risk=risk
            ))

        # Poor exit acceleration
        if cc.exit_speed_diff < -thresholds.exit_speed_low_kmh:
            time_benefit = estimate_time_benefit_physics(
                cc.exit_speed_diff, section_length * 1.5, 70.0, corner_type
            )
            insights.append(CoachInsight(
                priority=0,
                category='exit',
                location=location,
                time_benefit_ms=time_benefit,
                problem=t("analysis.coach.exit_slow.problem", diff=abs(cc.exit_speed_diff)),
                suggestion=t("analysis.coach.exit_slow.suggestion"),
                difficulty=difficulty,
                risk=risk
            ))

        # Conservative cornering (low lateral G)
        if cc.max_lateral_g_diff < -thresholds.lateral_g_low:
            insights.append(CoachInsight(
                priority=0,
                category='corner_speed',
                location=location,
                time_benefit_ms=abs(cc.time_lost_ms * 0.2),
                problem=t("analysis.coach.lateral_g_low.problem", diff=abs(cc.max_lateral_g_diff)),
                suggestion=t("analysis.coach.lateral_g_low.suggestion"),
                difficulty=difficulty,
                risk=risk
            ))

        # Weak braking force
        if cc.brake_peak_g_diff > thresholds.brake_force_weak_g:
            insights.append(CoachInsight(
                priority=0,
                category='braking',
                location=location,
                time_benefit_ms=abs(cc.time_lost_ms * 0.2),
                problem=t("analysis.coach.brake_weak.problem", diff=cc.brake_peak_g_diff),
                suggestion=t("analysis.coach.brake_weak.suggestion"),
                difficulty=difficulty,
                risk=risk
            ))

    # Analyze straights - find sections with sustained positive lon_acc
    distances, aligned = align_laps([ref_lap, compare_lap], resolution_m=1.0)
    if len(distances) > 0:
        ref_data = aligned[0]
        cmp_data = aligned[1]

        ref_speed = ref_data['speed_kmh'].values
        cmp_speed = cmp_data['speed_kmh'].values
        ref_lon = ref_data['longitudinal_acc'].values

        # Find straight sections (positive longitudinal acc, high speed)
        straight_mask = (ref_lon > 0.1) & (ref_speed > 80)  # Accelerating at >80km/h

        if straight_mask.sum() > 50:  # At least 50m of straight
            # Group consecutive straight sections
            changes = np.diff(straight_mask.astype(int))
            starts = np.where(changes == 1)[0] + 1
            ends = np.where(changes == -1)[0] + 1

            if len(starts) > 0 and len(ends) > 0:
                # Handle edge cases
                if straight_mask[0]:
                    starts = np.insert(starts, 0, 0)
                if straight_mask[-1]:
                    ends = np.append(ends, len(straight_mask))

                for start, end in zip(starts[:3], ends[:3], strict=False):  # Max 3 straights
                    if end - start < 50:
                        continue

                    section_ref = ref_speed[start:end]
                    section_cmp = cmp_speed[start:end]

                    valid_ref = section_ref[~np.isnan(section_ref)]
                    valid_cmp = section_cmp[~np.isnan(section_cmp)]

                    if len(valid_ref) > 0 and len(valid_cmp) > 0:
                        max_ref = np.max(valid_ref)
                        max_cmp = np.max(valid_cmp)
                        speed_diff = max_cmp - max_ref

                        if speed_diff < -5:  # 5+ km/h slower top speed
                            # Estimate time loss from speed difference
                            dist_m = distances[end] - distances[start] if end < len(distances) else 100
                            avg_speed = (max_ref + max_cmp) / 2
                            if avg_speed > 0:
                                time_benefit = (dist_m / avg_speed * 3.6) * (abs(speed_diff) / avg_speed) * 1000

                                insights.append(CoachInsight(
                                    priority=0,
                                    category='straight',
                                    location=f"Straight ({distances[start]:.0f}m)",
                                    time_benefit_ms=time_benefit,
                                    problem=t("analysis.coach.top_speed_low.problem", diff=abs(speed_diff)),
                                    suggestion=t("analysis.coach.top_speed_low.suggestion"),
                                    difficulty="low",
                                    risk="low"
                                ))

    # Sort by time benefit and assign priorities
    insights.sort(key=lambda x: x.time_benefit_ms, reverse=True)

    # Deduplicate by location - keep only the highest benefit issue per location
    seen_locations = set()
    deduplicated = []
    for insight in insights:
        if insight.location not in seen_locations:
            seen_locations.add(insight.location)
            deduplicated.append(insight)

    # Assign priorities and limit count
    effective_max = rule_config.max_insights if rule_config else max_insights
    result = []
    for i, insight in enumerate(deduplicated[:effective_max]):
        insight.priority = i + 1
        result.append(insight)

    return result


def get_friction_circle_analysis(
    ref_lap: LapData,
    compare_lap: LapData
) -> dict:
    """
    Analyze friction circle utilization by quadrant.

    Returns dict with utilization percentages for each zone.
    """
    result = {
        'ref': {'pure_brake': 0, 'trail_brake': 0, 'pure_corner': 0, 'exit_accel': 0},
        'compare': {'pure_brake': 0, 'trail_brake': 0, 'pure_corner': 0, 'exit_accel': 0},
        'left_right_balance': {'ref': 0, 'compare': 0}
    }

    for lap, key in [(ref_lap, 'ref'), (compare_lap, 'compare')]:
        df = lap.df
        if 'lateral_acc' not in df.columns or 'longitudinal_acc' not in df.columns:
            continue

        lat = df['lateral_acc'].values
        lon = df['longitudinal_acc'].values

        total_samples = len(lat)
        if total_samples == 0:
            continue

        # Pure braking (lon < -0.3, |lat| < 0.3)
        pure_brake = np.sum((lon < -0.3) & (np.abs(lat) < 0.3))
        result[key]['pure_brake'] = pure_brake / total_samples * 100

        # Trail braking (lon < -0.1, |lat| > 0.3)
        trail_brake = np.sum((lon < -0.1) & (np.abs(lat) > 0.3))
        result[key]['trail_brake'] = trail_brake / total_samples * 100

        # Pure cornering (|lon| < 0.2, |lat| > 0.5)
        pure_corner = np.sum((np.abs(lon) < 0.2) & (np.abs(lat) > 0.5))
        result[key]['pure_corner'] = pure_corner / total_samples * 100

        # Exit acceleration (lon > 0.1, |lat| > 0.3)
        exit_accel = np.sum((lon > 0.1) & (np.abs(lat) > 0.3))
        result[key]['exit_accel'] = exit_accel / total_samples * 100

        # Left vs right balance (average lateral G in each direction)
        left_samples = lat[lat < -0.3]
        right_samples = lat[lat > 0.3]
        if len(left_samples) > 0 and len(right_samples) > 0:
            left_avg = np.nanmean(np.abs(left_samples))
            right_avg = np.nanmean(right_samples)
            result['left_right_balance'][key] = right_avg - left_avg  # positive = stronger right

    return result


def _rate_high_g(percentage: float) -> str:
    """Rate high G percentage based on professional standards."""
    if percentage < 20:
        return "low"
    elif percentage < 30:
        return "medium"
    elif percentage < 40:
        return "good"
    else:
        return "excellent"


def _rate_trail_brake(percentage: float) -> str:
    """Rate trail braking percentage based on professional standards."""
    if percentage < 5:
        return "low"
    elif percentage < 15:
        return "medium"
    elif percentage < 25:
        return "good"
    else:
        return "excellent"


def _rate_throttle(percentage: float) -> str:
    """Rate full throttle percentage based on professional standards."""
    if percentage < 15:
        return "low"
    elif percentage < 20:
        return "medium"
    elif percentage < 30:
        return "good"
    else:
        return "excellent"


def compute_tire_utilization(lap: LapData) -> TireUtilization:
    """
    Compute tire utilization metrics for a single lap.

    Analyzes how well the driver is using the available tire grip by examining:
    - Combined G-force (how much total grip is being used)
    - High G percentage (time spent at >1.0G)
    - Trail braking (braking while turning)
    - Full throttle usage (if OBD data available)
    """
    df = lap.df

    # Get acceleration data
    lat_acc = df['lateral_acc'].values if 'lateral_acc' in df.columns else np.zeros(len(df))
    lon_acc = df['longitudinal_acc'].values if 'longitudinal_acc' in df.columns else np.zeros(len(df))
    combined_acc = df['combined_acc'].values if 'combined_acc' in df.columns else np.sqrt(lat_acc**2 + lon_acc**2)

    # Filter out NaN values
    valid_mask = np.isfinite(combined_acc) & np.isfinite(lat_acc) & np.isfinite(lon_acc)
    combined_acc_valid = combined_acc[valid_mask]
    lat_acc_valid = lat_acc[valid_mask]
    lon_acc_valid = lon_acc[valid_mask]

    total_samples = len(combined_acc_valid)
    if total_samples == 0:
        return TireUtilization(
            avg_combined_g=0.0,
            max_combined_g=0.0,
            high_g_percentage=0.0,
            trail_brake_percentage=0.0,
            full_throttle_percentage=None,
            high_g_rating="low",
            trail_brake_rating="low",
            throttle_rating=None
        )

    # Calculate combined G metrics
    avg_combined_g = float(np.nanmean(combined_acc_valid))
    max_combined_g = float(np.nanmax(combined_acc_valid))

    # High G percentage (>1.0G)
    high_g_count = np.sum(combined_acc_valid > 1.0)
    high_g_percentage = (high_g_count / total_samples) * 100

    # Trail braking: braking (lon < -0.2) while turning (|lat| > 0.3)
    trail_brake_mask = (lon_acc_valid < -0.2) & (np.abs(lat_acc_valid) > 0.3)
    trail_brake_count = np.sum(trail_brake_mask)
    trail_brake_percentage = (trail_brake_count / total_samples) * 100

    # Full throttle percentage (if OBD data available)
    full_throttle_percentage = None
    throttle_rating = None
    throttle_col = None
    for col in ['throttle_pos', 'accelerator_pos']:
        if col in df.columns and not df[col].isna().all():
            throttle_col = col
            break

    if throttle_col:
        throttle_data = df[throttle_col].values
        valid_throttle = throttle_data[np.isfinite(throttle_data)]
        if len(valid_throttle) > 0:
            full_throttle_count = np.sum(valid_throttle > 90)
            full_throttle_percentage = (full_throttle_count / len(valid_throttle)) * 100
            throttle_rating = _rate_throttle(full_throttle_percentage)

    return TireUtilization(
        avg_combined_g=avg_combined_g,
        max_combined_g=max_combined_g,
        high_g_percentage=high_g_percentage,
        trail_brake_percentage=trail_brake_percentage,
        full_throttle_percentage=full_throttle_percentage,
        high_g_rating=_rate_high_g(high_g_percentage),
        trail_brake_rating=_rate_trail_brake(trail_brake_percentage),
        throttle_rating=throttle_rating
    )


def analyze_gg_diagram(lap: LapData) -> GGDiagramInsight:
    """
    Analyze G-G diagram distribution and generate insights.

    Examines:
    - Left vs right turn utilization balance
    - Quadrant fill rates (brake+turn, accel+turn combinations)
    - Generates human-readable insights about weaknesses
    """
    df = lap.df

    lat_acc = df['lateral_acc'].values if 'lateral_acc' in df.columns else np.zeros(len(df))
    lon_acc = df['longitudinal_acc'].values if 'longitudinal_acc' in df.columns else np.zeros(len(df))

    # Filter valid samples
    valid_mask = np.isfinite(lat_acc) & np.isfinite(lon_acc)
    lat = lat_acc[valid_mask]
    lon = lon_acc[valid_mask]

    total_samples = len(lat)
    if total_samples == 0:
        return GGDiagramInsight(
            left_turn_avg_g=0.0,
            right_turn_avg_g=0.0,
            left_right_balance=0.0,
            brake_left_pct=0.0,
            brake_right_pct=0.0,
            accel_left_pct=0.0,
            accel_right_pct=0.0,
            insights=[]
        )

    # Left/right turn utilization (lat < -0.3 = left turn, lat > 0.3 = right turn)
    left_turn_mask = lat < -0.3
    right_turn_mask = lat > 0.3

    left_turn_samples = np.abs(lat[left_turn_mask])
    right_turn_samples = lat[right_turn_mask]

    left_turn_avg_g = float(np.nanmean(left_turn_samples)) if len(left_turn_samples) > 0 else 0.0
    right_turn_avg_g = float(np.nanmean(right_turn_samples)) if len(right_turn_samples) > 0 else 0.0
    left_right_balance = right_turn_avg_g - left_turn_avg_g

    # Quadrant analysis
    # brake_left: lon < -0.2 AND lat < -0.3
    brake_left = np.sum((lon < -0.2) & (lat < -0.3))
    brake_left_pct = (brake_left / total_samples) * 100

    # brake_right: lon < -0.2 AND lat > 0.3
    brake_right = np.sum((lon < -0.2) & (lat > 0.3))
    brake_right_pct = (brake_right / total_samples) * 100

    # accel_left: lon > 0.1 AND lat < -0.3
    accel_left = np.sum((lon > 0.1) & (lat < -0.3))
    accel_left_pct = (accel_left / total_samples) * 100

    # accel_right: lon > 0.1 AND lat > 0.3
    accel_right = np.sum((lon > 0.1) & (lat > 0.3))
    accel_right_pct = (accel_right / total_samples) * 100

    # Generate insights
    insights = []

    # Left/right balance insight
    if abs(left_right_balance) > 0.1:
        if left_right_balance < 0:
            # Right turns are weaker
            diff = abs(left_right_balance)
            insights.append(t("gg_analysis.right_weak", left=left_turn_avg_g, right=right_turn_avg_g, diff=diff))
        else:
            # Left turns are weaker
            diff = abs(left_right_balance)
            insights.append(t("gg_analysis.left_weak", left=left_turn_avg_g, right=right_turn_avg_g, diff=diff))

    # Find weakest quadrant
    quadrants = {
        "brake_left": brake_left_pct,
        "brake_right": brake_right_pct,
        "accel_left": accel_left_pct,
        "accel_right": accel_right_pct
    }

    # Identify quadrants that are significantly lower than average
    avg_quadrant = sum(quadrants.values()) / 4
    for name, pct in quadrants.items():
        if pct < avg_quadrant * 0.5 and avg_quadrant > 2:  # Less than half of average
            insights.append(t("gg_analysis.quadrant_weak", quadrant=t(f"gg_analysis.quadrants.{name}"), pct=pct))

    return GGDiagramInsight(
        left_turn_avg_g=left_turn_avg_g,
        right_turn_avg_g=right_turn_avg_g,
        left_right_balance=left_right_balance,
        brake_left_pct=brake_left_pct,
        brake_right_pct=brake_right_pct,
        accel_left_pct=accel_left_pct,
        accel_right_pct=accel_right_pct,
        insights=insights
    )


def get_corner_difficulty_risk(
    corner_type: str | None,
    min_speed_kmh: float | None = None
) -> tuple[str, str]:
    """
    Determine difficulty and risk based on corner type.

    Returns (difficulty, risk) tuple.

    Corner type mapping:
    - hairpin: high difficulty, low risk (technical but slow)
    - medium: medium difficulty, medium risk
    - fast: medium difficulty, high risk (speed makes mistakes costly)
    - straight: low difficulty, low risk
    """
    corner_type_map = {
        "hairpin": ("high", "low"),
        "fast": ("medium", "high"),
        "medium": ("medium", "medium"),
        "straight": ("low", "low"),
    }
    if corner_type:
        result = corner_type_map.get(corner_type.lower())
        if result:
            return result

    # Fallback: infer from min speed if available
    if min_speed_kmh is not None:
        if min_speed_kmh < 60:
            return ("high", "low")  # Slow = hairpin-like
        elif min_speed_kmh > 90:
            return ("medium", "high")  # Fast corner
        else:
            return ("medium", "medium")

    return ("medium", "medium")
