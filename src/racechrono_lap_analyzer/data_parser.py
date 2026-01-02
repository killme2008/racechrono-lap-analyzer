"""
RaceChrono Pro CSV3 format parser.

Handles the specific format exported by RaceChrono Pro, extracting
both metadata (session info) and telemetry data.
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class SessionMetadata:
    """Metadata extracted from RaceChrono CSV header."""

    filename: str = ""
    format_version: str = ""
    session_title: str = ""
    session_type: str = ""
    track_name: str = ""
    driver_name: str = ""
    created_date: str = ""
    note: str = ""


@dataclass
class LapData:
    """Container for a single lap's data."""

    name: str
    metadata: SessionMetadata
    df: pd.DataFrame
    lap_number: int = 0
    lap_time: float = 0.0
    lap_distance: float = 0.0

    # Computed statistics
    max_speed_kmh: float = 0.0
    avg_speed_kmh: float = 0.0
    max_lateral_g: float = 0.0
    max_brake_g: float = 0.0
    max_accel_g: float = 0.0

    # Available data columns
    available_columns: list = field(default_factory=list)
    has_obd_data: bool = False
    has_throttle: bool = False
    has_brake: bool = False
    has_rpm: bool = False


def parse_metadata(filepath: str) -> SessionMetadata:
    """Parse metadata from RaceChrono CSV header."""
    metadata = SessionMetadata(filename=Path(filepath).stem)

    with open(filepath, encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines[:10]:
        line = line.strip()
        if line.startswith('Format,'):
            metadata.format_version = line.split(',', 1)[1].strip()
        elif line.startswith('Session title,'):
            metadata.session_title = line.split(',', 1)[1].strip().strip('"')
        elif line.startswith('Session type,'):
            metadata.session_type = line.split(',', 1)[1].strip()
        elif line.startswith('Track name,'):
            metadata.track_name = line.split(',', 1)[1].strip().strip('"')
        elif line.startswith('Driver name,'):
            metadata.driver_name = line.split(',', 1)[1].strip()
        elif line.startswith('Created,'):
            metadata.created_date = line.split(',', 1)[1].strip()
        elif line.startswith('Note,'):
            metadata.note = line.split(',', 1)[1].strip()

    return metadata


def parse_racechrono_csv(filepath: str, target_lap: int | None = None) -> tuple[pd.DataFrame, int]:
    """
    Parse RaceChrono Pro CSV3 format file.

    RaceChrono exports include data from adjacent laps (previous lap end,
    next lap start). This function filters to keep only the target lap.

    The CSV structure:
    - Lines 1-9: Header info (file info, session details)
    - Line 10: Column names
    - Line 11: Units
    - Line 12: Data sources
    - Line 13+: Data rows

    Args:
        filepath: Path to CSV file
        target_lap: Specific lap number to extract. If None, auto-detect
                   the main lap (the one with most data points).

    Returns:
        Tuple of (DataFrame, lap_number)
    """
    with open(filepath, encoding='utf-8') as f:
        lines = f.readlines()

    # Find the header line (starts with 'timestamp,')
    header_line = None
    for i, line in enumerate(lines):
        if line.startswith('timestamp,'):
            header_line = i
            break

    if header_line is None:
        raise ValueError(f"Could not find header in {filepath}")

    # Read CSV with proper header
    df = pd.read_csv(
        filepath,
        skiprows=header_line,
        header=0,
        low_memory=False
    )

    # Skip unit and source rows (first 2 rows after header)
    df = df.iloc[2:].reset_index(drop=True)

    # Convert all columns to numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Filter to target lap only
    if 'lap_number' in df.columns:
        lap_counts = df['lap_number'].value_counts()

        actual_lap = target_lap if target_lap is not None else int(lap_counts.idxmax())

        # Filter to only this lap
        df = df[df['lap_number'] == actual_lap].reset_index(drop=True)
    else:
        actual_lap = 0

    return df, actual_lap


def normalize_lap_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize lap data for comparison.

    - Reset distance to start from 0
    - Reset time to start from 0
    - Convert speed to km/h
    """
    df = df.copy()

    # Normalize distance (handle NaN by using nanmin)
    if 'distance_traveled' in df.columns and not df['distance_traveled'].isna().all():
        min_dist = df['distance_traveled'].min(skipna=True)
        df['distance'] = df['distance_traveled'] - min_dist
    else:
        df['distance'] = np.arange(len(df), dtype=float)

    # Normalize time (handle NaN)
    if 'elapsed_time' in df.columns and not df['elapsed_time'].isna().all():
        min_time = df['elapsed_time'].min(skipna=True)
        df['time'] = df['elapsed_time'] - min_time
    elif 'timestamp' in df.columns and not df['timestamp'].isna().all():
        df['time'] = df['timestamp'] - df['timestamp'].min(skipna=True)
    else:
        df['time'] = np.arange(len(df), dtype=float) * 0.01  # Assume 100Hz

    # Convert speed to km/h (m/s * 3.6)
    if 'speed' in df.columns:
        df['speed_kmh'] = df['speed'] * 3.6

    return df


def detect_available_columns(df: pd.DataFrame) -> dict[str, list[str]]:
    """Detect which data columns are available and valid."""

    categories = {
        'gps': ['latitude', 'longitude', 'altitude', 'bearing', 'accuracy'],
        'motion': ['speed', 'speed_kmh', 'distance', 'time'],
        'acceleration': ['lateral_acc', 'longitudinal_acc', 'combined_acc'],
        'gyro': ['lean_angle', 'x_rate_of_rotation', 'y_rate_of_rotation', 'z_rate_of_rotation'],
        'obd_throttle': ['throttle_pos', 'accelerator_pos'],
        'obd_engine': ['rpm', 'engine_load'],
        'obd_other': ['brake_pos', 'gear', 'coolant_temp', 'intake_temp', 'boost'],
    }

    available = {}
    for category, cols in categories.items():
        valid_cols = []
        for col in cols:
            if col in df.columns and not df[col].isna().all():
                valid_cols.append(col)
        if valid_cols:
            available[category] = valid_cols

    return available


def load_lap(filepath: str, target_lap: int | None = None) -> LapData:
    """
    Load and process a single lap file.

    Args:
        filepath: Path to RaceChrono CSV file
        target_lap: Specific lap number to extract. If None, auto-detect.
    """
    path = Path(filepath)

    # Parse metadata
    metadata = parse_metadata(filepath)

    # Parse data (with lap filtering)
    df, lap_number = parse_racechrono_csv(filepath, target_lap)
    df = normalize_lap_data(df)

    # Detect available columns
    available = detect_available_columns(df)
    all_cols = [col for cols in available.values() for col in cols]

    # Validate DataFrame is not empty
    if df.empty:
        raise ValueError(f"No data found for lap {lap_number} in {filepath}")

    # Extract lap info with NaN handling
    lap_time = df['time'].max() - df['time'].min()
    lap_distance = df['distance'].max()

    # Handle potential NaN values
    if pd.isna(lap_time):
        lap_time = 0.0
    if pd.isna(lap_distance):
        lap_distance = 0.0

    # Compute statistics with NaN-safe operations
    def safe_max(series):
        """Return max value or 0 if all NaN."""
        val = series.max()
        return 0.0 if pd.isna(val) else float(val)

    def safe_min(series):
        """Return min value or 0 if all NaN."""
        val = series.min()
        return 0.0 if pd.isna(val) else float(val)

    def safe_mean(series):
        """Return mean value or 0 if all NaN."""
        val = series.mean()
        return 0.0 if pd.isna(val) else float(val)

    max_speed = safe_max(df['speed_kmh']) if 'speed_kmh' in df.columns else 0.0
    avg_speed = safe_mean(df['speed_kmh']) if 'speed_kmh' in df.columns else 0.0
    max_lat_g = safe_max(df['lateral_acc'].abs()) if 'lateral_acc' in df.columns else 0.0
    max_brake_g = abs(safe_min(df['longitudinal_acc'])) if 'longitudinal_acc' in df.columns else 0.0
    max_accel_g = safe_max(df['longitudinal_acc']) if 'longitudinal_acc' in df.columns else 0.0

    return LapData(
        name=path.stem,
        metadata=metadata,
        df=df,
        lap_number=lap_number,
        lap_time=lap_time,
        lap_distance=lap_distance,
        max_speed_kmh=max_speed,
        avg_speed_kmh=avg_speed,
        max_lateral_g=max_lat_g,
        max_brake_g=max_brake_g,
        max_accel_g=max_accel_g,
        available_columns=all_cols,
        has_obd_data='obd_throttle' in available or 'obd_engine' in available,
        has_throttle='obd_throttle' in available,
        has_brake='brake_pos' in all_cols,
        has_rpm='rpm' in all_cols,
    )


def format_lap_time(seconds: float) -> str:
    """Format lap time as mm:ss.sss"""
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes}:{secs:06.3f}"


def generate_friendly_name(filename: str, lap_number: int, metadata: SessionMetadata) -> str:
    """
    Generate a user-friendly display name from filename/metadata.

    Examples:
        session_20251231_111959_tianma_lap13_v3 → "12/31 L13"
        session_20251222_102302_tianma_lap14_v3 → "12/22 L14"
    """
    import re

    # Try to extract date from filename: session_YYYYMMDD_HHMMSS_...
    date_match = re.search(r'session_(\d{4})(\d{2})(\d{2})_', filename)
    if date_match:
        month = int(date_match.group(2))
        day = int(date_match.group(3))
        date_str = f"{month}/{day}"
    elif metadata.created_date:
        parts = metadata.created_date.split()
        date_str = parts[0].replace('-', '/')[-5:] if parts else ""
    else:
        date_str = ""

    # Lap number
    lap_str = f"L{lap_number}" if lap_number > 0 else ""

    # Combine
    parts = [p for p in [date_str, lap_str] if p]
    return " ".join(parts) if parts else filename[:20]
