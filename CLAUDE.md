# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RaceChrono Lap Analyzer - A Streamlit web application for analyzing and comparing lap telemetry data exported from RaceChrono Pro (CSV3 format). Used for motorsport performance analysis.

## Commands

```bash
# Install dependencies
uv sync

# Install with dev dependencies
uv sync --extra dev

# Run the application
uv run streamlit run src/racechrono_lap_analyzer/app.py

# Run tests
uv run pytest

# Run single test
uv run pytest tests/test_data_parser.py::LoadLapTests::test_load_lap_computes_statistics -v

# Lint
uv run ruff check .

# Type check
uv run mypy .
```

## Architecture

```
src/racechrono_lap_analyzer/
├── app.py           # Streamlit UI, orchestrates data loading and display
├── data_parser.py   # RaceChrono CSV3 parsing, metadata extraction, lap normalization
├── analysis.py      # Lap alignment, corner detection, bottleneck analysis, coach insights
├── charts.py        # Plotly chart generation (speed, G-force, track map, etc.)
└── tracks/          # Track configuration JSON files (corner definitions)
```

### Key Data Structures

- `LapData` (data_parser.py): Container for single lap with DataFrame, metadata, and computed stats
- `SessionMetadata` (data_parser.py): RaceChrono session header info
- `Corner`, `BottleneckSection`, `CoachInsight` (analysis.py): Analysis results

### Data Flow

1. User uploads RaceChrono CSV files via Streamlit sidebar
2. `load_lap()` parses CSV, extracts metadata, normalizes distance/time to start from 0
3. `align_laps()` resamples all laps to common distance points (1m resolution) using interpolation
4. Charts compare aligned data; analysis functions detect corners and generate insights

### Track Configuration

Track configs in `tracks/*.json` define corner positions with apex distances, start/end markers, and typical speeds. Used by `corners_from_track_config()` for accurate corner-by-corner analysis.

## Key Implementation Details

- RaceChrono CSV3: header rows (1-9), column names (10), units (11), sources (12), then data
- Speed stored as m/s, converted to km/h via `* 3.6`
- Lateral acceleration: positive = right turn, negative = left turn
- Longitudinal acceleration: positive = accelerating, negative = braking
- Time delta: compares cumulative time at each distance point between laps

## Testing

Tests in `tests/` directory use real RaceChrono CSV data from `tests/data/`:
- `test_data_parser.py`: CSV parsing, metadata extraction, lap loading
- `test_analysis.py`: Corner detection, bottleneck analysis, coach insights
- `test_charts.py`: Chart generation edge cases
