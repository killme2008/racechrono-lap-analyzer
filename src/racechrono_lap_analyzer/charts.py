"""
Plotly chart generation for lap comparison.

Creates interactive charts for:
- Speed comparison with delta overlay
- Acceleration (lateral/longitudinal)
- Throttle/Brake position
- Track map with speed coloring
- G-G diagram (friction circle)
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from racechrono_lap_analyzer.analysis import align_laps, compute_time_delta
from racechrono_lap_analyzer.data_parser import LapData
from racechrono_lap_analyzer.i18n import t

# Color palette for multiple laps
LAP_COLORS = [
    '#1f77b4',  # Blue
    '#ff7f0e',  # Orange
    '#2ca02c',  # Green
    '#d62728',  # Red
    '#9467bd',  # Purple
    '#8c564b',  # Brown
]


def hex_to_rgba(color: str, alpha: float = 0.2) -> str:
    """Convert hex color to rgba string."""
    if color.startswith('#'):
        hex_color = color.lstrip('#')
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        return f'rgba({r}, {g}, {b}, {alpha})'
    elif color.startswith('rgb('):
        return color.replace(')', f', {alpha})').replace('rgb(', 'rgba(')
    return color


def create_speed_comparison_chart(
    laps: list[LapData],
    show_delta: bool = True
) -> go.Figure:
    """
    Create speed vs distance comparison chart.

    Optionally shows speed delta between first two laps as a filled area.
    """
    distances, aligned = align_laps(laps)

    # Convert to lists for Plotly/Streamlit compatibility
    distances = list(distances)

    fig = make_subplots(
        rows=2 if show_delta and len(laps) >= 2 else 1,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        row_heights=[0.65, 0.35] if show_delta and len(laps) >= 2 else [1.0],
        subplot_titles=[t('charts.speed.title'), t('charts.speed.delta_title')] if show_delta and len(laps) >= 2 else [t('charts.speed.title')]
    )

    # Speed traces
    for i, (lap, data) in enumerate(zip(laps, aligned, strict=False)):
        color = LAP_COLORS[i % len(LAP_COLORS)]
        fig.add_trace(
            go.Scatter(
                x=list(data['distance']),
                y=list(data['speed_kmh']),
                mode='lines',
                name=f"{lap.name} ({lap.lap_time:.3f}s)",
                line={'color': color, 'width': 1.5},
                hovertemplate='%{x:.0f}m: %{y:.1f} km/h<extra></extra>'
            ),
            row=1, col=1
        )

    # Speed delta (if 2+ laps)
    if show_delta and len(laps) >= 2:
        ref_speed = aligned[0]['speed_kmh'].values
        cmp_speed = aligned[1]['speed_kmh'].values
        delta = cmp_speed - ref_speed

        # Positive delta (lap 2 faster)
        fig.add_trace(
            go.Scatter(
                x=distances,
                y=list(np.where(delta > 0, delta, 0)),
                fill='tozeroy',
                mode='none',
                name=t('charts.common.faster_label', name=laps[1].name),
                fillcolor='rgba(0, 200, 0, 0.3)',
                hoverinfo='skip'
            ),
            row=2, col=1
        )

        # Negative delta (lap 1 faster)
        fig.add_trace(
            go.Scatter(
                x=distances,
                y=list(np.where(delta < 0, delta, 0)),
                fill='tozeroy',
                mode='none',
                name=t('charts.common.faster_label', name=laps[0].name),
                fillcolor='rgba(200, 0, 0, 0.3)',
                hoverinfo='skip'
            ),
            row=2, col=1
        )

        # Delta line
        fig.add_trace(
            go.Scatter(
                x=distances,
                y=list(delta),
                mode='lines',
                name=t('charts.common.delta'),
                line={'color': 'black', 'width': 1},
                hovertemplate='%{x:.0f}m: %{y:+.1f} km/h<extra></extra>'
            ),
            row=2, col=1
        )

        abs_delta = np.abs(delta)
        abs_delta = abs_delta[np.isfinite(abs_delta)]
        if abs_delta.size > 0:
            limit = np.nanpercentile(abs_delta, 99)
            limit = 10 if not np.isfinite(limit) or limit <= 0 else max(10, limit * 1.2)
        else:
            limit = 10
        fig.update_yaxes(range=[-limit, limit], row=2, col=1)

    # Add rangeslider for zoom capability
    fig.update_xaxes(
        title_text=t('charts.common.distance'),
        row=2 if show_delta and len(laps) >= 2 else 1,
        rangeslider={'visible': True, 'thickness': 0.05},
        showspikes=True,
        spikemode='across',
        spikesnap='cursor',
        spikecolor='gray',
        spikethickness=1
    )

    fig.update_layout(
        height=650,
        hovermode='x unified',
        legend={'orientation': 'h', 'yanchor': 'bottom', 'y': 1.08, 'x': 0.5, 'xanchor': 'center'},
        margin={'l': 50, 'r': 50, 't': 100, 'b': 50},
        template='plotly_white'
    )

    # Move subplot titles down to avoid legend overlap
    for annotation in fig['layout']['annotations']:
        if annotation['text'] == t('charts.speed.title'):
            annotation['y'] = 0.95  # Move speed title down
        elif annotation['text'] == t('charts.speed.delta_title'):
            annotation['y'] = 0.32  # Adjust delta title position

    return fig


def create_acceleration_chart(laps: list[LapData]) -> go.Figure:
    """
    Create lateral and longitudinal acceleration comparison chart.
    """
    distances, aligned = align_laps(laps)

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=[t('charts.acceleration.lateral_title'), t('charts.acceleration.longitudinal_title')]
    )

    for i, (lap, data) in enumerate(zip(laps, aligned, strict=False)):
        color = LAP_COLORS[i % len(LAP_COLORS)]

        # Lateral acceleration
        fig.add_trace(
            go.Scatter(
                x=list(data['distance']),
                y=list(data['lateral_acc']),
                mode='lines',
                name=lap.name,
                line={'color': color, 'width': 1},
                legendgroup=lap.name,
                hovertemplate='%{x:.0f}m: %{y:.2f}G<extra></extra>'
            ),
            row=1, col=1
        )

        # Longitudinal acceleration
        fig.add_trace(
            go.Scatter(
                x=list(data['distance']),
                y=list(data['longitudinal_acc']),
                mode='lines',
                name=lap.name,
                line={'color': color, 'width': 1},
                legendgroup=lap.name,
                showlegend=False,
                hovertemplate='%{x:.0f}m: %{y:.2f}G<extra></extra>'
            ),
            row=2, col=1
        )

    # Add zero lines
    for row in [1, 2]:
        fig.add_hline(y=0, line_dash='dash', line_color='gray', opacity=0.5, row=row, col=1)

    fig.update_xaxes(
        title_text=t('charts.common.distance'),
        row=2,
        showspikes=True,
        spikemode='across',
        spikesnap='cursor',
        spikecolor='gray',
        spikethickness=1
    )
    fig.update_yaxes(title_text='G', row=1)
    fig.update_yaxes(title_text='G', row=2)

    fig.update_layout(
        height=450,
        hovermode='x unified',
        legend={'orientation': 'h', 'yanchor': 'bottom', 'y': 1.02},
        margin={'l': 50, 'r': 50, 't': 60, 'b': 50},
        template='plotly_white'
    )

    return fig


def create_throttle_brake_chart(laps: list[LapData]) -> go.Figure | None:
    """
    Create throttle and brake position chart (if OBD data available).
    """
    # Check if any lap has throttle data
    has_data = any(lap.has_throttle for lap in laps)
    if not has_data:
        return None

    distances, aligned = align_laps(laps)

    fig = go.Figure()

    throttle_col = None
    throttle_label = t('charts.throttle_brake.throttle')
    for col, label in [('throttle_pos', t('charts.throttle_brake.throttle')), ('accelerator_pos', t('charts.throttle_brake.pedal'))]:
        if any(col in data.columns and not np.isnan(data[col]).all() for data in aligned):
            throttle_col = col
            throttle_label = label
            break

    if throttle_col is None:
        return None

    for i, (lap, data) in enumerate(zip(laps, aligned, strict=False)):
        color = LAP_COLORS[i % len(LAP_COLORS)]

        if throttle_col in data.columns and not np.isnan(data[throttle_col]).all():
            fig.add_trace(
                go.Scatter(
                    x=list(data['distance']),
                    y=list(data[throttle_col]),
                    mode='lines',
                    name=f'{lap.name} {throttle_label}',
                    line={'color': color, 'width': 1.5},
                    hovertemplate='%{x:.0f}m: %{y:.1f}%<extra></extra>'
                )
            )

        # Brake position (if available)
        if 'brake_pos' in data.columns and not np.isnan(data['brake_pos']).all():
            fig.add_trace(
                go.Scatter(
                    x=list(data['distance']),
                    y=list(data['brake_pos']),
                    mode='lines',
                    name=f'{lap.name} {t("charts.throttle_brake.brake")}',
                    line={'color': color, 'width': 1.5, 'dash': 'dot'},
                    hovertemplate='%{x:.0f}m: %{y:.1f}%<extra></extra>'
                )
            )

    fig.update_layout(
        title=t('charts.throttle_brake.title'),
        xaxis_title=t('charts.common.distance'),
        yaxis_title=t('charts.throttle_brake.y_label'),
        yaxis_range=[0, 105],
        height=350,
        hovermode='x unified',
        legend={'orientation': 'h', 'yanchor': 'bottom', 'y': 1.02},
        margin={'l': 50, 'r': 50, 't': 60, 'b': 50},
        template='plotly_white'
    )

    return fig


def create_rpm_chart(laps: list[LapData]) -> go.Figure | None:
    """
    Create RPM chart (if OBD data available).
    """
    has_rpm = any(lap.has_rpm for lap in laps)
    if not has_rpm:
        return None

    distances, aligned = align_laps(laps)

    fig = go.Figure()

    for i, (lap, data) in enumerate(zip(laps, aligned, strict=False)):
        color = LAP_COLORS[i % len(LAP_COLORS)]

        if 'rpm' in data.columns and not np.isnan(data['rpm']).all():
            fig.add_trace(
                go.Scatter(
                    x=list(data['distance']),
                    y=list(data['rpm']),
                    mode='lines',
                    name=lap.name,
                    line={'color': color, 'width': 1.5},
                    hovertemplate='%{x:.0f}m: %{y:.0f} RPM<extra></extra>'
                )
            )

    fig.update_layout(
        title=t('charts.rpm.title'),
        xaxis_title=t('charts.common.distance'),
        yaxis_title=t('charts.rpm.y_label'),
        height=300,
        hovermode='x unified',
        legend={'orientation': 'h', 'yanchor': 'bottom', 'y': 1.02},
        margin={'l': 50, 'r': 50, 't': 60, 'b': 50},
        template='plotly_white'
    )

    return fig


def create_track_map(laps: list[LapData]) -> go.Figure:
    """
    Create GPS track map with speed coloring.
    """
    fig = go.Figure()

    # Find global speed range for consistent coloring
    all_speeds = []
    for lap in laps:
        if 'speed_kmh' in lap.df.columns:
            all_speeds.extend(lap.df['speed_kmh'].dropna().tolist())

    speed_min = min(all_speeds) if all_speeds else 0
    speed_max = max(all_speeds) if all_speeds else 100

    trace_index = 0
    center_lats = []
    center_lons = []

    for lap in laps:
        df = lap.df

        if 'latitude' not in df.columns or 'longitude' not in df.columns:
            continue

        coords = df[['latitude', 'longitude']].dropna()
        if coords.empty:
            continue

        lat = list(coords['latitude'].values)
        lon = list(coords['longitude'].values)
        if 'speed_kmh' in df.columns:
            speed = df.loc[coords.index, 'speed_kmh'].fillna(0).tolist()
        else:
            speed = [0] * len(lat)

        # Track line with speed coloring
        fig.add_trace(
            go.Scattermapbox(
                lat=lat,
                lon=lon,
                mode='markers',
                marker={
                    'size': 4,
                    'color': speed,
                    'colorscale': 'RdYlGn',
                    'cmin': speed_min,
                    'cmax': speed_max,
                    'showscale': trace_index == 0,
                    'colorbar': {'title': 'km/h'} if trace_index == 0 else None,
                },
                name=lap.name,
                hovertemplate=f'{lap.name}<br>%{{customdata:.1f}} km/h<extra></extra>',
                customdata=speed
            )
        )

        # Start/Finish markers
        fig.add_trace(
            go.Scattermapbox(
                lat=[lat[0]],
                lon=[lon[0]],
                mode='markers',
                marker={'size': 12, 'color': 'green', 'symbol': 'circle'},
                name=t('charts.track_map.start') if trace_index == 0 else '',
                showlegend=trace_index == 0,
                hoverinfo='skip'
            )
        )

        center_lats.append(coords['latitude'].mean())
        center_lons.append(coords['longitude'].mean())
        trace_index += 1

    # Center map
    if center_lats and center_lons:
        center_lat = float(np.nanmean(center_lats))
        center_lon = float(np.nanmean(center_lons))
        zoom = 14
    else:
        center_lat = 0
        center_lon = 0
        zoom = 1
        fig.add_annotation(
            x=0.5, y=0.5,
            xref='paper', yref='paper',
            text=t('charts.track_map.no_gps'),
            showarrow=False,
            font={'size': 12, 'color': 'gray'}
        )

    fig.update_layout(
        mapbox={
            'style': 'open-street-map',
            'center': {'lat': center_lat, 'lon': center_lon},
            'zoom': zoom
        },
        height=500,
        margin={'l': 0, 'r': 0, 't': 30, 'b': 0},
        title=t('charts.track_map.title')
    )

    return fig


def create_track_line_comparison(
    laps: list[LapData],
    mode: str = "line_comparison",
    highlight_deviations: bool = True,
    deviation_threshold_m: float = 1.0
) -> go.Figure:
    """
    Create track map with line comparison or speed coloring mode.

    Args:
        laps: List of LapData objects to compare
        mode: "line_comparison" (distinct colors per lap) or "speed_coloring" (existing behavior)
        highlight_deviations: Whether to highlight deviation zones in line_comparison mode
        deviation_threshold_m: Minimum offset to consider as deviation (default 1m)

    Returns:
        Plotly Figure with track visualization
    """
    from racechrono_lap_analyzer.analysis import (
        compute_line_deviations,
        detect_deviation_zones,
    )

    # For speed_coloring mode, use existing function
    if mode == "speed_coloring":
        return create_track_map(laps)

    # Line comparison mode
    fig = go.Figure()

    center_lats = []
    center_lons = []
    has_gps_data = False

    # First pass: draw base lines for each lap
    for i, lap in enumerate(laps):
        df = lap.df

        if 'latitude' not in df.columns or 'longitude' not in df.columns:
            continue

        coords = df[['latitude', 'longitude']].dropna()
        if coords.empty:
            continue

        has_gps_data = True
        lat = list(coords['latitude'].values)
        lon = list(coords['longitude'].values)
        color = LAP_COLORS[i % len(LAP_COLORS)]

        # Track line
        fig.add_trace(
            go.Scattermapbox(
                lat=lat,
                lon=lon,
                mode='lines',
                line={'width': 3, 'color': color},
                name=lap.name,
                hovertemplate=f'{lap.name}<extra></extra>',
            )
        )

        # Start marker
        fig.add_trace(
            go.Scattermapbox(
                lat=[lat[0]],
                lon=[lon[0]],
                mode='markers',
                marker={'size': 10, 'color': 'green', 'symbol': 'circle'},
                name=t('charts.track_map.start') if i == 0 else '',
                showlegend=i == 0,
                hoverinfo='skip'
            )
        )

        center_lats.append(coords['latitude'].mean())
        center_lons.append(coords['longitude'].mean())

    # Second pass: highlight deviation zones (if enabled and 2+ laps)
    if highlight_deviations and len(laps) >= 2 and has_gps_data:
        ref_lap = laps[0]

        for i, cmp_lap in enumerate(laps[1:], start=1):
            # Compute deviations
            offsets = compute_line_deviations(ref_lap, cmp_lap)

            if offsets.empty:
                continue

            # Detect deviation zones
            zones = detect_deviation_zones(
                offsets,
                threshold_m=deviation_threshold_m,
                min_length_m=10.0,
                merge_gap_m=20.0,
                max_zones=10
            )

            if not zones:
                continue

            # Get aligned data for GPS coordinates
            from racechrono_lap_analyzer.analysis import align_laps
            distances, aligned = align_laps([ref_lap, cmp_lap], resolution_m=1.0)

            if len(aligned) < 2:
                continue

            cmp_df = aligned[1]
            color = LAP_COLORS[i % len(LAP_COLORS)]

            # Highlight each zone with thicker line
            for zone in zones:
                # Find indices for this zone
                zone_mask = (distances >= zone.start_m) & (distances <= zone.end_m)
                zone_indices = np.where(zone_mask)[0]

                if len(zone_indices) == 0:
                    continue

                zone_lats = cmp_df.iloc[zone_indices]['latitude'].dropna().tolist()
                zone_lons = cmp_df.iloc[zone_indices]['longitude'].dropna().tolist()

                if not zone_lats or not zone_lons:
                    continue

                # Draw highlighted segment
                fig.add_trace(
                    go.Scattermapbox(
                        lat=zone_lats,
                        lon=zone_lons,
                        mode='lines',
                        line={'width': 8, 'color': hex_to_rgba(color, 0.6)},
                        name=f"{t('charts.track_map.deviation_zone')} ({zone.max_offset_m:.1f}m)",
                        showlegend=False,
                        hovertemplate=(
                            f"{cmp_lap.name}<br>"
                            f"{t('charts.track_map.offset_tooltip').format(offset=zone.avg_offset_m)}"
                            f"<extra></extra>"
                        ),
                    )
                )

    # Center map
    if center_lats and center_lons:
        center_lat = float(np.nanmean(center_lats))
        center_lon = float(np.nanmean(center_lons))
        zoom = 14
    else:
        center_lat = 0
        center_lon = 0
        zoom = 1
        fig.add_annotation(
            x=0.5, y=0.5,
            xref='paper', yref='paper',
            text=t('charts.track_map.no_gps'),
            showarrow=False,
            font={'size': 12, 'color': 'gray'}
        )

    title = t('charts.track_map.line_comparison_title') if mode == "line_comparison" else t('charts.track_map.title')

    fig.update_layout(
        mapbox={
            'style': 'open-street-map',
            'center': {'lat': center_lat, 'lon': center_lon},
            'zoom': zoom
        },
        height=500,
        margin={'l': 0, 'r': 0, 't': 30, 'b': 0},
        title=title
    )

    return fig


def create_gg_diagram(laps: list[LapData]) -> go.Figure:
    """
    Create G-G diagram (friction circle) for each lap.
    """
    n_laps = len(laps)
    fig = make_subplots(
        rows=1, cols=n_laps,
        subplot_titles=[lap.name for lap in laps]
    )

    for i, lap in enumerate(laps, 1):
        df = lap.df

        if 'lateral_acc' not in df.columns or 'longitudinal_acc' not in df.columns:
            continue

        lat_acc = list(df['lateral_acc'].values)
        lon_acc = list(df['longitudinal_acc'].values)
        speed = list(df['speed_kmh'].values) if 'speed_kmh' in df.columns else [0] * len(lat_acc)

        # G-G scatter
        fig.add_trace(
            go.Scatter(
                x=lat_acc,
                y=lon_acc,
                mode='markers',
                marker={
                    'size': 3,
                    'color': speed,
                    'colorscale': 'RdYlGn',
                    'showscale': i == n_laps,
                    'colorbar': {'title': 'km/h', 'x': 1.02} if i == n_laps else None,
                    'opacity': 0.5
                },
                name=lap.name,
                hovertemplate='Lat: %{x:.2f}G<br>Lon: %{y:.2f}G<extra></extra>'
            ),
            row=1, col=i
        )

        # Reference circles
        for r in [0.5, 1.0, 1.5]:
            theta = np.linspace(0, 2 * np.pi, 100)
            fig.add_trace(
                go.Scatter(
                    x=list(r * np.cos(theta)),
                    y=list(r * np.sin(theta)),
                    mode='lines',
                    line={'color': 'gray', 'width': 1, 'dash': 'dash'},
                    showlegend=False,
                    hoverinfo='skip'
                ),
                row=1, col=i
            )

        # Axis lines
        fig.add_hline(y=0, line_color='black', line_width=0.5, row=1, col=i)
        fig.add_vline(x=0, line_color='black', line_width=0.5, row=1, col=i)

        # Set fixed range for both axes with equal scaling
        axis_suffix = '' if i == 1 else str(i)
        fig.update_xaxes(
            range=[-2, 2],
            title_text=t('charts.gg_diagram.lateral_g'),
            row=1,
            col=i,
            constrain='domain',
            scaleanchor=f'y{axis_suffix}',
            scaleratio=1
        )
        fig.update_yaxes(
            range=[-2, 2],
            title_text=t('charts.gg_diagram.longitudinal_g') if i == 1 else '',
            row=1,
            col=i,
            constrain='domain'
        )

    fig.update_layout(
        height=450,
        title=t('charts.gg_diagram.title'),
        showlegend=False,
        margin={'l': 50, 'r': 80, 't': 60, 'b': 50},
        template='plotly_white'
    )

    return fig


def create_time_delta_chart(laps: list[LapData]) -> go.Figure | None:
    """
    Create cumulative time delta chart.

    Shows how much time is gained/lost relative to the fastest lap.
    """
    if len(laps) < 2:
        return None

    # Sort by lap time (fastest first)
    sorted_laps = sorted(laps, key=lambda x: x.lap_time)
    ref_lap = sorted_laps[0]

    fig = go.Figure()

    for i, lap in enumerate(sorted_laps[1:], 1):
        distances, time_delta = compute_time_delta(ref_lap, lap)
        color = LAP_COLORS[i % len(LAP_COLORS)]

        fig.add_trace(
            go.Scatter(
                x=list(distances),
                y=list(time_delta),
                mode='lines',
                name=f'{lap.name} vs {ref_lap.name}',
                line={'color': color, 'width': 2},
                fill='tozeroy',
                fillcolor=hex_to_rgba(color, 0.2),
                hovertemplate='%{x:.0f}m: %{y:+.3f}s<extra></extra>'
            )
        )

    fig.add_hline(y=0, line_dash='dash', line_color='gray', opacity=0.5)

    # Add annotation for interpretation
    fig.add_annotation(
        x=0.02, y=0.98,
        xref='paper', yref='paper',
        text=f'↑ {t("charts.time_delta.slower")}',
        showarrow=False,
        font={'size': 10, 'color': 'gray'},
        align='left'
    )
    fig.add_annotation(
        x=0.02, y=0.02,
        xref='paper', yref='paper',
        text=f'↓ {t("charts.time_delta.faster")}',
        showarrow=False,
        font={'size': 10, 'color': 'gray'},
        align='left'
    )

    fig.update_layout(
        title=t('charts.time_delta.title', ref_name=ref_lap.name),
        xaxis_title=t('charts.common.distance'),
        yaxis_title=t('charts.time_delta.y_label'),
        height=280,
        hovermode='x unified',
        legend={'orientation': 'h', 'yanchor': 'bottom', 'y': 1.02},
        margin={'l': 50, 'r': 50, 't': 60, 'b': 50},
        template='plotly_white'
    )

    return fig
