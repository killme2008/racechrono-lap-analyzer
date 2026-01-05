"""
RaceChrono Lap Comparison Tool - Streamlit App

A visual tool for comparing lap data from RaceChrono Pro.
Supports multiple lap files with speed, acceleration, and OBD data comparison.
"""

import tempfile
from itertools import groupby
from pathlib import Path

import streamlit as st

from racechrono_lap_analyzer.analysis import (
    analyze_gg_diagram,
    analyze_racing_technique,
    compute_tire_utilization,
    corners_from_track_config,
    detect_corners,
    detect_track,
    find_bottlenecks,
    generate_coach_insights,
    get_available_tracks,
    load_track_config,
)
from racechrono_lap_analyzer.charts import (
    create_acceleration_chart,
    create_gg_diagram,
    create_rpm_chart,
    create_speed_comparison_chart,
    create_throttle_brake_chart,
    create_time_delta_chart,
    create_track_line_comparison,
)
from racechrono_lap_analyzer.data_parser import (
    LapData,
    format_lap_time,
    generate_friendly_name,
    load_lap,
)
from racechrono_lap_analyzer.i18n import (
    get_available_languages,
    get_language,
    set_language,
    t,
)
from racechrono_lap_analyzer.rules import RuleConfig

# Page config
st.set_page_config(
    page_title="RaceChrono Lap Comparison",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Minimal custom CSS
st.markdown("""
<style>
    /* Reduce padding in metric containers for compact display */
    [data-testid="stMetricValue"] {
        font-size: 1.5rem;
    }
    [data-testid="stMetricDelta"] {
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)


def _init_language() -> None:
    """Initialize language from session state or default."""
    if "language" not in st.session_state:
        st.session_state.language = "en"
    set_language(st.session_state.language)


def _init_rule_config() -> RuleConfig:
    """Initialize rule config from session state."""
    if "rule_config" not in st.session_state:
        st.session_state.rule_config = RuleConfig()
    return st.session_state.rule_config


def main():
    _init_language()
    rule_config = _init_rule_config()

    st.title(f"🏎️ {t('app.title')}")
    st.markdown(t("app.subtitle"))

    # Sidebar - Language and Settings
    with st.sidebar:
        # Language selector at top
        languages = get_available_languages()
        lang_options = list(languages.keys())

        current_idx = lang_options.index(get_language()) if get_language() in lang_options else 0
        selected_lang = st.selectbox(
            t("app.sidebar.language"),
            options=lang_options,
            format_func=lambda x: languages[x],
            index=current_idx,
            key="lang_selector"
        )

        if selected_lang != st.session_state.language:
            st.session_state.language = selected_lang
            set_language(selected_lang)
            st.rerun()

        st.divider()

        st.header(f"📁 {t('app.sidebar.data_files')}")

        uploaded_files = st.file_uploader(
            t("app.sidebar.upload_label"),
            type=['csv'],
            accept_multiple_files=True,
            help=t("app.sidebar.upload_help")
        )

        if uploaded_files:
            st.success(t("app.sidebar.files_uploaded", count=len(uploaded_files)))

        st.divider()

        st.header(f"⚙️ {t('app.sidebar.settings')}")

        # Track selection
        available_tracks = get_available_tracks()
        track_options = [t("app.sidebar.auto_detect")] + available_tracks
        selected_track = st.selectbox(
            t("app.sidebar.track_config"),
            track_options,
            help=t("app.sidebar.track_config_help")
        )

        show_delta = st.checkbox(t("app.sidebar.show_speed_delta"), value=True)
        show_time_delta = st.checkbox(t("app.sidebar.show_time_delta"), value=True)
        show_track_map = st.checkbox(t("app.sidebar.show_track_map"), value=True)
        show_gg = st.checkbox(t("app.sidebar.show_gg_diagram"), value=True)

        st.divider()

        # Advanced rule settings
        with st.expander(t("settings.advanced_rules")):
            sensitivity = st.slider(
                t("settings.sensitivity"),
                min_value=0.5,
                max_value=2.0,
                value=rule_config.sensitivity,
                step=0.1,
                help=t("settings.sensitivity_help")
            )

            brake_threshold = st.number_input(
                t("settings.brake_threshold"),
                min_value=1.0,
                max_value=30.0,
                value=rule_config.brake_early_m or 10.0
            )

            apex_threshold = st.number_input(
                t("settings.apex_threshold"),
                min_value=1.0,
                max_value=20.0,
                value=rule_config.apex_speed_low_kmh or 5.0
            )

            exit_threshold = st.number_input(
                t("settings.exit_threshold"),
                min_value=1.0,
                max_value=20.0,
                value=rule_config.exit_speed_low_kmh or 8.0
            )

            if st.button(t("settings.reset_defaults")):
                st.session_state.rule_config = RuleConfig()
                st.rerun()

            # Update rule config
            rule_config.sensitivity = sensitivity
            rule_config.brake_early_m = brake_threshold
            rule_config.apex_speed_low_kmh = apex_threshold
            rule_config.exit_speed_low_kmh = exit_threshold
            st.session_state.rule_config = rule_config

        st.divider()

        st.markdown(f"""
        **{t('app.tips.header')}**
        - {t('app.tips.upload_multiple')}
        - {t('app.tips.fastest_reference')}
        - {t('app.tips.color_meaning')}
        """)

    # Main content
    if not uploaded_files:
        st.info(f"👆 {t('app.upload_prompt')}")

        with st.expander(f"📖 {t('app.export_guide.title')}"):
            st.markdown(f"""
            {t('app.export_guide.step1')}
            {t('app.export_guide.step2')}
            {t('app.export_guide.step3')}
            {t('app.export_guide.step4')}
            {t('app.export_guide.step5')}
            {t('app.export_guide.step6')}
            """)
        return

    # Load lap data
    laps: list[LapData] = []
    load_errors = []

    for uploaded_file in uploaded_files:
        try:
            # Save to temp file for parsing (write as binary, then close before reading)
            with tempfile.NamedTemporaryFile(delete=False, suffix='.csv', mode='wb') as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name

            lap = load_lap(tmp_path)
            # Generate friendly name for display
            original_name = uploaded_file.name.replace('.csv', '')
            lap.name = generate_friendly_name(original_name, lap.lap_number, lap.metadata)
            lap.original_name = original_name  # Keep original for metadata display
            laps.append(lap)

            # Cleanup
            Path(tmp_path).unlink()

        except Exception as e:
            import traceback
            load_errors.append(f"{uploaded_file.name}: {str(e)}\n{traceback.format_exc()}")

    if load_errors:
        for error in load_errors:
            st.error(t("app.errors.load_failed", error=error))

    if not laps:
        st.error(t("app.errors.no_valid_files"))
        return

    # Sort laps by time (fastest first)
    laps.sort(key=lambda x: x.lap_time)

    # === Lap Summary Cards ===
    st.header(f"📊 {t('summary.header')}")

    ref_lap = laps[0]  # Fastest lap as reference

    cols = st.columns(len(laps))
    for i, (col, lap) in enumerate(zip(cols, laps, strict=False)):
        with col:
            # Header with trophy for fastest lap
            if i == 0:
                st.markdown(f"### 🏆 {lap.name}")
                st.caption(t("summary.reference"))
            else:
                st.markdown(f"### {lap.name}")
                time_delta = lap.lap_time - ref_lap.lap_time
                st.caption(t("summary.vs_reference", delta=time_delta))

            # Lap time metric
            st.metric(
                label=t("summary.lap_time"),
                value=format_lap_time(lap.lap_time),
                delta=f"+{lap.lap_time - ref_lap.lap_time:.3f}s" if i > 0 else None,
                delta_color="inverse"
            )

            # Speed metrics in 2 columns
            speed_cols = st.columns(2)
            with speed_cols[0]:
                speed_delta = lap.max_speed_kmh - ref_lap.max_speed_kmh if i > 0 else None
                st.metric(
                    label=t("summary.max_speed"),
                    value=f"{lap.max_speed_kmh:.1f}",
                    delta=f"{speed_delta:+.1f}" if speed_delta else None,
                    help=t("summary.unit_kmh")
                )
            with speed_cols[1]:
                avg_delta = lap.avg_speed_kmh - ref_lap.avg_speed_kmh if i > 0 else None
                st.metric(
                    label=t("summary.avg_speed"),
                    value=f"{lap.avg_speed_kmh:.1f}",
                    delta=f"{avg_delta:+.1f}" if avg_delta else None,
                    help=t("summary.unit_kmh")
                )

            # G-force metrics in 2 columns
            g_cols = st.columns(2)
            with g_cols[0]:
                lat_delta = lap.max_lateral_g - ref_lap.max_lateral_g if i > 0 else None
                st.metric(
                    label=t("summary.lateral_g"),
                    value=f"{lap.max_lateral_g:.2f}",
                    delta=f"{lat_delta:+.2f}" if lat_delta else None,
                    help=t("summary.max_cornering")
                )
            with g_cols[1]:
                brake_delta = lap.max_brake_g - ref_lap.max_brake_g if i > 0 else None
                st.metric(
                    label=t("summary.brake_g"),
                    value=f"{lap.max_brake_g:.2f}",
                    delta=f"{brake_delta:+.2f}" if brake_delta else None,
                    help=t("summary.max_braking")
                )

    # === Metadata ===
    with st.expander(f"📋 {t('metadata.header')}"):
        meta_cols = st.columns(len(laps))
        for col, lap in zip(meta_cols, laps, strict=False):
            with col:
                original_name = getattr(lap, 'original_name', lap.name)
                st.markdown(f"**{lap.name}**")
                st.text(f"{t('metadata.file')}: {original_name}")
                st.text(f"{t('metadata.track')}: {lap.metadata.track_name}")
                st.text(f"{t('metadata.session')}: {lap.metadata.session_title}")
                st.text(f"{t('metadata.date')}: {lap.metadata.created_date}")
                st.text(f"{t('metadata.lap_number')}: {lap.lap_number}")
                st.text(f"{t('metadata.distance')}: {lap.lap_distance:.0f}m")

                # Available data
                obd_status = f"✅ {t('metadata.obd_available')}" if lap.has_obd_data else f"❌ {t('metadata.obd_not_available')}"
                st.text(obd_status)

    # === Tire Utilization Analysis (All Laps Comparison) ===
    with st.expander(f"📊 {t('statistics.header')}", expanded=True):
        # Compute stats for all laps
        all_tire_stats = [(lap, compute_tire_utilization(lap)) for lap in laps]

        def format_with_diff(value: float, ref_value: float | None, fmt: str = ".2f", suffix: str = "") -> str:
            """Format value with difference from reference."""
            if ref_value is None:
                return f"{value:{fmt}}{suffix}"
            diff = value - ref_value
            if abs(diff) < 0.01:
                return f"{value:{fmt}}{suffix}"
            sign = "+" if diff > 0 else ""
            return f"{value:{fmt}}{suffix} ({sign}{diff:{fmt}})"

        # Build comparison table - metrics as rows, laps as columns
        metrics = [
            ("avg_combined_g", t("statistics.avg_combined_g"), ".2f", " G"),
            ("max_combined_g", t("statistics.max_combined_g"), ".2f", " G"),
            ("high_g_percentage", t("statistics.high_g_percentage"), ".1f", "%"),
            ("trail_brake_percentage", t("statistics.trail_brake_percentage"), ".1f", "%"),
        ]

        # Check if any lap has throttle data
        has_throttle = any(s.full_throttle_percentage is not None for _, s in all_tire_stats)
        if has_throttle:
            metrics.append(("full_throttle_percentage", t("statistics.full_throttle_percentage"), ".1f", "%"))

        comparison_data = []
        ref_stats = all_tire_stats[0][1]  # Reference lap stats

        for attr, label, fmt, suffix in metrics:
            row = {t("statistics.metric"): label}
            for i, (lap, stats) in enumerate(all_tire_stats):
                value = getattr(stats, attr)
                ref_value = getattr(ref_stats, attr)
                if value is None:
                    row[lap.name] = "—"
                elif i == 0:  # Reference lap
                    row[lap.name] = f"{value:{fmt}}{suffix} ⭐"
                else:
                    row[lap.name] = format_with_diff(value, ref_value, fmt, suffix)
            comparison_data.append(row)

        st.dataframe(comparison_data, width="stretch", hide_index=True)
        st.caption(f"⭐ = {t('summary.reference')}")

    st.divider()

    # === Charts with Tabs ===
    st.header(f"📈 {t('charts.header')}")

    tab_overview, tab_gforce, tab_telemetry, tab_track = st.tabs([
        f"🏁 {t('charts.tabs.overview')}",
        f"📊 {t('charts.tabs.gforce')}",
        f"🔧 {t('charts.tabs.telemetry')}",
        f"🗺️ {t('charts.tabs.track')}"
    ])

    # Tab 1: Overview (Speed + Time Delta)
    with tab_overview:
        speed_chart = create_speed_comparison_chart(laps, show_delta=show_delta)
        st.plotly_chart(speed_chart, width="stretch")

        if show_time_delta and len(laps) >= 2:
            time_delta_chart = create_time_delta_chart(laps)
            if time_delta_chart:
                st.plotly_chart(time_delta_chart, width="stretch")

    # Tab 2: G-Force (Acceleration + G-G Diagram)
    with tab_gforce:
        accel_chart = create_acceleration_chart(laps)
        st.plotly_chart(accel_chart, width="stretch")

        if show_gg:
            st.markdown("---")
            gg_chart = create_gg_diagram(laps)
            st.plotly_chart(gg_chart, width="stretch")

            # GG Diagram Insights - Compare all laps
            all_gg_insights = [(lap, analyze_gg_diagram(lap)) for lap in laps]

            # Show comparison table
            with st.expander(f"🎯 {t('gg_analysis.header')}", expanded=False):
                ref_gg = all_gg_insights[0][1]

                def fmt_gg_diff(value: float, ref_value: float) -> str:
                    diff = value - ref_value
                    if abs(diff) < 0.01:
                        return f"{value:.2f}G"
                    sign = "+" if diff > 0 else ""
                    return f"{value:.2f}G ({sign}{diff:.2f})"

                def fmt_pct_diff(value: float, ref_value: float) -> str:
                    diff = value - ref_value
                    if abs(diff) < 0.1:
                        return f"{value:.1f}%"
                    sign = "+" if diff > 0 else ""
                    return f"{value:.1f}% ({sign}{diff:.1f})"

                # Build comparison data
                gg_metrics = [
                    ("left_turn_avg_g", t("gg_analysis.left_turn"), "g"),
                    ("right_turn_avg_g", t("gg_analysis.right_turn"), "g"),
                    ("brake_left_pct", t("gg_analysis.quadrants.brake_left"), "pct"),
                    ("brake_right_pct", t("gg_analysis.quadrants.brake_right"), "pct"),
                    ("accel_left_pct", t("gg_analysis.quadrants.accel_left"), "pct"),
                    ("accel_right_pct", t("gg_analysis.quadrants.accel_right"), "pct"),
                ]

                gg_comparison = []
                for attr, label, fmt_type in gg_metrics:
                    row = {t("statistics.metric"): label}
                    for i, (lap, gg) in enumerate(all_gg_insights):
                        value = getattr(gg, attr)
                        ref_value = getattr(ref_gg, attr)
                        if i == 0:
                            if fmt_type == "g":
                                row[lap.name] = f"{value:.2f}G ⭐"
                            else:
                                row[lap.name] = f"{value:.1f}% ⭐"
                        else:
                            if fmt_type == "g":
                                row[lap.name] = fmt_gg_diff(value, ref_value)
                            else:
                                row[lap.name] = fmt_pct_diff(value, ref_value)
                    gg_comparison.append(row)

                st.dataframe(gg_comparison, width="stretch", hide_index=True)
                st.caption(f"⭐ = {t('summary.reference')}")

                # Show insights for each lap
                for lap, gg in all_gg_insights:
                    if gg.insights:
                        st.markdown(f"**{lap.name}:**")
                        for insight in gg.insights:
                            st.markdown(f"- {insight}")

    # Tab 3: Telemetry (Throttle/Brake/RPM if available)
    with tab_telemetry:
        has_telemetry = False

        throttle_chart = create_throttle_brake_chart(laps)
        if throttle_chart:
            has_telemetry = True
            st.plotly_chart(throttle_chart, width="stretch")

        rpm_chart = create_rpm_chart(laps)
        if rpm_chart:
            has_telemetry = True
            st.plotly_chart(rpm_chart, width="stretch")

        if not has_telemetry:
            st.info(t("charts.throttle_brake.no_data") if "charts.throttle_brake.no_data" in t("charts.throttle_brake.no_data") else
                    "No OBD/telemetry data available. Connect OBD adapter to RaceChrono for throttle, brake, and RPM data.")

    # Tab 4: Track (Map + Corner Analysis)
    with tab_track:
        if show_track_map:
            # View mode selector
            view_mode = st.radio(
                t("charts.track_map.view_mode"),
                options=["line_comparison", "speed_coloring"],
                format_func=lambda x: t(f"charts.track_map.mode_{x}"),
                horizontal=True
            )

            try:
                track_fig = create_track_line_comparison(laps, mode=view_mode)
                st.plotly_chart(track_fig, width="stretch")
            except Exception as e:
                st.warning(f"Could not create track map: {e}")

        # Corner analysis moved here
        with st.expander(f"🔄 {t('corners.header')} - {ref_lap.name}", expanded=True):
            if laps:
                # Use track config if selected, otherwise auto-detect
                track_config = None
                if selected_track != t("app.sidebar.auto_detect"):
                    track_config = load_track_config(selected_track)
                else:
                    # Try to auto-detect track based on GPS and lap distance
                    center_lat = ref_lap.df['latitude'].mean() if 'latitude' in ref_lap.df.columns else None
                    center_lon = ref_lap.df['longitude'].mean() if 'longitude' in ref_lap.df.columns else None
                    detected = detect_track(ref_lap.lap_distance, center_lat, center_lon)
                    if detected:
                        track_config = load_track_config(detected)

                if track_config:
                    corners = corners_from_track_config(ref_lap.df, track_config)
                    source = t("corners.source_track", name=track_config.get('name', selected_track))
                else:
                    corners = detect_corners(ref_lap.df)
                    source = t("corners.source_auto")

                if corners:
                    st.markdown(f"**{t('corners.count', count=len(corners))}** ({source}) - {ref_lap.name}:")

                    corner_data = []
                    for c in corners:
                        dir_char = t("corners.direction.left") if c.direction == "left" else t("corners.direction.right")
                        corner_data.append({
                            t("corners.table.turn"): f"T{c.index}",
                            t("corners.table.direction"): dir_char,
                            t("corners.table.apex"): f"{c.apex_distance:.0f}m",
                            t("corners.table.speed"): f"{c.min_speed_kmh:.0f}",
                            t("corners.table.g_force"): f"{c.max_lateral_g:.2f}",
                            t("corners.table.brake"): f"{c.brake_point:.0f}m",
                        })

                    st.dataframe(corner_data, width="stretch", hide_index=True)
                else:
                    st.info(t("corners.no_corners"))

    st.divider()

    # === Analysis ===
    st.header(f"🔍 {t('analysis.header')}")

    # Bottleneck detection
    if len(laps) >= 2:
        st.subheader(f"🎯 {t('analysis.bottleneck.header')}")

        # Load track config for corner mapping (auto-detect if needed)
        bottleneck_track_cfg = None
        if selected_track != t("app.sidebar.auto_detect"):
            bottleneck_track_cfg = load_track_config(selected_track)
        else:
            # Try to auto-detect track based on GPS and lap distance
            center_lat = ref_lap.df['latitude'].mean() if 'latitude' in ref_lap.df.columns else None
            center_lon = ref_lap.df['longitude'].mean() if 'longitude' in ref_lap.df.columns else None
            detected = detect_track(ref_lap.lap_distance, center_lat, center_lon)
            if detected:
                bottleneck_track_cfg = load_track_config(detected)

        def get_location_name(start_m: float, end_m: float, category: str) -> str:
            """Map distance range to corner name with phase."""
            if bottleneck_track_cfg and "corners" in bottleneck_track_cfg:
                mid_point = (start_m + end_m) / 2
                corners = bottleneck_track_cfg["corners"]

                for corner in corners:
                    corner_start = corner.get("start_m", 0)
                    corner_end = corner.get("end_m", 0)
                    apex = corner.get("apex_distance_m", (corner_start + corner_end) / 2)

                    # Check if bottleneck overlaps with this corner
                    if start_m <= corner_end and end_m >= corner_start:
                        corner_id = corner.get("id", "?")

                        # Determine phase based on category and position
                        if category == "braking":
                            phase = t("analysis.bottleneck.phase.entry")
                        elif category == "acceleration":
                            phase = t("analysis.bottleneck.phase.exit")
                        elif mid_point < apex - 10:
                            phase = t("analysis.bottleneck.phase.entry")
                        elif mid_point > apex + 10:
                            phase = t("analysis.bottleneck.phase.exit")
                        else:
                            phase = t("analysis.bottleneck.phase.apex")

                        return t("analysis.bottleneck.location_corner", corner=corner_id, phase=phase)

            # Fallback to distance range
            return t("analysis.bottleneck.location_distance", start=start_m, end=end_m)

        bottlenecks = find_bottlenecks(laps, threshold_kmh=5.0)

        if bottlenecks:
            # Group bottlenecks by lap
            bottlenecks.sort(key=lambda x: x.lap_name)
            grouped = {k: list(v) for k, v in groupby(bottlenecks, key=lambda x: x.lap_name)}

            # Process each lap's bottlenecks
            for lap_name, lap_bottlenecks in grouped.items():
                # Find the lap object to get current time
                lap_obj = next((lap for lap in laps if lap.name == lap_name), None)

                # Show lap section header
                st.markdown(f"#### {t('analysis.bottleneck.lap_section', lap_name=lap_name, ref_name=ref_lap.name)}")

                # Sort by absolute speed difference
                lap_bottlenecks.sort(key=lambda x: abs(x.speed_diff_kmh), reverse=True)

                # Filter bottlenecks where driver is slower (time_diff_ms > 0 means slower)
                slow_bottlenecks = [b for b in lap_bottlenecks if b.time_diff_ms > 0]

                for b in lap_bottlenecks[:5]:  # Top 5 per lap
                    icon = "🟢" if b.speed_diff_kmh > 0 else "🔴"
                    time_str = f"+{b.time_diff_ms:.0f}ms" if b.time_diff_ms > 0 else f"{b.time_diff_ms:.0f}ms"

                    # Translate category
                    category_key = f"analysis.bottleneck.categories.{b.category}"
                    category_name = t(category_key) if category_key != t(category_key) else b.category

                    # Get location name (corner or distance)
                    location = get_location_name(b.start_m, b.end_m, b.category)

                    st.markdown(f"""
                    {icon} **{location}** ({category_name})
                    - {t('analysis.bottleneck.speed_delta', delta=b.speed_diff_kmh)}
                    - {t('analysis.bottleneck.time_impact', time=time_str)}
                    - {b.description}
                    """)

                # Show potential improvement summary for this lap
                if slow_bottlenecks and lap_obj:
                    total_time_loss_ms = sum(b.time_diff_ms for b in slow_bottlenecks)
                    total_time_loss_s = total_time_loss_ms / 1000.0

                    current_time = lap_obj.lap_time
                    target_time = current_time - total_time_loss_s

                    st.success(t(
                        'analysis.bottleneck.potential_summary',
                        count=len(slow_bottlenecks),
                        time=f"{total_time_loss_s:.2f}s"
                    ))
                    st.info(t(
                        'analysis.bottleneck.potential_target',
                        target=format_lap_time(target_time),
                        current=format_lap_time(current_time)
                    ))

                st.divider()
        else:
            st.info(t("analysis.bottleneck.no_bottlenecks", threshold=5))

    # Coach insights (prioritized suggestions)
    if len(laps) >= 2:
        st.subheader(f"🎯 {t('analysis.coach.header')}")
        st.caption(t("analysis.coach.comparing", lap1=laps[1].name, lap2=ref_lap.name))

        # Load track config if selected or auto-detect
        track_cfg = None
        if selected_track != t("app.sidebar.auto_detect"):
            track_cfg = load_track_config(selected_track)
        else:
            center_lat = ref_lap.df['latitude'].mean() if 'latitude' in ref_lap.df.columns else None
            center_lon = ref_lap.df['longitude'].mean() if 'longitude' in ref_lap.df.columns else None
            detected = detect_track(ref_lap.lap_distance, center_lat, center_lon)
            if detected:
                track_cfg = load_track_config(detected)

        coach_insights = generate_coach_insights(
            ref_lap, laps[1],
            track_config=track_cfg,
            rule_config=rule_config
        )

        if coach_insights:
            # Build table data
            coach_data = []
            for ci in coach_insights:
                priority_icon = {1: "🥇", 2: "🥈", 3: "🥉"}.get(ci.priority, f"#{ci.priority}")

                coach_data.append({
                    t("coach_table.priority"): priority_icon,
                    t("coach_table.area"): ci.location,
                    t("coach_table.potential"): f"+{ci.time_benefit_ms:.0f}ms",
                    t("coach_table.difficulty"): t(f"coach_table.difficulty_levels.{ci.difficulty}"),
                    t("coach_table.risk"): t(f"coach_table.difficulty_levels.{ci.risk}"),
                    t("coach_table.problem"): ci.problem,
                })

            st.dataframe(coach_data, width="stretch", hide_index=True)

            # Also show detailed suggestions in expander
            with st.expander(t("coach_table.suggestions"), expanded=False):
                for ci in coach_insights:
                    priority_icon = {1: "🥇", 2: "🥈", 3: "🥉"}.get(ci.priority, "📍")
                    st.markdown(f"**{priority_icon} {ci.location}**: {ci.suggestion}")
        else:
            st.success(t("analysis.coach.no_issues"))

    st.divider()

    # Detailed Racing insights (collapsible)
    lap_names = ", ".join([lap.name for lap in laps])
    with st.expander(f"📊 {t('analysis.technique.header')} ({lap_names})"):
        insights = analyze_racing_technique(laps)

        if insights:
            for insight in insights:
                # Use translation keys if available, otherwise fall back to English
                if insight.title_key and insight.format_args:
                    title = t(insight.title_key).format(**insight.format_args)
                    description = t(insight.description_key).format(**insight.format_args)
                else:
                    title = insight.title
                    description = insight.description

                icon = {"info": "ℹ️", "suggestion": "💡", "warning": "⚠️"}.get(insight.severity, "📝")
                if insight.severity == "info":
                    st.info(f"{icon} **{title}**: {description}")
                elif insight.severity == "suggestion":
                    st.warning(f"{icon} **{title}**: {description}")
                else:
                    st.error(f"{icon} **{title}**: {description}")
        else:
            st.info(t("analysis.technique.no_insights"))

    st.divider()

    # Footer
    st.markdown(f"""
    ---
    **{t('footer.title')}** | {t('footer.built_with')}

    {t('footer.tips_header')}
    - 🟢 {t('footer.tip_green')}
    - 🔴 {t('footer.tip_red')}
    - {t('footer.tip_focus')}
    - {t('footer.tip_compare')}
    """)


if __name__ == "__main__":
    main()
