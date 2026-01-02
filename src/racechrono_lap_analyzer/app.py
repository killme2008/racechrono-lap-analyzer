"""
RaceChrono Lap Comparison Tool - Streamlit App

A visual tool for comparing lap data from RaceChrono Pro.
Supports multiple lap files with speed, acceleration, and OBD data comparison.
"""

import tempfile
from pathlib import Path

import streamlit as st

from racechrono_lap_analyzer.analysis import (
    analyze_racing_technique,
    corners_from_track_config,
    detect_corners,
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
    create_track_map,
)
from racechrono_lap_analyzer.data_parser import (
    LapData,
    format_lap_time,
    generate_friendly_name,
    load_lap,
)

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


def main():
    st.title("🏎️ RaceChrono Lap Comparison Tool")
    st.markdown("Compare lap data from RaceChrono Pro. Upload multiple CSV files to analyze.")

    # Sidebar - File Upload
    with st.sidebar:
        st.header("📁 Data Files")

        uploaded_files = st.file_uploader(
            "Upload RaceChrono CSV files",
            type=['csv'],
            accept_multiple_files=True,
            help="Select one or more CSV files exported from RaceChrono Pro (CSV3 format)"
        )

        if uploaded_files:
            st.success(f"{len(uploaded_files)} file(s) uploaded")

        st.divider()

        st.header("⚙️ Settings")

        # Track selection
        available_tracks = get_available_tracks()
        track_options = ["Auto-detect"] + available_tracks
        selected_track = st.selectbox(
            "Track Configuration",
            track_options,
            help="Select track for accurate corner detection"
        )

        show_delta = st.checkbox("Show speed delta overlay", value=True)
        show_time_delta = st.checkbox("Show time delta chart", value=True)
        show_track_map = st.checkbox("Show track map", value=True)
        show_gg = st.checkbox("Show G-G diagram", value=True)

        st.divider()

        st.markdown("""
        **Tips:**
        - Upload 2+ laps for comparison
        - Fastest lap is used as reference
        - Green = faster, Red = slower
        """)

    # Main content
    if not uploaded_files:
        st.info("👆 Upload CSV files from the sidebar to get started.")

        with st.expander("📖 How to export from RaceChrono Pro"):
            st.markdown("""
            1. Open RaceChrono Pro
            2. Select a session
            3. Tap the share icon
            4. Choose **Export as CSV**
            5. Select **CSV v3** format
            6. Include channels: GPS, Accelerometer, and OBD (if available)
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
            st.error(f"Failed to load: {error}")

    if not laps:
        st.error("No valid lap files loaded.")
        return

    # Sort laps by time (fastest first)
    laps.sort(key=lambda x: x.lap_time)

    # === Lap Summary Cards ===
    st.header("📊 Lap Summary")

    ref_lap = laps[0]  # Fastest lap as reference

    cols = st.columns(len(laps))
    for i, (col, lap) in enumerate(zip(cols, laps, strict=False)):
        with col:
            # Header with trophy for fastest lap
            if i == 0:
                st.markdown(f"### 🏆 {lap.name}")
                st.caption("Reference (Fastest)")
            else:
                st.markdown(f"### {lap.name}")
                time_delta = lap.lap_time - ref_lap.lap_time
                st.caption(f"+{time_delta:.3f}s vs reference")

            # Lap time metric
            st.metric(
                label="Lap Time",
                value=format_lap_time(lap.lap_time),
                delta=f"+{lap.lap_time - ref_lap.lap_time:.3f}s" if i > 0 else None,
                delta_color="inverse"
            )

            # Speed metrics in 2 columns
            speed_cols = st.columns(2)
            with speed_cols[0]:
                speed_delta = lap.max_speed_kmh - ref_lap.max_speed_kmh if i > 0 else None
                st.metric(
                    label="Max Speed",
                    value=f"{lap.max_speed_kmh:.1f}",
                    delta=f"{speed_delta:+.1f}" if speed_delta else None,
                    help="km/h"
                )
            with speed_cols[1]:
                avg_delta = lap.avg_speed_kmh - ref_lap.avg_speed_kmh if i > 0 else None
                st.metric(
                    label="Avg Speed",
                    value=f"{lap.avg_speed_kmh:.1f}",
                    delta=f"{avg_delta:+.1f}" if avg_delta else None,
                    help="km/h"
                )

            # G-force metrics in 2 columns
            g_cols = st.columns(2)
            with g_cols[0]:
                lat_delta = lap.max_lateral_g - ref_lap.max_lateral_g if i > 0 else None
                st.metric(
                    label="Lateral G",
                    value=f"{lap.max_lateral_g:.2f}",
                    delta=f"{lat_delta:+.2f}" if lat_delta else None,
                    help="Max cornering force"
                )
            with g_cols[1]:
                brake_delta = lap.max_brake_g - ref_lap.max_brake_g if i > 0 else None
                st.metric(
                    label="Brake G",
                    value=f"{lap.max_brake_g:.2f}",
                    delta=f"{brake_delta:+.2f}" if brake_delta else None,
                    help="Max braking force"
                )

    # === Metadata ===
    with st.expander("📋 Session Metadata"):
        meta_cols = st.columns(len(laps))
        for col, lap in zip(meta_cols, laps, strict=False):
            with col:
                original_name = getattr(lap, 'original_name', lap.name)
                st.markdown(f"**{lap.name}**")
                st.text(f"File: {original_name}")
                st.text(f"Track: {lap.metadata.track_name}")
                st.text(f"Session: {lap.metadata.session_title}")
                st.text(f"Date: {lap.metadata.created_date}")
                st.text(f"Lap #: {lap.lap_number}")
                st.text(f"Distance: {lap.lap_distance:.0f}m")

                # Available data
                obd_status = "✅ OBD data" if lap.has_obd_data else "❌ No OBD"
                st.text(obd_status)

    st.divider()

    # === Charts with Tabs ===
    st.header("📈 Comparison Charts")

    tab_overview, tab_gforce, tab_telemetry, tab_track = st.tabs([
        "🏁 Overview",
        "📊 G-Force",
        "🔧 Telemetry",
        "🗺️ Track"
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
            st.info("No OBD/telemetry data available. Connect OBD adapter to RaceChrono for throttle, brake, and RPM data.")

    # Tab 4: Track (Map + Corner Analysis)
    with tab_track:
        if show_track_map:
            try:
                track_map = create_track_map(laps)
                st.plotly_chart(track_map, width="stretch")
            except Exception as e:
                st.warning(f"Could not create track map: {e}")

        # Corner analysis moved here
        with st.expander("🔄 Corner-by-Corner Analysis", expanded=True):
            if laps:
                # Use track config if selected, otherwise auto-detect
                track_config = None
                if selected_track != "Auto-detect":
                    track_config = load_track_config(selected_track)

                if track_config:
                    corners = corners_from_track_config(ref_lap.df, track_config)
                    source = f"Track: {track_config.get('name', selected_track)}"
                else:
                    corners = detect_corners(ref_lap.df)
                    source = "Auto-detected"

                if corners:
                    st.markdown(f"**{len(corners)} corners** ({source}) - {ref_lap.name}:")

                    corner_data = []
                    for c in corners:
                        corner_data.append({
                            "T#": f"T{c.index}",
                            "Dir": c.direction.capitalize()[:1],
                            "Apex": f"{c.apex_distance:.0f}m",
                            "Speed": f"{c.min_speed_kmh:.0f}",
                            "G": f"{c.max_lateral_g:.2f}",
                            "Brake": f"{c.brake_point:.0f}m",
                        })

                    st.dataframe(corner_data, width="stretch", hide_index=True)
                else:
                    st.info("No corners detected. Check lateral acceleration data.")

    st.divider()

    # === Analysis ===
    st.header("🔍 Analysis")

    # Bottleneck detection
    if len(laps) >= 2:
        st.subheader("🎯 Bottleneck Analysis")

        bottlenecks = find_bottlenecks(laps, threshold_kmh=5.0)

        if bottlenecks:
            # Sort by absolute speed difference
            bottlenecks.sort(key=lambda x: abs(x.speed_diff_kmh), reverse=True)

            for b in bottlenecks[:10]:  # Top 10
                icon = "🟢" if b.speed_diff_kmh > 0 else "🔴"
                time_str = f"+{b.time_diff_ms:.0f}ms" if b.time_diff_ms > 0 else f"{b.time_diff_ms:.0f}ms"

                st.markdown(f"""
                {icon} **{b.start_m:.0f}m - {b.end_m:.0f}m** ({b.category})
                - Speed delta: {b.speed_diff_kmh:+.1f} km/h
                - Time impact: {time_str}
                - {b.description}
                """)
        else:
            st.info("No significant bottlenecks detected (threshold: 5 km/h difference)")

    # Coach insights (prioritized suggestions)
    if len(laps) >= 2:
        st.subheader("🎯 Coach Suggestions")
        st.caption(f"Comparing {laps[1].name} vs {ref_lap.name} (reference)")

        # Load track config if selected
        track_cfg = None
        if selected_track != "Auto-detect":
            track_cfg = load_track_config(selected_track)

        coach_insights = generate_coach_insights(ref_lap, laps[1], track_config=track_cfg)

        if coach_insights:
            for ci in coach_insights:
                priority_icon = {1: "🥇", 2: "🥈", 3: "🥉"}.get(ci.priority, "📍")
                category_icon = {
                    'braking': '🛑',
                    'corner_speed': '🔄',
                    'exit': '🚀',
                    'straight': '➡️',
                    'line': '📐'
                }.get(ci.category, '💡')

                time_str = f"+{ci.time_benefit_ms:.0f}ms" if ci.time_benefit_ms > 0 else ""

                st.markdown(f"""
                {priority_icon} **{ci.location}** {category_icon} {time_str}
                - **Problem**: {ci.problem}
                - **Suggestion**: {ci.suggestion}
                """)
        else:
            st.success("Great job! No significant areas for improvement detected.")

    st.divider()

    # Detailed Racing insights (collapsible)
    with st.expander("📊 Detailed Technical Analysis"):
        insights = analyze_racing_technique(laps)

        if insights:
            for insight in insights:
                icon = {"info": "ℹ️", "suggestion": "💡", "warning": "⚠️"}.get(insight.severity, "📝")
                if insight.severity == "info":
                    st.info(f"{icon} **{insight.title}**: {insight.description}")
                elif insight.severity == "suggestion":
                    st.warning(f"{icon} **{insight.title}**: {insight.description}")
                else:
                    st.error(f"{icon} **{insight.title}**: {insight.description}")
        else:
            st.info("Upload more laps or ensure data quality for detailed insights.")

    st.divider()

    # Footer
    st.markdown("""
    ---
    **RaceChrono Lap Comparison Tool** | Built with Streamlit & Plotly

    Tips for improvement:
    - 🟢 Green areas = you're faster here
    - 🔴 Red areas = opportunity to improve
    - Focus on the biggest bottlenecks first
    - Compare your best lap against previous attempts
    """)


if __name__ == "__main__":
    main()
