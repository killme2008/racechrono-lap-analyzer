"""
Rule configuration and adaptive thresholds for coach insights.

Provides:
- Corner type classification
- Adaptive thresholds based on corner type
- User-adjustable rule configuration
- Physics-based time estimation
"""

from dataclasses import dataclass, field
from enum import Enum


class CornerType(Enum):
    """Corner classification by typical speed."""

    HAIRPIN = "hairpin"  # < 60 km/h
    SLOW = "slow"  # 60-80 km/h
    MEDIUM = "medium"  # 80-100 km/h
    FAST = "fast"  # > 100 km/h


@dataclass
class CornerThresholds:
    """Thresholds for detecting issues in a corner."""

    brake_early_m: float  # Early braking threshold (meters)
    apex_speed_low_kmh: float  # Low apex speed threshold (km/h)
    exit_speed_low_kmh: float  # Low exit speed threshold (km/h)
    lateral_g_low: float  # Low lateral G threshold
    brake_force_weak_g: float  # Weak braking force threshold


# Default thresholds by corner type
DEFAULT_THRESHOLDS: dict[CornerType, CornerThresholds] = {
    CornerType.HAIRPIN: CornerThresholds(
        brake_early_m=8.0,
        apex_speed_low_kmh=3.0,
        exit_speed_low_kmh=5.0,
        lateral_g_low=0.10,
        brake_force_weak_g=0.15,
    ),
    CornerType.SLOW: CornerThresholds(
        brake_early_m=10.0,
        apex_speed_low_kmh=4.0,
        exit_speed_low_kmh=6.0,
        lateral_g_low=0.12,
        brake_force_weak_g=0.18,
    ),
    CornerType.MEDIUM: CornerThresholds(
        brake_early_m=12.0,
        apex_speed_low_kmh=5.0,
        exit_speed_low_kmh=8.0,
        lateral_g_low=0.15,
        brake_force_weak_g=0.20,
    ),
    CornerType.FAST: CornerThresholds(
        brake_early_m=15.0,
        apex_speed_low_kmh=8.0,
        exit_speed_low_kmh=10.0,
        lateral_g_low=0.20,
        brake_force_weak_g=0.25,
    ),
}


@dataclass
class RuleConfig:
    """User-adjustable rule configuration."""

    # Global sensitivity multiplier (0.5 = stricter, 2.0 = more lenient)
    sensitivity: float = 1.0

    # Override individual thresholds (optional)
    brake_early_m: float | None = None
    apex_speed_low_kmh: float | None = None
    exit_speed_low_kmh: float | None = None
    lateral_g_low: float | None = None
    brake_force_weak_g: float | None = None

    # Minimum time benefit to report (ms)
    min_time_benefit_ms: float = 10.0

    # Maximum insights to return
    max_insights: int = 5

    # Corner-specific overrides from track config
    corner_overrides: dict[int, dict] = field(default_factory=dict)


def get_corner_type(
    corner_config: dict | None = None, min_speed_kmh: float | None = None
) -> CornerType:
    """
    Determine corner type from config or speed.

    Priority:
    1. Explicit 'type' in corner_config
    2. 'typical_speed_kmh' in corner_config
    3. Provided min_speed_kmh
    4. Default to MEDIUM
    """
    # Try explicit type from config
    if corner_config and "type" in corner_config:
        type_str = corner_config["type"].lower()
        for ct in CornerType:
            if ct.value == type_str:
                return ct

    # Try typical_speed_kmh from config
    speed = None
    if corner_config and "typical_speed_kmh" in corner_config:
        speed = corner_config["typical_speed_kmh"]
    elif min_speed_kmh is not None:
        speed = min_speed_kmh

    if speed is not None:
        if speed < 60:
            return CornerType.HAIRPIN
        elif speed < 80:
            return CornerType.SLOW
        elif speed < 100:
            return CornerType.MEDIUM
        else:
            return CornerType.FAST

    return CornerType.MEDIUM


def get_thresholds(
    corner_type: CornerType,
    rule_config: RuleConfig | None = None,
    corner_config: dict | None = None,
) -> CornerThresholds:
    """
    Get thresholds for a corner, applying all overrides.

    Priority (highest to lowest):
    1. Corner-specific override in track config
    2. User override in RuleConfig
    3. Default for corner type (scaled by sensitivity)
    """
    # Start with defaults for this corner type
    defaults = DEFAULT_THRESHOLDS[corner_type]
    sensitivity = rule_config.sensitivity if rule_config else 1.0

    # Apply sensitivity to defaults
    brake_early_m = defaults.brake_early_m * sensitivity
    apex_speed_low_kmh = defaults.apex_speed_low_kmh * sensitivity
    exit_speed_low_kmh = defaults.exit_speed_low_kmh * sensitivity
    lateral_g_low = defaults.lateral_g_low * sensitivity
    brake_force_weak_g = defaults.brake_force_weak_g * sensitivity

    # Apply user overrides from RuleConfig
    if rule_config:
        if rule_config.brake_early_m is not None:
            brake_early_m = rule_config.brake_early_m
        if rule_config.apex_speed_low_kmh is not None:
            apex_speed_low_kmh = rule_config.apex_speed_low_kmh
        if rule_config.exit_speed_low_kmh is not None:
            exit_speed_low_kmh = rule_config.exit_speed_low_kmh
        if rule_config.lateral_g_low is not None:
            lateral_g_low = rule_config.lateral_g_low
        if rule_config.brake_force_weak_g is not None:
            brake_force_weak_g = rule_config.brake_force_weak_g

    # Apply corner-specific overrides from track config
    if corner_config and "thresholds_override" in corner_config:
        override = corner_config["thresholds_override"]
        if "brake_early_m" in override:
            brake_early_m = override["brake_early_m"]
        if "apex_speed_low_kmh" in override:
            apex_speed_low_kmh = override["apex_speed_low_kmh"]
        if "exit_speed_low_kmh" in override:
            exit_speed_low_kmh = override["exit_speed_low_kmh"]
        if "lateral_g_low" in override:
            lateral_g_low = override["lateral_g_low"]
        if "brake_force_weak_g" in override:
            brake_force_weak_g = override["brake_force_weak_g"]

    return CornerThresholds(
        brake_early_m=brake_early_m,
        apex_speed_low_kmh=apex_speed_low_kmh,
        exit_speed_low_kmh=exit_speed_low_kmh,
        lateral_g_low=lateral_g_low,
        brake_force_weak_g=brake_force_weak_g,
    )


def estimate_time_benefit_physics(
    speed_diff_kmh: float,
    section_length_m: float,
    base_speed_kmh: float,
    corner_type: CornerType,
) -> float:
    """
    Estimate time benefit using physics-based calculation.

    Formula: delta_t = section_length * (1/v_slow - 1/v_fast)

    Args:
        speed_diff_kmh: Speed difference (negative = slower than ref)
        section_length_m: Length of the section in meters
        base_speed_kmh: Reference speed in km/h
        corner_type: Type of corner for downstream impact multiplier

    Returns:
        Estimated time benefit in milliseconds
    """
    if base_speed_kmh <= 0 or section_length_m <= 0:
        return 0.0

    # Convert to m/s
    v_base = base_speed_kmh / 3.6
    v_compare = (base_speed_kmh + speed_diff_kmh) / 3.6

    # Prevent division by zero
    if v_compare <= 0:
        v_compare = 0.1

    # Time difference: t = distance / speed
    t_base = section_length_m / v_base
    t_compare = section_length_m / v_compare

    delta_ms = (t_compare - t_base) * 1000

    # Corner type multiplier for downstream impact
    # Exit speed matters more for hairpins (longer acceleration zone ahead)
    multipliers = {
        CornerType.HAIRPIN: 1.2,
        CornerType.SLOW: 1.1,
        CornerType.MEDIUM: 1.0,
        CornerType.FAST: 0.9,
    }

    return abs(delta_ms * multipliers.get(corner_type, 1.0))


def get_corner_config_by_index(
    corner_index: int, track_config: dict | None
) -> dict | None:
    """Get corner configuration by index from track config."""
    if not track_config or "corners" not in track_config:
        return None

    corners = track_config["corners"]
    if corner_index <= 0 or corner_index > len(corners):
        return None

    return corners[corner_index - 1]
