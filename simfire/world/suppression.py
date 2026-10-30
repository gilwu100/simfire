"""
Suppression
===========

The drop model that sets the rate-of-spread multiplier (alpha) of each pixel under a
direct attack with water or retardant. The fire crosses a treated pixel at alpha times
the speed it would have without the drop, so alpha = 1 is no effect and alpha = 0 stops
the fire in that pixel.

alpha is set per pixel, from the pixel's own spread rate, fireline intensity, fuel, and
the depth of water or retardant it received:

    alpha = alpha_full(R) ** min(x / X_REF, 1),    x = depth received / depth needed

- `alpha_full(R)` is the multiplier of a pixel that received at least the depth it
  needs. For water it is the power law 0.38 R^-0.37 (R in m/min), our log-scale refit
  of the 88 before/after spread rates measured on 22 experimental fires by Wheatley
  (2024), Appendix D-5, capped at 1. For retardant it is a constant 0.05 (assumed).
- The depth needed rises with fireline intensity and depends on the fuel (McFayden et
  al. 2023, Figure A6-3, extended in proportion to intensity below 1,000 kW/m). For
  retardant it is the coverage level of the fuel (George 1985, Figure 2).
- The exponent min(x, 1) gives a partial effect below the needed depth (the form of
  Matei et al. 2026, Eq. 7) and no extra effect above it (Wheatley 2024, Section
  5.3.1: fires given more water were not slowed more).

The depth each pixel receives comes from the measured drop patterns: the length of
ground that received at least a given depth, by aircraft (McFayden et al. 2023, Table
A2-2), stretched in proportion to the load for aircraft that were not tested.

All quantities here are metric (m, m/min, kW/m, mm, litres); `RothermelFireManager`
converts from its feet and BTU before calling `cell_alpha`.
"""

from dataclasses import astuple
from typing import Dict, Tuple, Union

import numpy as np

from .parameters import Fuel
from .presets import (
    Brush,
    Chaparral,
    ClosedShortNeedleTimberLitter,
    DormantBrushHardwoodSlash,
    GrassTimberShrubOverstory,
    HardwoodLongNeedlePineTimber,
    HeavyLoggingSlash,
    LightLoggingSlash,
    MediumLoggingSlash,
    ShortGrass,
    ShortSparseDryClimateGrass,
    SouthernRough,
    TallGrass,
    TimberLitterUnderstory,
)

ArrayLike = Union[float, np.ndarray]

# Unit conversions from the units of `world.rothermel`
FT_PER_MIN_TO_M_PER_MIN: float = 0.3048
BTU_PER_FT_S_TO_KW_PER_M: float = 1.05506 / 0.3048

# Full-dose water multiplier, ln(alpha) = A0 + A1 ln(R), R in m/min. Wheatley (2024)
# Appendix D-5, refit on log axes: ln R_after = -0.96 + 0.63 ln R_before
WATER_A0: float = -0.96
WATER_A1: float = -0.37
# Spread rates below this are treated as this, in m/min (slowest rows in Wheatley)
MIN_ROS_M_PER_MIN: float = 0.05
# Full-dose retardant multiplier (our assumption)
RETARDANT_ALPHA: float = 0.05
# Dose ratio from which the full-dose multiplier applies
X_REF: float = 1.0

# Depth of water (mm) a fire edge needs, by need curve, at 1,000, 3,000 and 5,000 kW/m.
# McFayden et al. (2023), Figure A6-3
NEED_INTENSITY_KW_PER_M: Tuple[float, float, float] = (1000.0, 3000.0, 5000.0)
NEED_DEPTH_MM: Dict[str, Tuple[float, float, float]] = {
    "grass": (0.53, 1.27, 1.83),
    "pine": (0.75, 1.65, 2.55),
    "spruce": (0.85, 1.80, 2.65),
    "slash": (1.45, 2.50, 3.35),
}
# Below 1,000 kW/m the need is scaled in proportion to intensity, floored at this value
MIN_NEED_INTENSITY_KW_PER_M: float = 100.0

# Retardant coverage levels (mm = L/m^2), 1, 2, 3 and 6 gal/100 ft^2. George (1985),
# Figure 2
COVERAGE_MM: Dict[str, float] = {
    "grass": 0.41,
    "conifer_with_litter": 0.81,
    "brush": 1.22,
    "chaparral_and_heavy_slash": 2.44,
}

# Fuel group of each SimFire preset: (water need curve, retardant coverage class).
# Grass, shrub, and timber follow the grouping used in the ELMFIRE demo; chaparral and
# medium/heavy slash take the highest coverage level and light slash the brush level,
# as listed by George (1985), Figure 2
FUEL_GROUPS: Dict[Tuple[float, ...], Tuple[str, str]] = {
    astuple(ShortGrass): ("grass", "grass"),
    astuple(GrassTimberShrubOverstory): ("grass", "grass"),
    astuple(TallGrass): ("grass", "grass"),
    astuple(ShortSparseDryClimateGrass): ("grass", "grass"),
    astuple(Chaparral): ("pine", "chaparral_and_heavy_slash"),
    astuple(Brush): ("pine", "brush"),
    astuple(DormantBrushHardwoodSlash): ("pine", "brush"),
    astuple(SouthernRough): ("pine", "brush"),
    astuple(ClosedShortNeedleTimberLitter): ("spruce", "conifer_with_litter"),
    astuple(HardwoodLongNeedlePineTimber): ("spruce", "conifer_with_litter"),
    astuple(TimberLitterUnderstory): ("spruce", "conifer_with_litter"),
    astuple(LightLoggingSlash): ("slash", "brush"),
    astuple(MediumLoggingSlash): ("slash", "chaparral_and_heavy_slash"),
    astuple(HeavyLoggingSlash): ("slash", "chaparral_and_heavy_slash"),
}
# Fuels that are not presets (and non-burnable fuels, which never ignite)
DEFAULT_FUEL_GROUP: Tuple[str, str] = ("spruce", "conifer_with_litter")

# Measured drop patterns: lower edge of each depth band (mm) and the length of ground
# (m) that received at least that depth. McFayden et al. (2023), Table A2-2
PATTERN_DEPTH_MM: Tuple[float, ...] = (0.05, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5)
PATTERN_LENGTH_M: Dict[str, Tuple[float, ...]] = {
    "bell212_fast": (55, 47, 42, 40, 39, 34, 31, 26),
    "bell212_slow": (21, 21, 21, 21, 21, 21, 21, 21),
    "cl415": (135, 118, 102, 86, 65, 49, 26, 22),
    "cl215": (124, 95, 75, 61, 48, 31, 27, 18),
    "at802_fireboss": (108, 86, 75, 60, 42, 20, 10, 5),
}
# Load (L) of each measured drop
PATTERN_LOAD_L: Dict[str, float] = {
    "bell212_fast": 1325,
    "bell212_slow": 1325,
    "cl415": 6130,
    "cl215": 5450,
    "at802_fireboss": 3030,
}
# Aircraft: (measured pattern it uses, load in L)
PLATFORMS: Dict[str, Tuple[str, float]] = {
    "kmax": ("bell212_fast", 2570),
    "s64": ("bell212_fast", 9460),
    "uh60": ("bell212_fast", 3410),
    "bell205": ("bell212_fast", 1890),
    "bell212": ("bell212_fast", 1325),
    "bell407": ("bell212_fast", 1020),
    "rj85": ("cl415", 11360),
    "at802": ("at802_fireboss", 3030),
    "cl415": ("cl415", 6130),
    "fireboss": ("at802_fireboss", 2650),
}


def fuel_group(fuel: Fuel) -> Tuple[str, str]:
    """
    Return the (water need curve, retardant coverage class) of a fuel.

    Arguments:
        fuel: The fuel of a pixel

    Returns:
        A key of `NEED_DEPTH_MM` and a key of `COVERAGE_MM`
    """
    return FUEL_GROUPS.get(astuple(fuel), DEFAULT_FUEL_GROUP)


def alpha_full_dose(
    ros_m_per_min: ArrayLike, agent: str = "water", log_noise: ArrayLike = 0.0
) -> np.ndarray:
    """
    The multiplier of a pixel that received at least the depth it needs.

    Arguments:
        ros_m_per_min: Spread rate of the pixel without the drop (m/min)
        agent: "water" or "retardant"
        log_noise: Offset added to ln(alpha) for water, one draw per drop to carry the
                   scatter of the measurements (log sd 0.7 per drop in the demo), a
                   single value or one per pixel

    Returns:
        The full-dose multiplier, between 0 and 1
    """
    ros = np.asarray(ros_m_per_min, dtype=float)
    if agent == "retardant":
        return np.full_like(ros, RETARDANT_ALPHA)
    log_alpha = WATER_A0 + WATER_A1 * np.log(np.maximum(ros, MIN_ROS_M_PER_MIN))
    return np.minimum(1.0, np.exp(log_alpha + log_noise))


def depth_needed_mm(intensity_kw_per_m: ArrayLike, need_curve: np.ndarray) -> np.ndarray:
    """
    The depth of water a fire edge needs to stop flaming.

    Arguments:
        intensity_kw_per_m: Fireline intensity of each pixel (kW/m)
        need_curve: Key of `NEED_DEPTH_MM` for each pixel

    Returns:
        The needed depth of each pixel (mm)
    """
    intensity, need_curve = np.broadcast_arrays(
        np.asarray(intensity_kw_per_m, dtype=float), np.asarray(need_curve)
    )
    need = np.empty(intensity.shape)
    for curve in np.unique(need_curve):
        k = need_curve == curve
        table = NEED_DEPTH_MM[curve]
        low = (
            table[0]
            * np.maximum(intensity[k], MIN_NEED_INTENSITY_KW_PER_M)
            / NEED_INTENSITY_KW_PER_M[0]
        )
        need[k] = np.where(
            intensity[k] < NEED_INTENSITY_KW_PER_M[0],
            low,
            np.interp(intensity[k], NEED_INTENSITY_KW_PER_M, table),
        )
    return need


def cell_alpha(
    ros_m_per_min: ArrayLike,
    intensity_kw_per_m: ArrayLike,
    depth_mm: ArrayLike,
    fuel_groups: np.ndarray,
    agent: str = "water",
    log_noise: ArrayLike = 0.0,
) -> np.ndarray:
    """
    The multiplier of each pixel of a drop, alpha_full(R) ** min(x / X_REF, 1).

    Arguments:
        ros_m_per_min: Spread rate of each pixel without the drop (m/min)
        intensity_kw_per_m: Fireline intensity of each pixel without the drop (kW/m)
        depth_mm: Depth of water or retardant each pixel received (mm)
        fuel_groups: (need curve, coverage class) of each pixel, shape (n, 2), as
                     returned by `fuel_group`
        agent: "water" or "retardant"
        log_noise: See `alpha_full_dose`

    Returns:
        The multiplier of each pixel, between 0 and 1
    """
    fuel_groups = np.asarray(fuel_groups).reshape(-1, 2)
    if agent == "retardant":
        need = np.array([COVERAGE_MM[c] for c in fuel_groups[:, 1]], dtype=float)
    else:
        need = depth_needed_mm(intensity_kw_per_m, fuel_groups[:, 0])
    x = np.asarray(depth_mm, dtype=float) / np.maximum(need, 1e-3)
    exponent = np.minimum(x / X_REF, 1.0)
    return alpha_full_dose(ros_m_per_min, agent, log_noise) ** exponent


def footprint_depths_mm(platform: str, distances_m: ArrayLike) -> np.ndarray:
    """
    The depth one load leaves at given distances from the centre of its pattern, along
    the flight line: the lower edge of the deepest measured band whose half length
    (stretched by the load) reaches the point, and 0 beyond the shallowest band. The
    width of the pattern across the flight line is not resolved.

    Arguments:
        platform: A key of `PLATFORMS`
        distances_m: Distance of each pixel from the centre of the pattern (m)

    Returns:
        The depth each pixel receives (mm)
    """
    pattern, load = PLATFORMS[platform]
    half = np.asarray(PATTERN_LENGTH_M[pattern], float) * load / PATTERN_LOAD_L[pattern]
    half /= 2.0
    d = np.abs(np.asarray(distances_m, dtype=float))
    depth = np.zeros_like(d)
    # Shallow to deep, so the deepest band that reaches the pixel wins
    for band_depth, band_half in zip(PATTERN_DEPTH_MM, half):
        depth = np.where(band_half >= d, band_depth, depth)
    return depth


def footprint_line_depths_mm(platform: str, spacing_m: float) -> np.ndarray:
    """
    The depth one load leaves in each pixel of a straight line of pixels centred on the
    pattern. The number of pixels is the length of the shallowest band over the pixel
    spacing, at least one.

    Arguments:
        platform: A key of `PLATFORMS`
        spacing_m: Distance between neighbouring pixels along the line (m)

    Returns:
        The depth each pixel of the line receives (mm), in order along the line
    """
    pattern, load = PLATFORMS[platform]
    length = PATTERN_LENGTH_M[pattern][0] * load / PATTERN_LOAD_L[pattern]
    n_pixels = max(1, int(round(length / spacing_m)))
    distances = (np.arange(n_pixels) - (n_pixels - 1) / 2.0) * spacing_m
    return footprint_depths_mm(platform, distances)
