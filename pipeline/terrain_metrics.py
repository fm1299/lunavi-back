"""Slope and roughness, computed the same way on Earth, the Moon and Mars (spec 5.6)."""

import numpy as np
from scipy.ndimage import uniform_filter


def terrain_metrics(dem: np.ndarray, dx_m: float, dy_m: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (slope in degrees, roughness in metres) for a DEM at about a 500 m baseline.

    Roughness is the standard deviation of height in a 3 x 3 pixel window.
    """
    dem = np.asarray(dem, dtype=np.float64)
    dz_dy, dz_dx = np.gradient(dem, dy_m, dx_m)
    slope = np.degrees(np.arctan(np.hypot(dz_dx, dz_dy)))
    mean = uniform_filter(dem, size=3, mode="nearest")
    mean_of_squares = uniform_filter(dem * dem, size=3, mode="nearest")
    roughness = np.sqrt(np.maximum(mean_of_squares - mean * mean, 0.0))
    return slope, roughness
