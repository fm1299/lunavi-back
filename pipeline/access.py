"""How easy a cell is to reach for field work (spec FR-P11): distance to roads and airports.

shortcut: Natural Earth roads and airports stand in for OpenStreetMap, and protected areas are
not included because Natural Earth has no global layer; swap the readers below when the team
picks a source for them (OpenStreetMap or the event data).
"""

from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from pipeline import scoring
from pipeline.adapters.land import _features
from pipeline.contract import AccessConfig
from pipeline.geo import EARTH_RADIUS_KM
from pipeline.grid import Grid


def _unit_vectors(lat, lon) -> np.ndarray:
    lat, lon = np.radians(lat), np.radians(lon)
    return np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=-1)


def nearest_km(lat2d: np.ndarray, lon2d: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Great-circle distance from each cell centre to the nearest (lon, lat) point."""
    if len(points) == 0:
        return np.full(lat2d.shape, np.nan)
    tree = cKDTree(_unit_vectors(points[:, 1], points[:, 0]))
    chord, _ = tree.query(_unit_vectors(lat2d.ravel(), lon2d.ravel()))
    return (2 * EARTH_RADIUS_KM * np.arcsin(np.clip(chord / 2, 0, 1))).reshape(lat2d.shape)


def road_points(raw_dir: Path) -> np.ndarray:
    """Every vertex of every Natural Earth road, as (lon, lat). Ferry routes are not roads."""
    lines = []
    for feature in _features(raw_dir, "ne_10m_roads"):
        if feature["properties"].get("type") == "Ferry Route":
            continue
        geometry = feature["geometry"]
        parts = geometry["coordinates"]
        lines.extend([parts] if geometry["type"] == "LineString" else parts)
    return np.concatenate([np.asarray(line, dtype=np.float64)[:, :2] for line in lines])


def airport_points(raw_dir: Path) -> np.ndarray:
    features = _features(raw_dir, "ne_10m_airports")
    return np.array([f["geometry"]["coordinates"][:2] for f in features], dtype=np.float64)


def access_layers(grid: Grid, raw_dir: Path, config: AccessConfig) -> dict[str, np.ndarray]:
    """Grid-shaped road_km, airport_km and the access score from 0 to 1."""
    lat2d, lon2d = grid.mesh()
    road_km = nearest_km(lat2d, lon2d, road_points(raw_dir))
    airport_km = nearest_km(lat2d, lon2d, airport_points(raw_dir))
    return {
        "road_km": road_km,
        "airport_km": airport_km,
        "access": scoring.access_score(
            road_km, airport_km, config.road_far_km, config.airport_far_km
        ),
    }
