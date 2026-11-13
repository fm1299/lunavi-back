"""Terrain classes that make a cell eligible (spec section 5.9). A cell with no class is masked."""

import json
from pathlib import Path

import numpy as np

from pipeline.contract import Masks
from pipeline.geo import haversine_km
from pipeline.grid import Grid
from pipeline.net import fetch

GVP_URL = (
    "https://webservices.volcano.si.edu/geoserver/GVP-VOTW/ows?service=WFS&version=2.0.0"
    "&request=GetFeature&typeName=GVP-VOTW:Smithsonian_VOTW_Holocene_Volcanoes"
    "&outputFormat=application/json"
)


def load_volcanoes(raw_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Smithsonian Holocene volcano list as (lats, lons)."""
    path = raw_dir / "gvp" / "holocene_volcanoes.geojson"
    if not path.exists():
        response = fetch(GVP_URL, timeout=180)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(response.text, encoding="utf-8")
    features = json.loads(path.read_text(encoding="utf-8"))["features"]
    points = [f["geometry"]["coordinates"] for f in features if f.get("geometry")]
    return np.array([p[1] for p in points]), np.array([p[0] for p in points])


def near_points(
    lat2d: np.ndarray, lon2d: np.ndarray, lats: np.ndarray, lons: np.ndarray, km: float
) -> np.ndarray:
    """True for cells within km of any point."""
    result = np.zeros(lat2d.shape, dtype=bool)
    for lat, lon in zip(lats, lons, strict=True):
        result |= haversine_km(lat2d, lon2d, lat, lon) <= km
    return result


def classify_terrain(
    grid: Grid,
    aridity_mm: np.ndarray,
    composition: np.ndarray,
    near_volcano: np.ndarray,
    glaciated: np.ndarray,
    masks: Masks,
) -> np.ndarray:
    """Return a grid-shaped object array holding each cell's list of classes.

    Ice sheets get no class: they are not rock or regolith, and the terrain tiles record the
    bedrock under them, so their slope and roughness mean nothing. A documented ice-free site
    inside a coarse glacier polygon, such as the Dry Valleys, stays through a mask exception.
    """
    lat2d, lon2d = grid.mesh()
    half = grid.res / 2
    ice_free = glaciated <= masks.max_glaciated_fraction
    with np.errstate(invalid="ignore"):
        arid = (aridity_mm < masks.arid_precip_mm) & ice_free
    polar = (np.abs(lat2d) >= masks.polar_lat_deg) & ice_free
    volcanic = ((composition == "basaltic") | near_volcano) & ice_free

    other = np.zeros(grid.shape, dtype=bool)
    for exception in masks.mask_exceptions:
        west, south, east, north = exception.bbox
        other |= (
            (lon2d + half >= west)
            & (lon2d - half <= east)
            & (lat2d + half >= south)
            & (lat2d - half <= north)
        )

    classes = np.empty(grid.shape, dtype=object)
    for index in np.ndindex(grid.shape):
        cell = []
        if arid[index]:
            cell.append("arid")
        if polar[index]:
            cell.append("polar")
        if volcanic[index]:
            cell.append("volcanic")
        if other[index]:
            cell.append("other")
        classes[index] = cell
    return classes
