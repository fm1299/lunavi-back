"""Land fraction and country names per cell, from Natural Earth (public domain)."""

import json
from pathlib import Path

import numpy as np
from rasterio.features import rasterize
from rasterio.transform import from_origin

from pipeline.grid import Grid
from pipeline.net import fetch

NATURAL_EARTH_URL = (
    "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/{name}.geojson"
)
SUBCELLS = 10  # land is rasterized at a tenth of a cell


def _features(raw_dir: Path, name: str) -> list[dict]:
    path = raw_dir / "natural_earth" / f"{name}.geojson"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(fetch(NATURAL_EARTH_URL.format(name=name), timeout=180).content)
    return json.loads(path.read_text(encoding="utf-8"))["features"]


def _fine_mask(grid: Grid, geometries: list[dict]) -> tuple[np.ndarray, float, float, float]:
    """Rasterize shapes at a tenth of a cell: (north-up mask, west edge, north edge, pixel size)."""
    fine = grid.res / SUBCELLS
    west = float(grid.lons[0] - grid.res / 2)
    north = float(grid.lats[-1] + grid.res / 2)
    mask = rasterize(
        geometries,
        out_shape=(grid.shape[0] * SUBCELLS, grid.shape[1] * SUBCELLS),
        transform=from_origin(west, north, fine, fine),
        fill=0,
        default_value=1,
        dtype="uint8",
    ).astype(bool)
    return mask, west, north, fine


def cell_shares(grid: Grid, mask: np.ndarray, west: float, fine: float) -> np.ndarray:
    """Share of each cell inside a fine mask, with latitude ascending like the grid.

    The first column of a global grid hangs over the antimeridian. Only the part on the map counts,
    otherwise those cells can never pass as more than half covered.
    """
    ny, nx = grid.shape
    blocks = mask.reshape(ny, SUBCELLS, nx, SUBCELLS)
    shares = blocks.mean(axis=(1, 3))
    off_map = round((-180.0 - west) / fine)
    if 0 < off_map < SUBCELLS:
        shares[:, 0] = blocks[:, :, 0, off_map:].mean(axis=(1, 2))
    return shares[::-1]


def fine_land_mask(grid: Grid, raw_dir: Path) -> tuple[np.ndarray, float, float, float]:
    """Return (north-up boolean mask, west edge, north edge, fine pixel size in degrees)."""
    return _fine_mask(grid, [f["geometry"] for f in _features(raw_dir, "ne_10m_land")])


def land_fraction(grid: Grid, raw_dir: Path) -> np.ndarray:
    """Share of each cell that is land, with latitude ascending like the grid."""
    mask, west, _, fine = fine_land_mask(grid, raw_dir)
    return cell_shares(grid, mask, west, fine)


def glaciated_fraction(grid: Grid, raw_dir: Path) -> np.ndarray:
    """Share of each cell under permanent ice: glaciers, ice sheets and Antarctic ice shelves."""
    geometries = [
        feature["geometry"]
        for layer in ("ne_10m_glaciated_areas", "ne_10m_antarctic_ice_shelves_polys")
        for feature in _features(raw_dir, layer)
    ]
    mask, west, _, fine = _fine_mask(grid, geometries)
    return cell_shares(grid, mask, west, fine)


def country_names(grid: Grid, raw_dir: Path) -> np.ndarray:
    """Country under each cell, or None, so a result can say where it is. Latitude ascending."""
    features = _features(raw_dir, "ne_50m_admin_0_countries")
    names = [None, *[feature["properties"]["NAME"] for feature in features]]
    west = float(grid.lons[0] - grid.res / 2)
    north = float(grid.lats[-1] + grid.res / 2)
    codes = rasterize(
        [(feature["geometry"], index) for index, feature in enumerate(features, start=1)],
        out_shape=grid.shape,
        transform=from_origin(west, north, grid.res, grid.res),
        fill=0,
        all_touched=True,  # coastal cells can have their centre over water
        dtype="int32",
    )[::-1]
    return np.array(names, dtype=object)[codes]
