"""Land fraction per cell from Natural Earth (public domain)."""

import json
from pathlib import Path

import httpx
import numpy as np
from rasterio.features import rasterize
from rasterio.transform import from_origin

from pipeline.grid import Grid

NATURAL_EARTH_URL = (
    "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/"
    "ne_10m_land.geojson"
)
SUBCELLS = 10  # land is rasterized at a tenth of a cell


def _land_shapes(raw_dir: Path) -> list:
    path = raw_dir / "natural_earth" / "ne_10m_land.geojson"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        response = httpx.get(NATURAL_EARTH_URL, timeout=180, follow_redirects=True)
        response.raise_for_status()
        path.write_bytes(response.content)
    data = json.loads(path.read_text(encoding="utf-8"))
    return [feature["geometry"] for feature in data["features"]]


def fine_land_mask(grid: Grid, raw_dir: Path) -> tuple[np.ndarray, float, float, float]:
    """Return (north-up boolean mask, west edge, north edge, fine pixel size in degrees)."""
    fine = grid.res / SUBCELLS
    west = float(grid.lons[0] - grid.res / 2)
    north = float(grid.lats[-1] + grid.res / 2)
    shape = (grid.shape[0] * SUBCELLS, grid.shape[1] * SUBCELLS)
    mask = rasterize(
        _land_shapes(raw_dir),
        out_shape=shape,
        transform=from_origin(west, north, fine, fine),
        fill=0,
        default_value=1,
        dtype="uint8",
    ).astype(bool)
    return mask, west, north, fine


def land_fraction(grid: Grid, raw_dir: Path) -> np.ndarray:
    """Share of each cell that is land, with latitude ascending like the grid."""
    mask, *_ = fine_land_mask(grid, raw_dir)
    ny, nx = grid.shape
    blocks = mask.reshape(ny, SUBCELLS, nx, SUBCELLS)
    return blocks.mean(axis=(1, 3))[::-1]
