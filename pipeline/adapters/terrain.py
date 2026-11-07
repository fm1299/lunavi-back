"""Slope, roughness and elevation from AWS Terrain Tiles (Terrarium PNGs, no account needed).

Zoom 8 gives about 575 m per pixel at 20 degrees latitude, close to the 500 m baseline that the
Moon (LOLA 480 m) and Mars (MOLA 463 m) targets use. Metrics are computed on the tiles, then
averaged per cell. They are never derived from the resampled grid (spec 5.6).
"""

from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path

import httpx
import numpy as np
from PIL import Image

from pipeline.adapters import Context, Layer, cached_arrays
from pipeline.adapters.land import fine_land_mask
from pipeline.terrain_metrics import terrain_metrics

TILE_URL = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
ZOOM = 8
TILE_SIZE = 256
EARTH_CIRCUMFERENCE_M = 40_075_016.686


def _tile_x(lon: float, z: int) -> int:
    return int((lon + 180.0) / 360.0 * 2**z)


def _tile_y(lat: float, z: int) -> int:
    rad = np.radians(lat)
    return int((1.0 - np.arcsinh(np.tan(rad)) / np.pi) / 2.0 * 2**z)


def _pixel_lats(y: int, z: int) -> np.ndarray:
    """Latitude of each pixel row centre in a tile."""
    fractions = (y + (np.arange(TILE_SIZE) + 0.5) / TILE_SIZE) / 2**z
    return np.degrees(np.arctan(np.sinh(np.pi * (1.0 - 2.0 * fractions))))


def _pixel_lons(x: int, z: int) -> np.ndarray:
    return (x + (np.arange(TILE_SIZE) + 0.5) / TILE_SIZE) / 2**z * 360.0 - 180.0


def _download_tile(raw_dir: Path, x: int, y: int) -> np.ndarray:
    path = raw_dir / "terrain" / f"{ZOOM}_{x}_{y}.png"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        response = httpx.get(TILE_URL.format(z=ZOOM, x=x, y=y), timeout=60)
        response.raise_for_status()
        path.write_bytes(response.content)
    pixels = np.asarray(Image.open(BytesIO(path.read_bytes())).convert("RGB"), dtype=np.float64)
    return pixels[..., 0] * 256 + pixels[..., 1] + pixels[..., 2] / 256 - 32768


def _build(ctx: Context) -> dict[str, np.ndarray]:
    grid = ctx.grid
    half = grid.res / 2
    xs = range(_tile_x(grid.lons[0] - half, ZOOM), _tile_x(grid.lons[-1] + half, ZOOM) + 1)
    ys = range(_tile_y(grid.lats[-1] + half, ZOOM), _tile_y(grid.lats[0] - half, ZOOM) + 1)
    tiles = [(x, y) for x in xs for y in ys]

    land, west, north, fine = fine_land_mask(grid, ctx.raw_dir)
    totals = {key: np.zeros(grid.shape) for key in ("slope", "roughness", "elevation")}
    counts = np.zeros(grid.shape)

    def process(tile: tuple[int, int]) -> None:
        x, y = tile
        dem = _download_tile(ctx.raw_dir, x, y)
        lats = _pixel_lats(y, ZOOM)
        lons = _pixel_lons(x, ZOOM)
        pixel_m = EARTH_CIRCUMFERENCE_M * np.cos(np.radians(lats.mean())) / (TILE_SIZE * 2**ZOOM)
        slope, roughness = terrain_metrics(dem, pixel_m, pixel_m)
        lon2d, lat2d = np.meshgrid(lons, lats)
        fine_rows = np.floor((north - lat2d) / fine).astype(int)
        fine_cols = np.floor((lon2d - west) / fine).astype(int)
        inside = (
            (fine_rows >= 0)
            & (fine_rows < land.shape[0])
            & (fine_cols >= 0)
            & (fine_cols < land.shape[1])
        )
        on_land = np.zeros(inside.shape, dtype=bool)
        on_land[inside] = land[fine_rows[inside], fine_cols[inside]]
        rows = np.round((lat2d[on_land] - grid.lats[0]) / grid.res).astype(int)
        cols = np.round((lon2d[on_land] - grid.lons[0]) / grid.res).astype(int)
        np.add.at(totals["slope"], (rows, cols), slope[on_land])
        np.add.at(totals["roughness"], (rows, cols), roughness[on_land])
        np.add.at(totals["elevation"], (rows, cols), dem[on_land])
        np.add.at(counts, (rows, cols), 1)

    # Fill the disk cache in parallel, then add up one tile at a time.
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda tile: _download_tile(ctx.raw_dir, *tile), tiles))
    for tile in tiles:
        process(tile)

    with np.errstate(invalid="ignore", divide="ignore"):
        result = {
            key: np.where(counts > 0, total / counts, np.nan) for key, total in totals.items()
        }
    return result


def _layer(ctx: Context, key: str) -> Layer:
    arrays = cached_arrays(ctx, "terrain", lambda: _build(ctx))
    return Layer(values=arrays[key], gap_filled=np.zeros(ctx.grid.shape))


def slope(ctx: Context) -> Layer:
    return _layer(ctx, "slope")


def roughness(ctx: Context) -> Layer:
    return _layer(ctx, "roughness")


def elevation(ctx: Context) -> Layer:
    return _layer(ctx, "elevation")
