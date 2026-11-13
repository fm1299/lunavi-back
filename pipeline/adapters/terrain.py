"""Slope, roughness and elevation from AWS Terrain Tiles (Terrarium PNGs, no account needed).

Metrics use a baseline of about 500 m, like the Moon (LOLA) and Mars (MOLA) sites. Zoom 8 pixels
shrink toward the poles, so tiles are averaged down at high latitude to hold that baseline. Metrics
are computed on the tiles and then averaged per cell, never from the resampled grid (spec 5.6).
"""

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path

import httpx
import numpy as np
from PIL import Image

from pipeline.adapters import Context, Layer, cached_arrays
from pipeline.adapters.land import fine_land_mask
from pipeline.grid import block_nanmean
from pipeline.net import fetch
from pipeline.terrain_metrics import terrain_metrics

TILE_URL = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
ZOOM = 8
TILE_SIZE = 256
BASELINE_M = 500.0
EARTH_CIRCUMFERENCE_M = 40_075_016.686
WORKERS = 16
MERCATOR_LIMIT_DEG = 85.05  # the tile pyramid stops here
LAND_SAMPLES = 32  # pixels per side checked to decide whether a tile needs downloading


def _tile_x(lon: float, z: int) -> int:
    return int(np.clip((lon + 180.0) / 360.0 * 2**z, 0, 2**z - 1))


def _tile_y(lat: float, z: int) -> int:
    lat = float(np.clip(lat, -MERCATOR_LIMIT_DEG, MERCATOR_LIMIT_DEG))
    position = (1.0 - np.arcsinh(np.tan(np.radians(lat))) / np.pi) / 2.0 * 2**z
    return int(np.clip(position, 0, 2**z - 1))


def _pixel_lats(y: int, z: int, size: int = TILE_SIZE) -> np.ndarray:
    """Latitude of each row centre in a tile."""
    fractions = (y + (np.arange(size) + 0.5) / size) / 2**z
    return np.degrees(np.arctan(np.sinh(np.pi * (1.0 - 2.0 * fractions))))


def _pixel_lons(x: int, z: int, size: int = TILE_SIZE) -> np.ndarray:
    return (x + (np.arange(size) + 0.5) / size) / 2**z * 360.0 - 180.0


def baseline_factor(pixel_m: float, baseline_m: float = BASELINE_M) -> int:
    """How many pixels to average so that one step is about the common baseline."""
    return max(1, round(baseline_m / pixel_m))


def _download_tile(raw_dir: Path, x: int, y: int) -> np.ndarray | None:
    """Heights in metres. None when the service has no tile here."""
    path = raw_dir / "terrain" / f"{ZOOM}_{x}_{y}.png"
    if not path.exists():
        try:
            content = fetch(TILE_URL.format(z=ZOOM, x=x, y=y), timeout=60).content
        except httpx.HTTPStatusError as error:
            if error.response.status_code in (403, 404):
                return None
            raise
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".part")
        temporary.write_bytes(content)
        temporary.replace(path)
    pixels = np.asarray(Image.open(BytesIO(path.read_bytes())).convert("RGB"), dtype=np.float64)
    return pixels[..., 0] * 256 + pixels[..., 1] + pixels[..., 2] / 256 - 32768


def _block_1d(values: np.ndarray, factor: int) -> np.ndarray:
    count = values.size // factor
    return values[: count * factor].reshape(count, factor).mean(axis=1)


def _land_lookup(ctx: Context) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """Return a function telling whether points fall on land inside the grid."""
    land, west, north, fine = fine_land_mask(ctx.grid, ctx.raw_dir)

    def is_land(lat2d: np.ndarray, lon2d: np.ndarray) -> np.ndarray:
        rows = np.floor((north - lat2d) / fine).astype(int)
        cols = np.floor((lon2d - west) / fine).astype(int)
        inside = (rows >= 0) & (rows < land.shape[0]) & (cols >= 0) & (cols < land.shape[1])
        result = np.zeros(inside.shape, dtype=bool)
        result[inside] = land[rows[inside], cols[inside]]
        return result

    return is_land


def _build(ctx: Context) -> dict[str, np.ndarray]:
    grid = ctx.grid
    half = grid.res / 2
    is_land = _land_lookup(ctx)

    xs = range(_tile_x(grid.lons[0] - half, ZOOM), _tile_x(grid.lons[-1] + half, ZOOM) + 1)
    ys = range(_tile_y(grid.lats[-1] + half, ZOOM), _tile_y(grid.lats[0] - half, ZOOM) + 1)
    tiles = []
    for y in ys:
        lats = _pixel_lats(y, ZOOM, LAND_SAMPLES)
        for x in xs:
            lon2d, lat2d = np.meshgrid(_pixel_lons(x, ZOOM, LAND_SAMPLES), lats)
            if is_land(lat2d, lon2d).any():
                tiles.append((x, y))
    print(f"    terrain: {len(tiles)} land tiles", flush=True)

    def process(tile: tuple[int, int]):
        x, y = tile
        dem = _download_tile(ctx.raw_dir, x, y)
        if dem is None:
            return None
        lats = _pixel_lats(y, ZOOM)
        lons = _pixel_lons(x, ZOOM)
        pixel_m = EARTH_CIRCUMFERENCE_M * np.cos(np.radians(lats.mean())) / (TILE_SIZE * 2**ZOOM)
        factor = baseline_factor(pixel_m)
        if factor > 1:
            dem = block_nanmean(dem, factor)
            lats = _block_1d(lats, factor)
            lons = _block_1d(lons, factor)
            pixel_m *= factor
        slope, roughness = terrain_metrics(dem, pixel_m, pixel_m)
        lon2d, lat2d = np.meshgrid(lons, lats)
        on_land = is_land(lat2d, lon2d)
        rows = np.round((lat2d[on_land] - grid.lats[0]) / grid.res).astype(int)
        cols = np.round((lon2d[on_land] - grid.lons[0]) / grid.res).astype(int)
        rows = np.clip(rows, 0, grid.shape[0] - 1)
        cols = np.clip(cols, 0, grid.shape[1] - 1)
        return rows, cols, slope[on_land], roughness[on_land], dem[on_land]

    totals = {key: np.zeros(grid.shape) for key in ("slope", "roughness", "elevation")}
    counts = np.zeros(grid.shape)
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for done, result in enumerate(pool.map(process, tiles), start=1):
            if result is not None:
                rows, cols, slope_values, roughness_values, heights = result
                np.add.at(totals["slope"], (rows, cols), slope_values)
                np.add.at(totals["roughness"], (rows, cols), roughness_values)
                np.add.at(totals["elevation"], (rows, cols), heights)
                np.add.at(counts, (rows, cols), 1)
            if done % 2000 == 0:
                print(f"    terrain: {done} of {len(tiles)} tiles", flush=True)

    with np.errstate(invalid="ignore", divide="ignore"):
        return {key: np.where(counts > 0, total / counts, np.nan) for key, total in totals.items()}


def _layer(ctx: Context, key: str) -> Layer:
    arrays = cached_arrays(ctx, "terrain", lambda: _build(ctx))
    return Layer(values=arrays[key], gap_filled=np.zeros(ctx.grid.shape))


def slope(ctx: Context) -> Layer:
    return _layer(ctx, "slope")


def roughness(ctx: Context) -> Layer:
    return _layer(ctx, "roughness")


def elevation(ctx: Context) -> Layer:
    return _layer(ctx, "elevation")
