"""Climate features from the NASA POWER climatology API (no account needed).

The regional endpoint allows one parameter and 10 x 10 degrees per request, so the area is split
into a fixed 10-degree layout. TS is surface (skin) temperature, the same kind of value as the Moon
and Mars temperatures.
"""

import json
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from pipeline.adapters import Context, Layer, cached_arrays
from pipeline.grid import Grid
from pipeline.net import fetch

API_URL = "https://power.larc.nasa.gov/api/temporal/climatology/regional"
MONTHS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
SPAN_DEG = 10
MARGIN_DEG = 1.0  # read past the edge of the land so that interpolation covers every cell
DAYS_PER_YEAR = 365.25
FILL_VALUE = -999
WORKERS = 4  # the API limits concurrent requests


def aligned_chunks(low: float, high: float, minimum: int, maximum: int) -> list[tuple[int, int]]:
    """Split a range into 10-degree steps on multiples of 10, inside the valid limits."""
    start = max(minimum, math.floor(low / SPAN_DEG) * SPAN_DEG)
    stop = min(maximum, math.ceil(high / SPAN_DEG) * SPAN_DEG)
    return [(edge, edge + SPAN_DEG) for edge in range(start, stop, SPAN_DEG)]


def _fetch_chunk(raw_dir: Path, parameter: str, lats: tuple, lons: tuple) -> list[dict]:
    path = raw_dir / "power" / f"{parameter}_{lats[0]}_{lats[1]}_{lons[0]}_{lons[1]}.json"
    if not path.exists():
        response = fetch(
            API_URL,
            params={
                "parameters": parameter,
                "community": "RE",
                "format": "JSON",
                "latitude-min": lats[0],
                "latitude-max": lats[1],
                "longitude-min": lons[0],
                "longitude-max": lons[1],
            },
            timeout=180,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(response.text, encoding="utf-8")
    return json.loads(path.read_text(encoding="utf-8"))["features"]


def _chunks_with_land(ctx: Context) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    grid = ctx.grid
    lat_chunks = aligned_chunks(grid.lats[0] - MARGIN_DEG, grid.lats[-1] + MARGIN_DEG, -90, 90)
    lon_chunks = aligned_chunks(grid.lons[0] - MARGIN_DEG, grid.lons[-1] + MARGIN_DEG, -180, 180)
    chunks = []
    for lats in lat_chunks:
        rows = (grid.lats >= lats[0] - MARGIN_DEG) & (grid.lats <= lats[1] + MARGIN_DEG)
        for lons in lon_chunks:
            cols = (grid.lons >= lons[0] - MARGIN_DEG) & (grid.lons <= lons[1] + MARGIN_DEG)
            if ctx.land_mask[np.ix_(rows, cols)].any():
                chunks.append((lats, lons))
    return chunks


def _fetch_parameter(ctx: Context, parameter: str) -> dict[str, np.ndarray]:
    """Return lats, lons and a 13 x ny x nx array: twelve months, then the annual value."""
    chunks = _chunks_with_land(ctx)
    print(f"    power {parameter}: {len(chunks)} requests", flush=True)

    def read(chunk: tuple) -> list[dict]:
        return _fetch_chunk(ctx.raw_dir, parameter, chunk[0], chunk[1])

    records: dict[tuple[float, float], list[float]] = {}
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for features in pool.map(read, chunks):
            for feature in features:
                lon, lat = feature["geometry"]["coordinates"][:2]
                values = feature["properties"]["parameter"][parameter]
                records[(lat, lon)] = [values[m] for m in MONTHS] + [values["ANN"]]

    lats = np.array(sorted({key[0] for key in records}))
    lons = np.array(sorted({key[1] for key in records}))
    stack = np.full((13, len(lats), len(lons)), np.nan)
    lat_index = {value: i for i, value in enumerate(lats)}
    lon_index = {value: i for i, value in enumerate(lons)}
    for (lat, lon), values in records.items():
        stack[:, lat_index[lat], lon_index[lon]] = values
    stack[stack == FILL_VALUE] = np.nan
    return {"lats": lats, "lons": lons, "stack": stack}


def _to_grid(lats: np.ndarray, lons: np.ndarray, values: np.ndarray, grid: Grid) -> np.ndarray:
    interpolate = RegularGridInterpolator(
        (lats, lons), values, method="linear", bounds_error=False, fill_value=None
    )
    lat2d, lon2d = grid.mesh()
    return interpolate(np.stack([lat2d, lon2d], axis=-1))


def _parameter(ctx: Context, parameter: str) -> dict[str, np.ndarray]:
    return cached_arrays(ctx, f"power_{parameter}", lambda: _fetch_parameter(ctx, parameter))


def _layer(ctx: Context, values: np.ndarray) -> Layer:
    return Layer(values=values, gap_filled=np.zeros(ctx.grid.shape))


def mean_temperature(ctx: Context) -> Layer:
    data = _parameter(ctx, "TS")
    return _layer(ctx, _to_grid(data["lats"], data["lons"], data["stack"][12], ctx.grid))


def temp_annual_range(ctx: Context) -> Layer:
    data = _parameter(ctx, "TS")
    months = data["stack"][:12]
    spread = months.max(axis=0) - months.min(axis=0)
    return _layer(ctx, _to_grid(data["lats"], data["lons"], spread, ctx.grid))


def aridity(ctx: Context) -> Layer:
    """Annual precipitation in mm/yr. POWER reports the annual climatology in mm/day."""
    data = _parameter(ctx, "PRECTOTCORR")
    per_year = data["stack"][12] * DAYS_PER_YEAR
    return _layer(ctx, _to_grid(data["lats"], data["lons"], per_year, ctx.grid))
