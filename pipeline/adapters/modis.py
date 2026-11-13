"""NDVI and day-night temperature from MODIS, read from the Planetary Computer's open COG mirror.

No account is needed: the catalog gives a short-lived read token for its storage. The same NASA
products come from NASA Earthdata with a login, so an event adapter can replace this one directly.
Overviews are read instead of full tiles, which is about nine times faster at this grid size.
"""

import math
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, date, datetime, timedelta
from urllib.parse import urlparse

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.errors import RasterioIOError
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window

from pipeline import http_setup  # noqa: F401
from pipeline.adapters import Context, Layer, cached_arrays
from pipeline.grid import block_nanmean
from pipeline.net import fetch

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
TOKEN_URL = "https://planetarycomputer.microsoft.com/api/sas/v1/token/{account}/{container}"
SUBCELLS = 5  # read at a fifth of a cell
MONTHS = (1, 3, 5, 7, 9, 11)  # six composites across the year
WORKERS = 12
TILE_ATTEMPTS = 3
NDVI_OVERVIEW = 2  # factor 8 of 463 m pixels, about 3.7 km
LST_OVERVIEW = 1  # factor 4 of 927 m pixels, about 3.7 km

_tokens: dict[tuple[str, str], tuple[str, datetime]] = {}
_tokens_lock = threading.Lock()


def _signed(href: str) -> str:
    """Add a read token. Tokens last under an hour, so they are renewed during long runs."""
    parts = urlparse(href)
    key = (parts.netloc.split(".")[0], parts.path.lstrip("/").split("/")[0])
    with _tokens_lock:
        token, expiry = _tokens.get(key, ("", datetime.min.replace(tzinfo=UTC)))
        if datetime.now(UTC) + timedelta(minutes=3) >= expiry:
            url = TOKEN_URL.format(account=key[0], container=key[1])
            body = fetch(url).json()
            token = body["token"]
            expiry = datetime.fromisoformat(body["msft:expiry"].replace("Z", "+00:00"))
            _tokens[key] = (token, expiry)
    return f"{href}?{token}"


def composite_start(item_id: str) -> date:
    """The start date in a MODIS product ID, for example MOD13A1.A2023017.h11v10 is 2023 day 17."""
    stamp = item_id.split(".")[1]
    return datetime.strptime(stamp[1:], "%Y%j").date()


def _search(body: dict) -> list[dict]:
    """Run a catalog search, following its pages."""
    items: list[dict] = []
    payload = body
    while True:
        result = fetch(STAC_URL, method="POST", json=payload).json()
        items.extend(result["features"])
        next_link = next((x for x in result.get("links", []) if x["rel"] == "next"), None)
        if next_link is None:
            return items
        payload = next_link["body"]


def _items_for_month(
    collection: str, platform: str, bbox: list, year: int, month: int
) -> list[dict]:
    """The earliest composite that starts in this month, one item per tile.

    The search also returns composites from earlier months that overlap the date window.
    """
    items = _search(
        {
            "collections": [collection],
            "bbox": bbox,
            "datetime": f"{year}-{month:02d}-01/{year}-{month:02d}-20",
            "query": {"platform": {"eq": platform}},
            "limit": 500,
        }
    )
    starts = {item["id"]: composite_start(item["id"]) for item in items}
    in_month = [i for i in items if (starts[i["id"]].year, starts[i["id"]].month) == (year, month)]
    if not in_month:
        raise RuntimeError(f"no {collection} items found for {year}-{month:02d}")
    earliest = min(starts[item["id"]] for item in in_month)
    return [item for item in in_month if starts[item["id"]] == earliest]


def _read_mosaic(
    items: list[dict], asset: str, ctx: Context, scale: float, fill: int, overview: int
) -> np.ndarray:
    """Warp one band from each tile onto the fine grid. Returns cell means, latitude ascending."""
    grid = ctx.grid
    fine = grid.res / SUBCELLS
    west = float(grid.lons[0] - grid.res / 2)
    north = float(grid.lats[-1] + grid.res / 2)
    shape = (grid.shape[0] * SUBCELLS, grid.shape[1] * SUBCELLS)
    mosaic = np.full(shape, np.nan, dtype=np.float32)

    def read(item: dict):
        # Only ask for the part of the grid this tile covers.
        left, bottom, right, top = item["bbox"]
        col0 = max(0, math.floor((left - west) / fine) - 1)
        col1 = min(shape[1], math.ceil((right - west) / fine) + 1)
        row0 = max(0, math.floor((north - top) / fine) - 1)
        row1 = min(shape[0], math.ceil((north - bottom) / fine) + 1)
        if col1 <= col0 or row1 <= row0:
            return None
        window = Window(col0, row0, col1 - col0, row1 - row0)
        for attempt in range(TILE_ATTEMPTS):
            try:
                url = _signed(item["assets"][asset]["href"])
                with (
                    rasterio.open(url, overview_level=overview) as src,
                    WarpedVRT(
                        src,
                        crs="EPSG:4326",
                        transform=from_origin(west, north, fine, fine),
                        width=shape[1],
                        height=shape[0],
                        resampling=Resampling.average,
                        src_nodata=fill,
                        nodata=fill,
                    ) as vrt,
                ):
                    return row0, col0, vrt.read(1, window=window)
            except RasterioIOError as error:
                if attempt == TILE_ATTEMPTS - 1:
                    # One lost tile becomes a gap that lowers confidence, not a failed build.
                    print(f"    skipped {item['id']}: {str(error)[:80]}", flush=True)
        return None

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for result in pool.map(read, items):
            if result is None:
                continue
            row0, col0, data = result
            valid = data != fill
            target = mosaic[row0 : row0 + data.shape[0], col0 : col0 + data.shape[1]]
            target[valid] = data[valid] * scale
    return block_nanmean(mosaic, SUBCELLS)[::-1]


def _bbox(ctx: Context) -> list[float]:
    grid = ctx.grid
    half = grid.res / 2
    return [
        float(max(grid.lons[0] - half, -180.0)),
        float(max(grid.lats[0] - half, -90.0)),
        float(min(grid.lons[-1] + half, 180.0)),
        float(min(grid.lats[-1] + half, 90.0)),
    ]


def _ndvi_month(ctx: Context, month: int) -> np.ndarray:
    def build() -> dict[str, np.ndarray]:
        items = _items_for_month("modis-13A1-061", "terra", _bbox(ctx), ctx.reference_year, month)
        print(f"    NDVI month {month}: {len(items)} tiles", flush=True)
        return {
            "values": _read_mosaic(items, "500m_16_days_NDVI", ctx, 0.0001, -3000, NDVI_OVERVIEW)
        }

    return cached_arrays(ctx, f"modis_ndvi_{ctx.reference_year}_{month:02d}", build)["values"]


def _diurnal_month(ctx: Context, month: int) -> np.ndarray:
    def build() -> dict[str, np.ndarray]:
        items = _items_for_month("modis-11A2-061", "aqua", _bbox(ctx), ctx.reference_year, month)
        print(f"    temperature month {month}: {len(items)} tiles", flush=True)
        day = _read_mosaic(items, "LST_Day_1km", ctx, 0.02, 0, LST_OVERVIEW)
        night = _read_mosaic(items, "LST_Night_1km", ctx, 0.02, 0, LST_OVERVIEW)
        return {"values": day - night}

    return cached_arrays(ctx, f"modis_lst_{ctx.reference_year}_{month:02d}", build)["values"]


def _stack(ctx: Context, read_month) -> tuple[np.ndarray, np.ndarray]:
    """Return the monthly stack, and the share of months with no valid data for each cell."""
    stack = np.stack([read_month(ctx, month) for month in MONTHS])
    valid = (~np.isnan(stack)).sum(axis=0)
    return stack, 1.0 - valid / len(MONTHS)


def vegetation(ctx: Context) -> Layer:
    """Maximum NDVI across the sampled composites."""
    stack, gaps = _stack(ctx, _ndvi_month)
    filled = np.where(np.isnan(stack), -np.inf, stack).max(axis=0)
    return Layer(values=np.where(gaps < 1.0, filled, np.nan), gap_filled=gaps)


def temp_diurnal_range(ctx: Context) -> Layer:
    """Mean day minus night land surface temperature, in kelvin."""
    stack, gaps = _stack(ctx, _diurnal_month)
    valid = (~np.isnan(stack)).sum(axis=0)
    totals = np.nan_to_num(stack).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return Layer(values=np.where(valid > 0, totals / valid, np.nan), gap_filled=gaps)
